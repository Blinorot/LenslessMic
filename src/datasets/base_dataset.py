import logging
import random
from pathlib import Path

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from src.datasets.data_utils import load_grayscale_video_ffv1, save_grayscale_video_ffv1
from src.lensless.simulate import simulate_lensless_codec
from src.lensless.utils import normalize_video, simulate_psf_from_mask
from src.transforms import MinMaxNormalize

logger = logging.getLogger(__name__)


DEFAULT_TRANSFORM = {"audio": MinMaxNormalize(min=0, dim=0)}


class BaseDataset(Dataset):
    """
    Base class for the datasets.

    Given a proper index (list[dict]), allows to process different datasets
    for the same task in the identical manner. Therefore, to work with
    several datasets, the user only have to define index in a nested class.
    """

    def __init__(
        self,
        index,
        target_sr=16000,
        limit=None,
        max_audio_length=None,
        max_text_length=None,
        shuffle_index=False,
        instance_transforms=DEFAULT_TRANSFORM,
        lensless_tag=None,
        codec_name=None,
        codec=None,
        roi_kwargs=None,
        sim_psf_config=None,
        psf_path=None,
        randomize_psf_percent=0,
        randomize_psf_seed=55,
        replace_min_val=None,
        replace_max_val=None,
    ):
        """
        Args:
            index (list[dict]): list, containing dict for each element of
                the dataset. The dict has required metadata information,
                such as label and object path.
            target_sr (int): supported sample rate.
            limit (int | None): if not None, limit the total number of elements
                in the dataset to 'limit' elements.
            max_audio_length (int | None): maximum allowed audio length.
            max_test_length (int | None): maximum allowed text length.
            shuffle_index (bool): if True, shuffle the index. Uses python
                random package with seed 42.
            instance_transforms (dict[Callable] | None): transforms that
                should be applied on the instance. Depend on the
                tensor name.
            lensless_tag (str | None): tag for saved lensless codec img.
            codec_name (str | None): audio codec.codec_name from src.transforms.
            codec (nn.Module | None): audio codec itself.
                Shall have the same codec_name.
            roi_kwargs (dict | None): top_left, height, and width for ROI.
            sim_psf_config (dict | None): config for simulating PSF.
            psf_path (str | None): path to psf (use for single-mask dataset).
            randomize_psf_percent (int): the percentage of the PSF that
                should be replaced with random values.
            randomize_psf_seed (int): random seed for the PSF randomizer.
            replace_min_val (None | float): if float, replaces min val value.
            replace_max_val (None | float): if float, replaces max val value.
        """
        self._assert_index_is_valid(index)

        index = self._filter_records_from_dataset(
            index, max_audio_length, max_text_length
        )
        index = self._shuffle_and_limit_index(index, limit, shuffle_index)
        if not shuffle_index:
            index = self._sort_index(index)

        self._index: list[dict] = index

        self.target_sr = target_sr
        self.instance_transforms = instance_transforms

        # lensless
        self.psf = None
        if psf_path is not None:
            self.psf = torch.from_numpy(np.load(psf_path))
        self.computed_psfs = {}

        self.sim_psf_config = sim_psf_config
        self.randomize_psf_percent = randomize_psf_percent
        self.randomize_psf_gen = np.random.default_rng(randomize_psf_seed)

        self.lensless_tag = lensless_tag
        self.codec_name = codec_name
        self.codec = codec
        self.roi_kwargs = roi_kwargs
        self.replace_min_val = replace_min_val
        self.replace_max_val = replace_max_val

        if self.codec is not None:
            if self.codec_name is None:
                self.codec_name = self.codec.codec_name
            else:
                names_match = self.codec_name == self.codec.codec_name
                assert names_match, "Codec name shall be the same as codec.codec_name"

    def __getitem__(self, ind):
        """
        Get element from the index, preprocess it, and combine it
        into a dict.

        Notice that the choice of key names is defined by the template user.
        However, they should be consistent across dataset getitem, collate_fn,
        loss_function forward method, and model forward method.

        Args:
            ind (int): index in the self.index list.
        Returns:
            instance_data (dict): dict, containing instance
                (a single dataset element).
        """
        data_dict = self._index[ind]
        audio_path = data_dict["audio_path"]
        if audio_path == "":  # special case for random dataset
            audio = torch.zeros(self.dummy_audio_length)
        else:
            audio = self.load_audio(audio_path)
        text = data_dict["text"]

        instance_data = {
            "audio": audio,
            "text": text,
            "audio_path": audio_path,
        }

        if self.lensless_tag is not None and self.codec is not None:
            lensless_dict = self.load_lensless(audio, **data_dict)
            instance_data.update(**lensless_dict)

        instance_data = self.preprocess_data(instance_data)

        return instance_data

    def __len__(self):
        """
        Get length of the dataset (length of the index).
        """
        return len(self._index)

    def prepare_codec_video(self, codec_video_path):
        codec_video = load_grayscale_video_ffv1(str(codec_video_path))

        codec_video = torch.from_numpy(codec_video)

        codec_video = codec_video.to(torch.float32)
        codec_video = codec_video / 255

        codec_video = codec_video.permute(1, 2, 0).contiguous()  # HxWxT
        codec_video = codec_video.unsqueeze(0)  # add plane
        codec_video = codec_video.unsqueeze(-2)  # add channel
        return codec_video

    def load_lensless(self, audio, audio_path, **data_dict):
        if audio_path == "":
            lensed_codec_video_path = Path(data_dict["lensed_codec_video_path"])
            lensed_codec_video = self.prepare_codec_video(lensed_codec_video_path)

            codec_length = lensed_codec_video.shape[-1]
            min_vals = torch.zeros((1, 1, 1, 1, 1, codec_length))
            max_vals = torch.ones((1, 1, 1, 1, 1, codec_length))

            filename = lensed_codec_video_path.stem
            lensless_video_dir = (
                lensed_codec_video_path.parents[1]  # codec_name is built-in
                / f"lensless_{self.lensless_tag}"
            )
        else:
            audio_path = Path(audio_path)
            filename = audio_path.stem
            video_dir = audio_path.parents[1] / f"{self.codec_name}" / "lensed"
            lensless_video_dir = (
                audio_path.parents[1]
                / f"{self.codec_name}"
                / f"lensless_{self.lensless_tag}"
            )

            # saved lensed is normalized
            # we want non-normalized lensed
            with torch.no_grad():
                lensed_codec_video = self.codec.audio_to_video(audio.unsqueeze(0))[0]

            min_vals_path = video_dir / f"{filename}_min_vals.pth"
            max_vals_path = video_dir / f"{filename}_max_vals.pth"

            min_vals_list = torch.load(min_vals_path, map_location="cpu")
            min_vals = torch.stack(min_vals_list, dim=-1)
            max_vals_list = torch.load(max_vals_path, map_location="cpu")
            max_vals = torch.stack(max_vals_list, dim=-1)

        if self.replace_max_val is not None:
            max_vals = torch.ones_like(max_vals) * self.replace_max_val
        if self.replace_min_val is not None:
            min_vals = torch.ones_like(min_vals) * self.replace_min_val

        lensless_path = lensless_video_dir / f"{filename}.mkv"
        lensless_mask_label = lensless_video_dir / f"{filename}.txt"
        if lensless_mask_label.exists():
            lensless_mask_label = lensless_mask_label.read_text()

            if self.randomize_psf_percent > 0:
                lensless_mask_path = (
                    lensless_video_dir / "masks" / f"mask_{lensless_mask_label}.npy"
                )
                lensless_mask = np.load(lensless_mask_path)
                noisy_mask = lensless_mask.copy()
                n_pixels = noisy_mask.size
                n_wrong_pixels = int(n_pixels * self.randomize_psf_percent / 100)
                wrong_pixels = self.randomize_psf_gen.choice(
                    n_pixels, n_wrong_pixels, replace=False
                )
                noisy_mask = noisy_mask.flatten()
                noisy_mask[wrong_pixels] = self.randomize_psf_gen.uniform(
                    size=n_wrong_pixels
                )
                noisy_mask = noisy_mask.reshape(lensless_mask.shape)
                lensless_psf = simulate_psf_from_mask(noisy_mask, **self.sim_psf_config)
            else:
                if self.computed_psfs.get(lensless_mask_label) is not None:
                    lensless_psf = self.computed_psfs[lensless_mask_label]
                else:
                    # compute psf only once
                    lensless_mask_path = (
                        lensless_video_dir / "masks" / f"mask_{lensless_mask_label}.npy"
                    )
                    lensless_mask = np.load(lensless_mask_path)
                    lensless_psf = simulate_psf_from_mask(
                        lensless_mask, **self.sim_psf_config
                    )
                    self.computed_psfs[lensless_mask_label] = lensless_psf.clone()
        else:
            lensless_psf = self.psf.clone()

        lensless_codec_video = self.prepare_codec_video(lensless_path)

        pad_mask = torch.zeros_like(lensed_codec_video)
        if "group" in self.lensless_tag:
            # measurement_group_nrows_ncols_rowspace_colspace
            n_rows = int(self.lensless_tag.split("_")[2])
            n_cols = int(self.lensless_tag.split("_")[3])
            n_frames = n_rows * n_cols
            n_diff = pad_mask.shape[-1] % n_frames
            if n_diff != 0:
                # mark padded regions as 1
                padded_pad_mask = torch.ones_like(pad_mask[..., :1])
                padded_pad_mask = padded_pad_mask.repeat(1, 1, 1, 1, n_frames - n_diff)
                pad_mask = torch.cat([pad_mask, padded_pad_mask], dim=-1)

        final_lensless_dict = {
            "lensed_codec_video": lensed_codec_video,
            "lensless_codec_video": lensless_codec_video,
            "lensless_psf": lensless_psf,
            "min_vals": min_vals,
            "max_vals": max_vals,
            "pad_mask": pad_mask,
            "n_orig_frames": lensed_codec_video.shape[-1],
        }

        return final_lensless_dict

    def load_audio(self, path):
        audio_tensor, sr = torchaudio.load(path)
        audio_tensor = audio_tensor[0:1, :]  # remove all channels but the first
        target_sr = self.target_sr
        if sr != target_sr:
            audio_tensor = torchaudio.functional.resample(audio_tensor, sr, target_sr)

        # peak-normalization
        audio_tensor = audio_tensor / audio_tensor.abs().max()

        return audio_tensor

    def get_spectrogram(self, audio):
        """
        Special instance transform with a special key to
        get spectrogram from audio.

        Args:
            audio (Tensor): original audio.
        Returns:
            spectrogram (Tensor): spectrogram for the audio.
        """
        return self.instance_transforms["get_spectrogram"](audio)

    def preprocess_data(self, instance_data):
        """
        Preprocess data with instance transforms.

        Each tensor in a dict undergoes its own transform defined by the key.

        Args:
            instance_data (dict): dict, containing instance
                (a single dataset element).
        Returns:
            instance_data (dict): dict, containing instance
                (a single dataset element) (possibly transformed via
                instance transform).
        """
        if self.instance_transforms is not None:
            for transform_name in self.instance_transforms.keys():
                if transform_name == "get_spectrogram":
                    continue  # skip special key
                elif transform_name == "all":
                    instance_data = self.instance_transforms[transform_name](
                        instance_data
                    )
                else:
                    instance_data[transform_name] = self.instance_transforms[
                        transform_name
                    ](instance_data[transform_name])
        return instance_data

    @staticmethod
    def _filter_records_from_dataset(
        index: list,
        max_audio_length,
        max_text_length,
    ) -> list:
        """
        Filter some of the elements from the dataset depending on
        the desired max_test_length or max_audio_length.

        Args:
            index (list[dict]): list, containing dict for each element of
                the dataset. The dict has required metadata information,
                such as label and object path.
            max_audio_length (int): maximum allowed audio length.
            max_test_length (int): maximum allowed text length.
        Returns:
            index (list[dict]): list, containing dict for each element of
                the dataset that satisfied the condition. The dict has
                required metadata information, such as label and object path.
        """
        initial_size = len(index)
        if max_audio_length is not None:
            exceeds_audio_length = (
                np.array([el["audio_len"] for el in index]) >= max_audio_length
            )
            _total = exceeds_audio_length.sum()
            logger.info(
                f"{_total} ({_total / initial_size:.1%}) records are longer then "
                f"{max_audio_length} seconds. Excluding them."
            )
        else:
            exceeds_audio_length = False

        initial_size = len(index)
        if max_text_length is not None:
            exceeds_text_length = (
                np.array([len(el["text"]) for el in index]) >= max_text_length
            )
            _total = exceeds_text_length.sum()
            logger.info(
                f"{_total} ({_total / initial_size:.1%}) records are longer then "
                f"{max_text_length} characters. Excluding them."
            )
        else:
            exceeds_text_length = False

        records_to_filter = exceeds_text_length | exceeds_audio_length

        if records_to_filter is not False and records_to_filter.any():
            _total = records_to_filter.sum()
            index = [el for el, exclude in zip(index, records_to_filter) if not exclude]
            logger.info(
                f"Filtered {_total} ({_total / initial_size:.1%}) records  from dataset"
            )

        return index

    @staticmethod
    def _assert_index_is_valid(index):
        """
        Check the structure of the index and ensure it satisfies the desired
        conditions.

        Args:
            index (list[dict]): list, containing dict for each element of
                the dataset. The dict has required metadata information,
                such as label and object path.
        """
        for entry in index:
            assert "audio_path" in entry, (
                "Each dataset item should include field 'path'" " - path to audio file."
            )
            assert "text" in entry, (
                "Each dataset item should include field 'text'"
                " - object ground-truth transcription."
            )
            assert "audio_len" in entry, (
                "Each dataset item should include field 'audio_len'"
                " - length of the audio."
            )

    @staticmethod
    def _sort_index(index):
        """
        Sort index by audio length.

        Args:
            index (list[dict]): list, containing dict for each element of
                the dataset. The dict has required metadata information,
                such as label and object path.
        Returns:
            index (list[dict]): sorted list, containing dict for each element
                of the dataset. The dict has required metadata information,
                such as label and object path.
        """
        return sorted(index, key=lambda x: x["audio_len"])

    @staticmethod
    def _shuffle_and_limit_index(index, limit, shuffle_index):
        """
        Shuffle elements in index and limit the total number of elements.

        Args:
            index (list[dict]): list, containing dict for each element of
                the dataset. The dict has required metadata information,
                such as label and object path.
            limit (int | None): if not None, limit the total number of elements
                in the dataset to 'limit' elements.
            shuffle_index (bool): if True, shuffle the index. Uses python
                random package with seed 42.
        """
        if shuffle_index:
            random.seed(42)
            random.shuffle(index)

        if limit is not None:
            index = index[:limit]
        return index

    def save_in_video_format(
        self, codec, min_vals=None, max_vals=None, normalize_dims=(0, 4)
    ):
        """
        Convert audio to codec video, normalize and save.

        Args:
            codec (nn.Module): audio codec from src.transforms.
            min_vals (float | Tensor): min_vals for original video.
            max_vals (float | Tensor): max_vals for original video.
            normalize_dims (int | tuple): dims for normalization. Use 0 for Batch-only.
                Use (0, 4) for batch and channel-wise normalization.
        """
        codec_sample_rate = codec.codec.metadata["kwargs"]["sample_rate"]
        assert self.target_sr == codec_sample_rate, "Sample rate mismatch."

        example_path = Path(self._index[0]["audio_path"])
        video_dir = example_path.parents[1] / f"{codec.codec_name}" / "lensed"
        video_dir.mkdir(exist_ok=True, parents=True)

        for elem in tqdm(self._index, desc="Saving as video"):
            audio_path = elem["audio_path"]
            audio = self.load_audio(audio_path)
            # peak-normalization
            audio = audio / audio.abs().max()

            with torch.no_grad():
                codec_video = codec.audio_to_video(audio.unsqueeze(0)).detach()

            codec_video, min_vals_list, max_vals_list = normalize_video(
                codec_video,
                min_vals=min_vals,
                max_vals=max_vals,
                return_min_max_values=True,
                normalize_dims=normalize_dims,
            )

            codec_video = codec_video.squeeze()  # (H, W, T)
            codec_video = codec_video.permute(2, 0, 1)
            codec_video = codec_video.contiguous()

            codec_video = (codec_video * 255).to(torch.uint8)

            audio_path = Path(audio_path)
            filename = audio_path.stem
            video_path = video_dir / f"{filename}.mkv"
            min_vals_path = video_dir / f"{filename}_min_vals.pth"
            max_vals_path = video_dir / f"{filename}_max_vals.pth"

            save_grayscale_video_ffv1(codec_video.clone().numpy(), str(video_path))

            torch.save(min_vals_list, min_vals_path)
            torch.save(max_vals_list, max_vals_path)

    def save_in_audio_format(self, codec):
        """
        Convert audio to codec video, then back to audio and save.

        Args:
            codec (nn.Module): audio codec from src.transforms.
        """
        codec_sample_rate = codec.codec.metadata["kwargs"]["sample_rate"]
        assert self.target_sr == codec_sample_rate, "Sample rate mismatch."

        example_path = Path(self._index[0]["audio_path"])
        audio_dir = example_path.parents[1] / f"{codec.codec_name}" / "codec_audio"
        audio_dir.mkdir(exist_ok=True, parents=True)

        for elem in tqdm(self._index, desc="Saving as audio"):
            audio_path = elem["audio_path"]
            audio = self.load_audio(audio_path)
            # peak-normalization
            audio = audio / audio.abs().max()

            with torch.no_grad():
                codec_video = codec.audio_to_video(audio.unsqueeze(0)).detach()
                codec_audio = codec.video_to_audio(codec_video)[0].detach().clone()
            audio_path = Path(audio_path)
            codec_audio_path = audio_dir / (audio_path.stem + ".wav")
            codec_audio = codec_audio / codec_audio.abs().max()
            torchaudio.save(codec_audio_path, codec_audio, sample_rate=self.target_sr)

    def simulate_lensless(
        self,
        lensless_tag,
        psf_path,
        codec,
        roi_kwargs=None,
        resize_coef=1,
        normalize=True,
        normalize_dims=(0, 4),
        group_frames_kwargs=None,
        patchify_video_kwargs=None,
    ):
        """
        Simulate lensless codec video given PSF, codec,
        and alignment hyperparameters.

        Args:
            lensless_tag (str): tag for saving lensless codec.
            psf_path (str): path to .npy array with saved PSF.
            codec (nn.Module): audio codec from src.transforms.
            roi_kwargs (dict | None): top_left, height, and width for ROI.
            resize_coef (int): the scaling factor for resize.
            normalize (bool): whether to rescale lensless output via peak-normalization.
            normalize_dims (int | tuple): dims for normalization. Use 0 for Batch-only.
                Use (0, 4) for batch and channel-wise normalization.
            group_frames_kwargs (dict | None): configuration for group_frames function.
                See src.lensless.utils.group_frames. Ignored if None.
            patchify_video_kwargs (dict | None): configuration for patchify_video function.
                See src.lensless.utils.patchify_video. Ignored if None.
        """
        codec_sample_rate = codec.codec.metadata["kwargs"]["sample_rate"]
        assert self.target_sr == codec_sample_rate, "Sample rate mismatch."

        psf = torch.from_numpy(np.load(psf_path))

        example_path = Path(self._index[0]["audio_path"])
        video_dir = example_path.parents[1] / f"{codec.codec_name}" / "lensed"
        video_dir.mkdir(exist_ok=True, parents=True)
        lensless_video_dir = (
            example_path.parents[1] / f"{codec.codec_name}" / f"lensless_{lensless_tag}"
        )
        lensless_video_dir.mkdir(exist_ok=True, parents=True)

        for elem in tqdm(self._index, desc="Simulating"):
            audio_path = elem["audio_path"]
            audio = self.load_audio(audio_path)

            with torch.no_grad():
                codec_video = codec.audio_to_video(audio.unsqueeze(0)).detach()

            lensless_codec_video, _, min_vals, max_vals = simulate_lensless_codec(
                codec_video,
                psf,
                roi_kwargs,
                resize_coef,
                return_min_max_values=True,
                normalize=normalize,
                normalize_dims=normalize_dims,
                group_frames_kwargs=group_frames_kwargs,
                patchify_video_kwargs=patchify_video_kwargs,
            )

            lensless_codec_video = lensless_codec_video.squeeze()  # (H, W, T)
            lensless_codec_video = lensless_codec_video.permute(2, 0, 1)
            lensless_codec_video = lensless_codec_video.contiguous()

            lensless_codec_video = (lensless_codec_video * 255).to(torch.uint8)

            filename = Path(audio_path).stem
            lensless_path = lensless_video_dir / f"{filename}.mkv"
            min_vals_path = video_dir / f"{filename}_min_vals.pth"
            max_vals_path = video_dir / f"{filename}_max_vals.pth"

            save_grayscale_video_ffv1(
                lensless_codec_video.clone().numpy(), str(lensless_path)
            )

            torch.save(min_vals, min_vals_path)
            torch.save(max_vals, max_vals_path)
