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
from src.lensless.utils import normalize_video
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
        codec=None,
        roi_kwargs=None,
        resize_coef=1,
        psf_path=None,
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
            lensless_tag (str | None): tag for saved lensless codec img
            codec (nn.Module | None): audio codec from src.transforms.
            roi_kwargs (dict | None): top_left, height, and width for ROI.
            resize_coef (int): the scaling factor for resize.
            psf_path (str | None): path to .npy array with saved PSF.
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

        self.lensless_tag = lensless_tag
        self.codec = codec
        self.roi_kwargs = roi_kwargs
        self.resize_coef = resize_coef

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
        audio = self.load_audio(audio_path)
        text = data_dict["text"]

        instance_data = {
            "audio": audio,
            "text": text,
            "audio_path": audio_path,
        }

        if self.lensless_tag is not None:
            lensless_dict = self.load_lensless(audio_path)
            instance_data.update(**lensless_dict)

        instance_data = self.preprocess_data(instance_data)

        return instance_data

    def __len__(self):
        """
        Get length of the dataset (length of the index).
        """
        return len(self._index)

    def load_lensless(self, audio_path):
        suffix = Path(audio_path).suffix
        lensless_path = audio_path[: -(len(suffix))] + f"_{self.lensless_tag}.mkv"
        min_vals_path = (
            audio_path[: -(len(suffix))] + f"_{self.lensless_tag}_min_vals.pth"
        )
        max_vals_path = (
            audio_path[: -(len(suffix))] + f"_{self.lensless_tag}_max_vals.pth"
        )

        min_vals = torch.load(min_vals_path, map_location="cpu")
        max_vals = torch.load(max_vals_path, map_location="cpu")

        lensless_codec_video = load_grayscale_video_ffv1(str(lensless_path))

        lensless_codec_video = torch.from_numpy(lensless_codec_video)

        lensless_codec_video = lensless_codec_video.to(torch.float32)
        lensless_codec_video = lensless_codec_video / 255

        lensless_codec_video = lensless_codec_video.permute(1, 2, 0)  # HxWxT
        lensless_codec_video = lensless_codec_video.unsqueeze(0)  # add plane
        lensless_codec_video = lensless_codec_video.unsqueeze(-2)  # add channel

        final_lensless_dict = {
            "lensless_codec_video": lensless_codec_video,
            "min_vals": min_vals,
            "max_vals": max_vals,
        }

        return final_lensless_dict

    def load_audio(self, path):
        audio_tensor, sr = torchaudio.load(path)
        audio_tensor = audio_tensor[0:1, :]  # remove all channels but the first
        target_sr = self.target_sr
        if sr != target_sr:
            audio_tensor = torchaudio.functional.resample(audio_tensor, sr, target_sr)
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

        for elem in tqdm(self._index, desc="Saving as video"):
            audio_path = elem["audio_path"]
            audio = self.load_audio(audio_path)
            # peak-normalization
            audio = audio / audio.abs().max()

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

            suffix = Path(audio_path).suffix
            video_path = audio_path[: -(len(suffix))] + "_codec.mkv"
            min_vals_path = audio_path[: -(len(suffix))] + "_min_vals.pth"
            max_vals_path = audio_path[: -(len(suffix))] + "_max_vals.pth"

            save_grayscale_video_ffv1(codec_video.clone().numpy(), str(video_path))

            torch.save(min_vals_list, min_vals_path)
            torch.save(max_vals_list, max_vals_path)

    def simulate_lensless(
        self,
        lensless_tag,
        psf_path,
        codec,
        roi_kwargs=None,
        resize_coef=1,
        normalize=True,
        normalize_dims=(0, 4),
    ):
        """
        Simulate lensless codec video given PSF, codec,
        and alignment hyperparameters.

        Uses given arguments instead of self.codec, etc.
        to create new simulations.

        Args:
            lensless_tag (str): tag for saving lensless codec.
            psf_path (str): path to .npy array with saved PSF.
            codec (nn.Module): audio codec from src.transforms.
            roi_kwargs (dict | None): top_left, height, and width for ROI.
            resize_coef (int): the scaling factor for resize.
            normalize (bool): whether to rescale lensless output via peak-normalization.
            normalize_dims (int | tuple): dims for normalization. Use 0 for Batch-only.
                Use (0, 4) for batch and channel-wise normalization.
        """
        codec_sample_rate = codec.codec.metadata["kwargs"]["sample_rate"]
        assert self.target_sr == codec_sample_rate, "Sample rate mismatch."

        psf = torch.from_numpy(np.load(psf_path))

        for elem in tqdm(self._index, desc="Simulating"):
            audio_path = elem["audio_path"]
            audio = self.load_audio(audio_path)

            codec_video = codec.audio_to_video(audio.unsqueeze(0)).detach()

            lensless_codec_video, _, min_vals, max_vals = simulate_lensless_codec(
                codec_video,
                psf,
                roi_kwargs,
                resize_coef,
                return_min_max_values=True,
                normalize=normalize,
                normalize_dims=normalize_dims,
            )

            lensless_codec_video = lensless_codec_video.squeeze()  # (H, W, T)
            lensless_codec_video = lensless_codec_video.permute(2, 0, 1)
            lensless_codec_video = lensless_codec_video.contiguous()

            lensless_codec_video = (lensless_codec_video * 255).to(torch.uint8)

            suffix = Path(audio_path).suffix
            lensless_path = audio_path[: -(len(suffix))] + f"_{lensless_tag}.mkv"
            min_vals_path = (
                audio_path[: -(len(suffix))] + f"_{lensless_tag}_min_vals.pth"
            )
            max_vals_path = (
                audio_path[: -(len(suffix))] + f"_{lensless_tag}_max_vals.pth"
            )

            save_grayscale_video_ffv1(
                lensless_codec_video.clone().numpy(), str(lensless_path)
            )

            torch.save(min_vals, min_vals_path)
            torch.save(max_vals, max_vals_path)
