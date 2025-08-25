from pathlib import Path

import safetensors
import torch
import torchaudio
from tqdm.auto import tqdm

from src.lensless.pipeline import reconstruct_codec
from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer
from src.utils.io_utils import ROOT_PATH


class Inferencer(BaseTrainer):
    """
    Inferencer (Like Trainer but for Inference) class

    The class is used to process data without
    the need of optimizers, writers, etc.
    Required to evaluate the model on the dataset, save predictions, etc.
    """

    def __init__(
        self,
        model,
        codec,
        config,
        device,
        dataloaders,
        save_path,
        dataset_tag,
        model_tag,
        metrics=None,
        batch_transforms=None,
        skip_model_load=False,
    ):
        """
        Initialize the Inferencer.

        Args:
            model (nn.Module): PyTorch model.
            codec (nn.Module): Audio codec.
            config (DictConfig): run config containing inferencer config.
            device (str): device for tensors and model.
            dataloaders (dict[DataLoader]): dataloaders for different
                sets of data.
            save_path (str): path to save model predictions and other
                information.
            dataset_tag (str): subdir name set to dataset tag for saving.
            model_tag (str): subdir name set to model tag for saving.
            metrics (dict): dict with the definition of metrics for
                inference (metrics[inference]). Each metric is an instance
                of src.metrics.BaseMetric.
            batch_transforms (dict[nn.Module] | None): transforms that
                should be applied on the whole batch. Depend on the
                tensor name.
            skip_model_load (bool): if False, require the user to set
                pre-trained checkpoint path. Set this argument to True if
                the model desirable weights are defined outside of the
                Inferencer Class.
        """
        assert (
            skip_model_load or config.inferencer.get("from_pretrained") is not None
        ), "Provide checkpoint or set skip_model_load=True"

        self.config = config
        self.cfg_trainer = self.config.inferencer

        self.device = device

        self.model = model
        self.codec = codec
        self.batch_transforms = batch_transforms

        # define dataloaders
        self.evaluation_dataloaders = {k: v for k, v in dataloaders.items()}

        # path definition

        self.save_path = save_path / dataset_tag
        self.model_tag = model_tag

        # define metrics
        self.metrics = metrics
        if self.metrics is not None:
            self.evaluation_metrics = MetricTracker(
                *[m.name for m in self.metrics["inference"]],
                writer=None,
            )
        else:
            self.evaluation_metrics = MetricTracker()
            self.metrics = {"inference": []}

        if not skip_model_load:
            # init model
            checkpoint_dir = ROOT_PATH / "data" / "lensless_exps" / model_tag
            checkpoint_path = checkpoint_dir / config.inferencer["from_pretrained"]
            self._from_pretrained(checkpoint_path)

    def run_inference(self, calculate_metrics=False):
        """
        Run inference on each partition.

        Args:
            calculate_metrics (bool): if True, calculate metrics
                instead of running the model.
        Returns:
            part_logs (dict): part_logs[part_name] contains logs
                for the part_name partition.
        """
        part_logs = {}
        for part, dataloader in self.evaluation_dataloaders.items():
            logs = self._inference_part(part, dataloader, calculate_metrics)
            part_logs[part] = logs
        return part_logs

    def process_batch(self, batch_idx, batch, metrics, part):
        """
        Run batch through the model, compute metrics, and
        save predictions to disk.

        Save directory is defined by save_path in the inference
        config and current partition.

        Args:
            batch_idx (int): the index of the current batch.
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type
                of the partition (train or inference).
            part (str): name of the partition. Used to define proper saving
                directory.
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform)
                and model outputs.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        recon_codec_video = reconstruct_codec(
            recon_model=self.model,
            return_raw=False,
            **self.config.reconstruction,
            **batch,
        )
        recon_audio, recon_codes = self.codec.video_to_audio(
            recon_codec_video, return_codes=True
        )
        outputs = {
            "recon_codec_video": recon_codec_video,
            # "raw_recon_codec_video": raw_recon_codec_video,
            "recon_audio": recon_audio,
            "recon_codes": recon_codes,
        }
        batch.update(outputs)

        # codec_audio, codec_codes = self.codec.video_to_audio(
        #     batch["lensed_codec_video"], return_codes=True
        # )
        # batch.update({"codec_audio": codec_audio, "codec_codes": codec_codes})

        if metrics is not None:
            for met in self.metrics["inference"]:
                metrics.update(met.name, met(**batch))

        # Some saving logic. This is an example
        # Use if you need to save predictions on disk

        batch_size = batch["recon_codec_video"].shape[0]

        for i in range(batch_size):
            # clone because of
            # https://github.com/pytorch/pytorch/issues/1995
            recon_codec_video = batch["recon_codec_video"][i].clone()
            recon_codes = batch["recon_codes"][i].clone()
            recon_audio = batch["recon_audio"][i].clone()
            # peak-normalize to avoid clipping
            recon_audio = recon_audio / recon_audio.abs().max()

            audio_path = Path(batch["audio_path"][i]).resolve()

            output = {
                "recon_codec_video": recon_codec_video.detach().cpu(),
                "recon_codes": recon_codes.detach().cpu(),
            }
            filename = audio_path.stem

            if self.save_path is not None:
                # you can use safetensors or other lib here
                save_dir = self.save_path / part / self.model_tag
                safetensors.torch.save_file(
                    output, save_dir / "lensed" / f"{filename}.safetensors"
                )
                sr = self.codec.codec.metadata["kwargs"]["sample_rate"]
                torchaudio.save(
                    save_dir / "audio" / f"{filename}.wav",
                    recon_audio.detach().cpu(),
                    sample_rate=sr,
                )

        return batch

    def _inference_part(self, part, dataloader, calculate_metrics):
        """
        Run inference on a given partition and save predictions

        Args:
            part (str): name of the partition.
            dataloader (DataLoader): dataloader for the given partition.
            calculate_metrics (bool): if True, calculate metrics
                instead of running the model. Model outputs are pre-saved.
        Returns:
            logs (dict): metrics, calculated on the partition.
        """

        self.is_train = False

        if not calculate_metrics:
            self.model.eval()

        self.evaluation_metrics.reset()

        # create Save dir
        if self.save_path is not None:
            save_dir = self.save_path / part / self.model_tag
            (save_dir / "lensed").mkdir(exist_ok=True, parents=True)
            (save_dir / "audio").mkdir(exist_ok=True, parents=True)

        with torch.no_grad():
            for batch_idx, batch in tqdm(
                enumerate(dataloader),
                desc=part,
                total=len(dataloader),
            ):
                if calculate_metrics:
                    # only calculate metrics
                    # the model output is already saved
                    self.calculate_metrics(
                        batch_idx=batch_idx,
                        batch=batch,
                        part=part,
                        metrics=self.evaluation_metrics,
                    )
                else:
                    # normal inference, run the model
                    batch = self.process_batch(
                        batch_idx=batch_idx,
                        batch=batch,
                        part=part,
                        metrics=self.evaluation_metrics,
                    )

        return self.evaluation_metrics.result()

    def calculate_metrics(self, batch_idx, batch, metrics, part):
        """
        Compute metrics, the model outputs are pre-saved.

        Args:
            batch_idx (int): the index of the current batch.
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type
                of the partition (train or inference).
            part (str): name of the partition. Used to define proper saving
                directory.
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform)
                and model outputs.
        """
        batch_size = batch["lensed_codec_video"].shape[0]

        outputs = []

        for i in range(batch_size):
            audio_path = Path(batch["audio_path"][i]).resolve()
            filename = audio_path.stem

            if self.save_path is not None:
                # you can use safetensors or other lib here
                save_dir = self.save_path / part / self.model_tag
                recon_data = safetensors.torch.load_file(
                    save_dir / "lensed" / f"{filename}.safetensors"
                )
                target_sr = self.codec.codec.metadata["kwargs"]["sample_rate"]
                recon_audio, sr = torchaudio.load(
                    save_dir / "audio" / f"{filename}.wav",
                )
                recon_data["recon_audio"] = recon_audio
                assert target_sr == sr, "Codec and audio sample rate mismatch."
                outputs.append(recon_data)

        for k in ["recon_audio", "recon_codec_video", "recon_codes"]:
            batch[k] = torch.stack([outputs[i][k] for i in range(batch_size)], dim=0)

        codec_audio, codec_codes = self.codec.video_to_audio(
            batch["lensed_codec_video"], return_codes=True
        )
        batch.update({"codec_audio": codec_audio, "codec_codes": codec_codes})

        if metrics is not None:
            for met in self.metrics["inference"]:
                metrics.update(met.name, met(**batch))

        return batch
