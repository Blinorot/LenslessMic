import torch
from numpy import inf

from src.datasets.data_utils import inf_loop
from src.lensless.pipeline import reconstruct_codec
from src.logger.utils import plot_images
from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer
from src.utils.io_utils import ROOT_PATH


class TrainerGAN(BaseTrainer):
    """
    Trainer class for GAN-losses. Defines the logic of batch logging and processing.
    """

    def __init__(
        self,
        model,
        discriminator,
        codec,
        criterion,
        metrics,
        optimizer_G,
        optimizer_D,
        lr_scheduler_G,
        lr_scheduler_D,
        config,
        device,
        dataloaders,
        logger,
        writer,
        epoch_len=None,
        skip_oom=True,
        batch_transforms=None,
    ):
        """
        Args:
            model (nn.Module): PyTorch model.
            discriminator (nn.Module): Discriminator.
            codec (nn.Module): Audio codec.
            criterion (nn.Module): loss function for model training.
            metrics (dict): dict with the definition of metrics for training
                (metrics[train]) and inference (metrics[inference]). Each
                metric is an instance of src.metrics.BaseMetric.
            optimizer_G (Optimizer): optimizer for the model.
            optimizer_D (Optimizer): optimizer for the discriminator.
            lr_scheduler_G (LRScheduler): learning rate scheduler for the
                optimizer_G.
            lr_scheduler_D (LRScheduler): learning rate scheduler for the
                optimizer_D.
            config (DictConfig): experiment config containing training config.
            device (str): device for tensors and model.
            dataloaders (dict[DataLoader]): dataloaders for different
                sets of data.
            logger (Logger): logger that logs output.
            writer (WandBWriter | CometMLWriter): experiment tracker.
            epoch_len (int | None): number of steps in each epoch for
                iteration-based training. If None, use epoch-based
                training (len(dataloader)).
            skip_oom (bool): skip batches with the OutOfMemory error.
            batch_transforms (dict[Callable] | None): transforms that
                should be applied on the whole batch. Depend on the
                tensor name.
        """
        self.is_train = True

        self.config = config
        self.cfg_trainer = self.config.trainer

        self.device = device
        self.skip_oom = skip_oom

        self.logger = logger
        self.log_step = config.trainer.get("log_step", 50)

        self.model = model
        self.discriminator = discriminator
        self.codec = codec
        self.criterion = criterion
        self.optimizer_G = optimizer_G
        self.optimizer_D = optimizer_D
        self.lr_scheduler = lr_scheduler_G
        self.lr_scheduler_D = lr_scheduler_D
        self.batch_transforms = batch_transforms

        # define dataloaders
        self.train_dataloader = dataloaders["train"]
        if epoch_len is None:
            # epoch-based training
            self.epoch_len = len(self.train_dataloader)
        else:
            # iteration-based training
            self.train_dataloader = inf_loop(self.train_dataloader)
            self.epoch_len = epoch_len

        self.evaluation_dataloaders = {
            k: v for k, v in dataloaders.items() if k != "train"
        }

        # define epochs
        self._last_epoch = 0  # required for saving on interruption
        self.start_epoch = 1
        self.epochs = self.cfg_trainer.n_epochs

        # configuration to monitor model performance and save best

        self.save_period = (
            self.cfg_trainer.save_period
        )  # checkpoint each save_period epochs
        self.monitor = self.cfg_trainer.get(
            "monitor", "off"
        )  # format: "mnt_mode mnt_metric"

        if self.monitor == "off":
            self.mnt_mode = "off"
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ["min", "max"]

            self.mnt_best = inf if self.mnt_mode == "min" else -inf
            self.early_stop = self.cfg_trainer.get("early_stop", inf)
            if self.early_stop <= 0:
                self.early_stop = inf

        # setup visualization writer instance
        self.writer = writer

        # define metrics
        self.metrics = metrics
        self.train_metrics = MetricTracker(
            *self.config.writer.loss_names,
            "grad_norm",
            *[m.name for m in self.metrics["train"]],
            writer=self.writer,
        )
        self.evaluation_metrics = MetricTracker(
            *self.config.writer.loss_names,
            *[m.name for m in self.metrics["inference"]],
            writer=self.writer,
        )

        # define checkpoint dir and init everything if required

        self.checkpoint_dir = (
            ROOT_PATH / config.trainer.save_dir / config.writer.run_name
        )

        if config.trainer.get("resume_from") is not None:
            resume_path = self.checkpoint_dir / config.trainer.resume_from
            self._resume_checkpoint(resume_path)

        if config.trainer.get("from_pretrained") is not None:
            self._from_pretrained(config.trainer.get("from_pretrained"))

    def process_batch(self, batch, metrics: MetricTracker):
        """
        Run batch through the model, compute metrics, compute loss,
        and do training step (during training stage).

        The function expects that criterion aggregates all losses
        (if there are many) into a single one defined in the 'loss' key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type of
                the partition (train or inference).
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform),
                model outputs, and losses.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]

        recon_codec_video, raw_recon_codec_video = reconstruct_codec(
            recon_model=self.model,
            return_raw=True,
            **self.config.reconstruction,
            **batch,
        )

        if self.cfg_trainer.get("filter_padded_frames", False):
            # assumes batch_size == 1
            mask_T = recon_codec_video.any(dim=(0, 1, 2, 3, 4))
            recon_codec_video = recon_codec_video[..., mask_T]
            lensed_codec_video = batch["lensed_codec_video"]
            mask_T = lensed_codec_video.any(dim=(0, 1, 2, 3, 4))
            lensed_codec_video = lensed_codec_video[..., mask_T]
            batch["lensed_codec_video"] = lensed_codec_video

        recon_audio, recon_codes = self.codec.video_to_audio(
            recon_codec_video, return_codes=True
        )
        outputs = {
            "recon_codec_video": recon_codec_video,
            "raw_recon_codec_video": raw_recon_codec_video,
            "recon_audio": recon_audio,
            "recon_codes": recon_codes,
        }
        batch.update(outputs)

        codec_audio, codec_codes = self.codec.video_to_audio(
            batch["lensed_codec_video"], return_codes=True
        )
        batch.update({"codec_audio": codec_audio, "codec_codes": codec_codes})

        if self.is_train:
            discriminator_outputs = self.discriminator(
                recon_audio=batch["recon_audio"].detach(),
                codec_audio=batch["codec_audio"],
            )
            batch.update(discriminator_outputs)

            d_loss = self.criterion.discriminator_loss(**batch)
            batch.update(d_loss)
            self.optimizer_D.zero_grad()
            batch["d_loss"].backward()
            if self.lr_scheduler_D is not None:
                self.lr_scheduler_D.step()

            self.optimizer_G.zero_grad()

        discriminator_outputs = self.discriminator(
            recon_audio=batch["recon_audio"], codec_audio=batch["codec_audio"]
        )
        batch.update(discriminator_outputs)

        all_losses = self.criterion(**batch)
        batch.update(all_losses)

        if self.is_train:
            batch["loss"].backward()  # sum of all losses is always called loss

            if self.cfg_trainer.get("skip_NaN", False) and self._isnan_grad():
                # skip this batch as there is a NaN gradient
                print("NaN detected, skipping step...")
                return batch

            self._clip_grad_norm()
            self.optimizer_G.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        # update metrics for each loss (in case of multiple losses)
        for loss_name in self.config.writer.loss_names:
            metrics.update(loss_name, batch[loss_name].item())

        for met in metric_funcs:
            metrics.update(met.name, met(**batch))
        return batch

    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        # method to log data from you batch
        # such as audio, text or images, for example

        # logging scheme might be different for different partitions
        if mode == "train":  # the method is called only every self.log_step steps
            # Log Stuff
            self.log_frame(**batch)
            self.log_audio(**batch)
        else:
            # Log Stuff
            self.log_frame(**batch)
            self.log_audio(**batch)

    def log_audio(self, audio, codec_audio, recon_audio, **batch):
        audio_example = audio[0].detach().cpu()
        codec_audio_example = codec_audio[0].detach().cpu()
        recon_audio_example = recon_audio[0].detach().cpu()
        sr = self.config.writer.sample_rate
        self.writer.add_audio("audio", audio_example, sample_rate=sr)
        self.writer.add_audio("codec_audio", codec_audio_example, sample_rate=sr)
        self.writer.add_audio("recon_audio", recon_audio_example, sample_rate=sr)

    def log_frame(
        self, lensed_codec_video, lensless_codec_video, recon_codec_video, **batch
    ):
        lensed_frames = lensed_codec_video[:, 0, :, :, :, 0].detach().cpu()
        recon_frames = recon_codec_video[:, 0, :, :, :, 0].detach().cpu()
        lensless_frames = lensless_codec_video[:, 0, :, :, :, 0].detach().cpu()
        lensed_img = plot_images(lensed_frames, self.config, permute=False)
        recon_img = plot_images(recon_frames, self.config, permute=False)
        lensless_img = plot_images(lensless_frames, self.config, permute=False)
        self.writer.add_image("lensed_frames", lensed_img)
        self.writer.add_image("recon_frames", recon_img)
        self.writer.add_image("lensless_frames", lensless_img)

    def _save_checkpoint(self, epoch, save_best=False, only_best=False):
        """
        Save the checkpoints.

        Args:
            epoch (int): current epoch number.
            save_best (bool): if True, rename the saved checkpoint to 'model_best.pth'.
            only_best (bool): if True and the checkpoint is the best, save it only as
                'model_best.pth'(do not duplicate the checkpoint as
                checkpoint-epochEpochNumber.pth)
        """
        arch = type(self.model).__name__
        state = {
            "arch": arch,
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "state_dict_D": self.discriminator.state_dict(),
            "optimizer_G": self.optimizer_G.state_dict(),
            "lr_scheduler_G": self.lr_scheduler.state_dict(),
            "optimizer_D": self.optimizer_D.state_dict(),
            "lr_scheduler_D": self.lr_scheduler_D.state_dict(),
            "monitor_best": self.mnt_best,
            "config": self.config,
        }
        filename = str(self.checkpoint_dir / f"checkpoint-epoch{epoch}.pth")
        if not (only_best and save_best):
            torch.save(state, filename)
            if self.config.writer.log_checkpoints:
                self.writer.add_checkpoint(filename, str(self.checkpoint_dir.parent))
            self.logger.info(f"Saving checkpoint: {filename} ...")
        if save_best:
            best_path = str(self.checkpoint_dir / "model_best.pth")
            torch.save(state, best_path)
            if self.config.writer.log_checkpoints:
                self.writer.add_checkpoint(best_path, str(self.checkpoint_dir.parent))
            self.logger.info("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path):
        """
        Resume from a saved checkpoint (in case of server crash, etc.).
        The function loads state dicts for everything, including model,
        optimizers, etc.

        Notice that the checkpoint should be located in the current experiment
        saved directory (where all checkpoints are saved in '_save_checkpoint').

        Args:
            resume_path (str): Path to the checkpoint to be resumed.
        """
        resume_path = str(resume_path)
        self.logger.info(f"Loading checkpoint: {resume_path} ...")
        checkpoint = torch.load(resume_path, self.device)
        self.start_epoch = checkpoint["epoch"] + 1
        self.mnt_best = checkpoint["monitor_best"]

        # load architecture params from checkpoint.
        if (
            checkpoint["config"]["model"] != self.config["model"]
            or checkpoint["config"]["discriminator"] != self.config["discriminator"]
        ):
            self.logger.warning(
                "Warning: Architecture configuration given in the config file is different from that "
                "of the checkpoint. This may yield an exception when state_dict is loaded."
            )
        self.model.load_state_dict(checkpoint["state_dict"])
        self.discriminator.load_state_dict(checkpoint["state_dict_D"])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if (
            checkpoint["config"]["optimizer_G"] != self.config["optimizer_G"]
            or checkpoint["config"]["lr_scheduler_G"] != self.config["lr_scheduler_G"]
            or checkpoint["config"]["optimizer_D"] != self.config["optimizer_D"]
            or checkpoint["config"]["lr_scheduler_D"] != self.config["lr_scheduler_D"]
        ):
            self.logger.warning(
                "Warning: Optimizer or lr_scheduler given in the config file is different "
                "from that of the checkpoint. Optimizer and scheduler parameters "
                "are not resumed."
            )
        else:
            self.optimizer_G.load_state_dict(checkpoint["optimizer_G"])
            self.optimizer_D.load_state_dict(checkpoint["optimizer_D"])
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler_G"])
            self.lr_scheduler_D.load_state_dict(checkpoint["lr_scheduler_D"])

        self.logger.info(
            f"Checkpoint loaded. Resume training from epoch {self.start_epoch}"
        )

    def _from_pretrained(self, pretrained_path):
        """
        Init model with weights from pretrained pth file.

        Notice that 'pretrained_path' can be any path on the disk. It is not
        necessary to locate it in the experiment saved dir. The function
        initializes only the model.

        Args:
            pretrained_path (str): path to the model state dict.
        """
        pretrained_path = str(pretrained_path)
        if hasattr(self, "logger"):  # to support both trainer and inferencer
            self.logger.info(f"Loading model weights from: {pretrained_path} ...")
        else:
            print(f"Loading model weights from: {pretrained_path} ...")
        checkpoint = torch.load(pretrained_path, self.device)

        if checkpoint.get("state_dict") is not None:
            self.model.load_state_dict(checkpoint["state_dict"])
        else:
            self.model.load_state_dict(checkpoint)
