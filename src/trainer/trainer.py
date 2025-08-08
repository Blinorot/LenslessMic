from src.lensless.pipeline import reconstruct_codec
from src.logger.utils import plot_images
from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """

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
            self.optimizer.zero_grad()

        recon_codec_video, raw_recon_codec_video = reconstruct_codec(
            recon_model=self.model,
            return_raw=True,
            **self.config.reconstruction,
            **batch
        )
        recon_audio = self.codec.video_to_audio(recon_codec_video)
        outputs = {
            "recon_codec_video": recon_codec_video,
            "raw_recon_codec_video": raw_recon_codec_video,
            "recon_audio": recon_audio,
        }
        batch.update(outputs)

        batch["codec_audio"] = self.codec.video_to_audio(batch["lensed_codec_video"])

        all_losses = self.criterion(**batch)
        batch.update(all_losses)

        if self.is_train:
            batch["loss"].backward()  # sum of all losses is always called loss
            self._clip_grad_norm()
            self.optimizer.step()
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
