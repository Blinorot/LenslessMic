import torch
from torch import nn

from src.loss.snr import PairwiseNegSDR


class ReconstructionLoss(nn.Module):
    """
    Reconstruction loss for lensless images.
    """

    def __init__(self, codec_mse_coef=1, audio_l1_coef=1, audio_snr_coef=1):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.snr_loss = PairwiseNegSDR(sdr_type="snr")

        self.codec_mse_coef = codec_mse_coef
        self.audio_l1_coef = audio_l1_coef
        self.audio_snr_coef = audio_snr_coef

    def forward(
        self, lensed_codec_video, recon_codec_video, codec_audio, recon_audio, **batch
    ):
        codec_mse_loss = self.mse_loss(recon_codec_video, lensed_codec_video)
        audio_l1_loss = self.l1_loss(recon_audio, codec_audio)
        audio_snr_loss = self.snr_loss(recon_audio[:, 0, :], codec_audio[:, 0, :])
        loss = (
            self.codec_mse_coef * codec_mse_loss
            + self.audio_l1_coef * audio_l1_loss
            + self.audio_snr_coef * audio_snr_loss
        )
        return {
            "loss": loss,
            "codec_mse_loss": codec_mse_loss,
            "audio_l1_loss": audio_l1_loss,
            "audio_snr_loss": audio_snr_loss,
        }
