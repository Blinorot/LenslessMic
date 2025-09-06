import torch
from piq import GMSDLoss, SSIMLoss
from torch import nn

from src.lensless.utils import group_frames
from src.loss.sisdr import SISDRLoss
from src.loss.timefreqloss import MelSpectrogramLoss, MultiScaleSTFTLoss
from src.transforms import MinMaxNormalize


class ReconstructionGANLoss(nn.Module):
    """
    Reconstruction loss for lensless images.
    """

    def __init__(
        self,
        codec_mse_coef=1,
        codec_ssim_coef=1,
        codec_gmsd_coef=0,
        raw_codec_ssim_coef=1,
        raw_codec_l1_coef=1,
        audio_l1_coef=1,
        audio_sisdr_coef=0,
        audio_stft_coef=0,
        audio_mel_coef=0,
        gan_coef=1,
        gan_fm_coef=1,
        ssim_kernel=3,
        ssim_sigma=0.5,
        raw_ssim_kernel=7,
        raw_ssim_sigma=1.0,
        resize_coef=16,
        group_frames_kwargs=None,
        audio_stft_config={},
        audio_mel_config={},
    ):
        super().__init__()
        # codec
        self.mse_loss = nn.MSELoss()
        self.ssim_loss = SSIMLoss(
            kernel_size=ssim_kernel, kernel_sigma=ssim_sigma, data_range=1.0
        )
        self.gmsd_loss = GMSDLoss()

        # raw codec
        self.raw_ssim_loss = SSIMLoss(
            kernel_size=raw_ssim_kernel, kernel_sigma=raw_ssim_sigma, data_range=1.0
        )
        self.raw_l1_loss = nn.L1Loss()

        # audio
        self.l1_loss = nn.L1Loss()
        self.sisdr_loss = SISDRLoss()
        self.stft_loss = MultiScaleSTFTLoss(**audio_stft_config)
        self.mel_loss = MelSpectrogramLoss(**audio_mel_config)

        self.codec_mse_coef = codec_mse_coef
        self.codec_ssim_coef = codec_ssim_coef
        self.codec_gmsd_coef = codec_gmsd_coef
        self.raw_codec_ssim_coef = raw_codec_ssim_coef
        self.raw_codec_l1_coef = raw_codec_l1_coef
        self.audio_l1_coef = audio_l1_coef
        self.audio_sisdr_coef = audio_sisdr_coef
        self.audio_stft_coef = audio_stft_coef
        self.audio_mel_coef = audio_mel_coef

        # GAN
        self.gan_coef = gan_coef
        self.gan_fm_coef = gan_fm_coef

        self.resize_coef = resize_coef
        self.group_frames_kwargs = group_frames_kwargs

        # video is B x D x H x W x C x T
        self.normalizer = MinMaxNormalize(dim=0)  # across batches
        self.group_normalizer = MinMaxNormalize(dim=(0, 4))

    def forward(
        self,
        lensed_codec_video,
        recon_codec_video,
        raw_recon_codec_video,
        codec_audio,
        recon_audio,
        d_fmaps_recon,
        d_fmaps_codec,
        **batch
    ):
        # codec losses

        # normalize video to stabilize training
        # merge batch and T
        normalized_lensed_codec_video = self.prepare_video_for_loss(lensed_codec_video)
        normalized_recon_codec_video = self.prepare_video_for_loss(recon_codec_video)

        if self.codec_mse_coef > 0:
            codec_mse_loss = self.mse_loss(
                normalized_recon_codec_video, normalized_lensed_codec_video
            )
        else:
            codec_mse_loss = torch.tensor(0, device=recon_codec_video.device)

        if self.codec_ssim_coef > 0:
            codec_ssim_loss = self.ssim_loss(
                normalized_recon_codec_video, normalized_lensed_codec_video
            )
        else:
            codec_ssim_loss = torch.tensor(0, device=codec_mse_loss.device)

        if self.codec_gmsd_coef > 0:
            codec_gmsd_loss = self.gmsd_loss(
                normalized_recon_codec_video, normalized_lensed_codec_video
            )
        else:
            codec_gmsd_loss = torch.tensor(0, device=codec_mse_loss.device)

        # raw codec losses
        normalized_raw_recon_codec_video = self.prepare_video_for_loss(
            raw_recon_codec_video
        )
        if self.group_frames_kwargs is not None:
            normalized_raw_lensed_codec_video = self.group_normalizer(
                lensed_codec_video
            )
            normalized_raw_lensed_codec_video = group_frames(
                normalized_raw_lensed_codec_video, **self.group_frames_kwargs
            )
            normalized_raw_lensed_codec_video = self.prepare_video_for_loss(
                normalized_raw_lensed_codec_video
            )
        else:
            normalized_raw_lensed_codec_video = normalized_lensed_codec_video

        if self.resize_coef > 1:
            normalized_raw_lensed_codec_video = nn.functional.interpolate(
                normalized_raw_lensed_codec_video,
                scale_factor=self.resize_coef,
                mode="nearest",
            )

        if self.raw_codec_ssim_coef > 0:
            raw_codec_ssim_loss = self.raw_ssim_loss(
                normalized_raw_recon_codec_video, normalized_raw_lensed_codec_video
            )
        else:
            raw_codec_ssim_loss = torch.tensor(0, device=codec_mse_loss.device)

        if self.raw_codec_l1_coef > 0:
            raw_codec_l1_loss = self.raw_l1_loss(
                normalized_raw_recon_codec_video, normalized_raw_lensed_codec_video
            )
        else:
            raw_codec_l1_loss = torch.tensor(0, device=codec_mse_loss.device)

        # audio losses

        if self.audio_l1_coef > 0:
            audio_l1_loss = self.l1_loss(recon_audio, codec_audio)
        else:
            audio_l1_loss = torch.tensor(0, device=codec_mse_loss.device)

        if self.audio_stft_coef > 0:
            audio_stft_loss = self.stft_loss(recon_audio, codec_audio)
        else:
            audio_stft_loss = torch.tensor(0, device=codec_mse_loss.device)

        if self.audio_mel_coef > 0:
            audio_mel_loss = self.mel_loss(recon_audio, codec_audio)
        else:
            audio_mel_loss = torch.tensor(0, device=codec_mse_loss.device)

        if self.audio_sisdr_coef > 0:
            audio_sisdr_loss = self.sisdr_loss(
                references=codec_audio, estimates=recon_audio
            )
        else:
            audio_sisdr_loss = torch.tensor(0, device=codec_mse_loss.device)

        # GAN loss
        gan_loss = 0
        for x_fake in d_fmaps_recon:
            gan_loss += torch.mean((1 - x_fake[-1]) ** 2)

        gan_fm_loss = 0

        for i in range(len(d_fmaps_recon)):
            for j in range(len(d_fmaps_recon[i]) - 1):
                gan_fm_loss += nn.functional.l1_loss(
                    d_fmaps_recon[i][j], d_fmaps_codec[i][j].detach()
                )

        # total loss

        loss = (
            self.codec_mse_coef * codec_mse_loss
            + self.codec_ssim_coef * codec_ssim_loss
            + self.codec_gmsd_coef * codec_gmsd_loss
            + self.raw_codec_ssim_coef * raw_codec_ssim_loss
            + self.raw_codec_l1_coef * raw_codec_l1_loss
            + self.audio_l1_coef * audio_l1_loss
            + self.audio_sisdr_coef * audio_sisdr_loss
            + self.audio_stft_coef * audio_stft_loss
            + self.audio_mel_coef * audio_mel_loss
            + self.gan_coef * gan_loss
            + self.gan_fm_coef * gan_fm_loss
        )
        return {
            "loss": loss,
            "codec_mse_loss": codec_mse_loss,
            "codec_ssim_loss": codec_ssim_loss,
            "codec_gmsd_loss": codec_gmsd_loss,
            "raw_codec_ssim_loss": raw_codec_ssim_loss,
            "raw_codec_l1_loss": raw_codec_l1_loss,
            "audio_l1_loss": audio_l1_loss,
            "audio_sisdr_loss": audio_sisdr_loss,
            "audio_stft_loss": audio_stft_loss,
            "audio_mel_loss": audio_mel_loss,
            "gan_loss": gan_loss,
            "gan_fm_loss": gan_fm_loss,
        }

    def prepare_video_for_loss(self, video):
        video = video.permute(0, 5, 1, 4, 2, 3).contiguous()
        B, T, D, C, H, W = video.shape
        video = video.reshape(B * T * D, C, H, W)
        video = self.normalizer(video)
        return video

    def discriminator_loss(self, d_fmaps_recon, d_fmaps_codec, **kwargs):
        d_loss = 0
        for x_fake, x_real in zip(d_fmaps_recon, d_fmaps_codec):
            d_loss += torch.mean(x_fake[-1] ** 2)
            d_loss += torch.mean((1 - x_real[-1]) ** 2)
        return {"d_loss": d_loss}
