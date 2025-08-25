import torch
from piq import gmsd, psnr, ssim
from torchmetrics.functional.audio import scale_invariant_signal_distortion_ratio

from src.metrics.base_metric import BaseMetric
from src.transforms.normalize import MinMaxNormalize


class CodecMetric(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.normalizer = MinMaxNormalize(dim=0)

    def prepare_video_for_metric(self, video):
        video = video.detach()
        video = video.permute(0, 5, 1, 4, 2, 3).contiguous()
        B, T, D, C, H, W = video.shape
        video = video.reshape(B * T * D, C, H, W)
        video = self.normalizer(video)
        return video


class SSIMMetric(CodecMetric):
    def __init__(self, *args, ssim_kernel=3, ssim_sigma=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.ssim_kernel = ssim_kernel
        self.ssim_sigma = ssim_sigma

    def __call__(self, lensed_codec_video, recon_codec_video, **kwargs):
        normalized_lensed_codec_video = self.prepare_video_for_metric(
            lensed_codec_video
        )
        normalized_recon_codec_video = self.prepare_video_for_metric(recon_codec_video)

        return ssim(
            normalized_lensed_codec_video,
            normalized_recon_codec_video,
            kernel_size=self.ssim_kernel,
            kernel_sigma=self.ssim_sigma,
        ).item()


class GMSDMetric(CodecMetric):
    def __call__(self, lensed_codec_video, recon_codec_video, **kwargs):
        normalized_lensed_codec_video = self.prepare_video_for_metric(
            lensed_codec_video
        )
        normalized_recon_codec_video = self.prepare_video_for_metric(recon_codec_video)

        return gmsd(normalized_lensed_codec_video, normalized_recon_codec_video).item()


class PSNRMetric(CodecMetric):
    def __call__(self, lensed_codec_video, recon_codec_video, **kwargs):
        normalized_lensed_codec_video = self.prepare_video_for_metric(
            lensed_codec_video
        )
        normalized_recon_codec_video = self.prepare_video_for_metric(recon_codec_video)

        return psnr(normalized_lensed_codec_video, normalized_recon_codec_video).item()


class MSEMetric(CodecMetric):
    def __init__(self, normalized, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.normalized = normalized

    def __call__(self, lensed_codec_video, recon_codec_video, **kwargs):
        recon = recon_codec_video.detach()
        lensed = lensed_codec_video.detach()

        if self.normalized:
            recon = self.prepare_video_for_metric(recon)
            lensed = self.prepare_video_for_metric(lensed)

        return torch.nn.functional.mse_loss(recon, lensed).item()


class QuantizationMatchMetric(CodecMetric):
    def __init__(self, codebook_index, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.codebook_index = codebook_index

    def __call__(self, codec_codes, recon_codes, **kwargs):
        recon = recon_codes.detach()
        lensed = codec_codes.detach()

        if self.codebook_index == "all":
            # element-wise accuracy
            result = (recon == lensed).to(torch.float32).mean()
        else:
            # exact match up to self.codebook_index
            batch_size = recon.shape[0]
            recon = recon[:, : self.codebook_index].reshape(batch_size, -1)
            lensed = lensed[:, : self.codebook_index].reshape(batch_size, -1)
            result = (recon == lensed).all(dim=1).to(torch.float32).mean()

        return result.item()
