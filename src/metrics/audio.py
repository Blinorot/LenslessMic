import torch
from torchmetrics.functional.audio import scale_invariant_signal_distortion_ratio

from src.metrics.base_metric import BaseMetric


class SISDRMetric(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metric = scale_invariant_signal_distortion_ratio

    def __call__(self, codec_audio: torch.Tensor, recon_audio: torch.Tensor, **kwargs):
        return self.metric(recon_audio, codec_audio)
