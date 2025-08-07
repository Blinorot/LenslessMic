import torch
from torchmetrics.functional.audio import (
    scale_invariant_signal_distortion_ratio,
    short_time_objective_intelligibility,
)
from torchmetrics.functional.text import word_error_rate as wer

from src.metrics.base_metric import BaseMetric
from src.metrics.wer_utils import init_asr_model, run_asr_model


class SISDRMetric(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metric = scale_invariant_signal_distortion_ratio

    def __call__(self, codec_audio: torch.Tensor, recon_audio: torch.Tensor, **kwargs):
        return self.metric(recon_audio.detach(), codec_audio.detach()).item()


class STOIMetric(BaseMetric):
    def __init__(self, *args, sampling_rate=16000, **kwargs):
        super().__init__(*args, **kwargs)
        self.metric = short_time_objective_intelligibility
        self.sampling_rate = sampling_rate

    def __call__(self, codec_audio: torch.Tensor, recon_audio: torch.Tensor, **kwargs):
        return self.metric(
            recon_audio.detach(), codec_audio.detach(), fs=self.sampling_rate
        ).item()


class WERMetric(BaseMetric):
    def __init__(self, *args, model_id="openai/whisper-tiny", device="cpu", **kwargs):
        super().__init__(*args, **kwargs)
        self.asr_pipeline = init_asr_model(model_id=model_id, device=device)

    def __call__(self, text, recon_audio, **kwargs):
        recon_text = run_asr_model(
            self.asr_pipeline, recon_audio.detach().cpu(), normalize=True
        )
        return wer(recon_text, text).item()
