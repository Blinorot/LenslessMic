import torch
from torchmetrics.functional.audio import (
    perceptual_evaluation_speech_quality,
    short_time_objective_intelligibility,
)
from torchmetrics.functional.text import word_error_rate as wer

from src.loss.sisdr import SISDRLoss
from src.loss.timefreqloss import MelSpectrogramLoss, MultiScaleSTFTLoss
from src.metrics.base_metric import BaseMetric
from src.metrics.wer_utils import init_asr_model, run_asr_model


class SISDRMetric(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metric = SISDRLoss()

    def __call__(self, codec_audio: torch.Tensor, recon_audio: torch.Tensor, **kwargs):
        result = -self.metric(
            references=codec_audio.detach(), estimates=recon_audio.detach()
        )
        return result.item()


class MelMetric(BaseMetric):
    def __init__(self, audio_mel_config={}, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metric = MelSpectrogramLoss(**audio_mel_config)

    def __call__(self, codec_audio: torch.Tensor, recon_audio: torch.Tensor, **kwargs):
        return self.metric(recon_audio.detach(), codec_audio.detach())


class STFTMetric(BaseMetric):
    def __init__(self, audio_stft_config={}, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metric = MultiScaleSTFTLoss(**audio_stft_config)

    def __call__(self, codec_audio: torch.Tensor, recon_audio: torch.Tensor, **kwargs):
        return self.metric(recon_audio.detach(), codec_audio.detach())


class STOIMetric(BaseMetric):
    def __init__(self, *args, sampling_rate=16000, **kwargs):
        super().__init__(*args, **kwargs)
        self.metric = short_time_objective_intelligibility
        self.sampling_rate = sampling_rate

    def __call__(self, audio: torch.Tensor, recon_audio: torch.Tensor, **kwargs):
        return self.metric(
            recon_audio.detach(),
            audio[..., : recon_audio.shape[-1]].detach(),
            fs=self.sampling_rate,
        ).item()


class PESQMetric(BaseMetric):
    def __init__(self, *args, sampling_rate=16000, mode="wb", **kwargs):
        super().__init__(*args, **kwargs)
        self.metric = perceptual_evaluation_speech_quality
        self.sampling_rate = sampling_rate
        self.mode = mode

    def __call__(self, audio: torch.Tensor, recon_audio: torch.Tensor, **kwargs):
        return self.metric(
            recon_audio.detach(),
            audio[..., : recon_audio.shape[-1]].detach(),
            fs=self.sampling_rate,
            mode=self.mode,
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
