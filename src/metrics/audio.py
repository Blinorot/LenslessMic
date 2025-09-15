import torch
import torchaudio
from torchmetrics.functional.audio import (
    perceptual_evaluation_speech_quality,
    short_time_objective_intelligibility,
)
from torchmetrics.functional.text import word_error_rate as wer

from src.loss.sisdr import SISDRLoss
from src.loss.timefreqloss import MelSpectrogramLoss, MultiScaleSTFTLoss
from src.metrics.base_metric import BaseMetric
from src.metrics.visqol_utils import calc_visqol
from src.metrics.wer_utils import init_asr_model, run_asr_model


class SISDRMetric(BaseMetric):
    def __init__(self, version="codec_recon", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metric = SISDRLoss()
        self.version = version

    def __call__(
        self,
        codec_audio: torch.Tensor,
        recon_audio: torch.Tensor,
        audio: torch.Tensor,
        **kwargs
    ):
        if self.version == "codec_recon":
            references = codec_audio.detach()
            estimates = recon_audio.detach()
        elif self.version == "audio_recon":
            min_length = min(audio.shape[-1], recon_audio.shape[-1])
            references = audio[..., :min_length].detach()
            estimates = recon_audio[..., :min_length].detach()
        elif self.version == "audio_codec":
            min_length = min(audio.shape[-1], codec_audio.shape[-1])
            references = audio[..., :min_length].detach()
            estimates = codec_audio[..., :min_length].detach()
        else:
            raise NotImplementedError()
        result = -self.metric(references=references, estimates=estimates)
        return result.item()


class MelMetric(BaseMetric):
    def __init__(self, audio_mel_config={}, version="codec_recon", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metric = MelSpectrogramLoss(**audio_mel_config)
        self.version = version

    def __call__(
        self,
        codec_audio: torch.Tensor,
        recon_audio: torch.Tensor,
        audio: torch.Tensor,
        **kwargs
    ):
        if self.version == "codec_recon":
            reference = codec_audio.detach()
            estimate = recon_audio.detach()
        elif self.version == "audio_recon":
            min_length = min(audio.shape[-1], recon_audio.shape[-1])
            reference = audio[..., :min_length].detach()
            estimate = recon_audio[..., :min_length].detach()
        elif self.version == "audio_codec":
            min_length = min(audio.shape[-1], codec_audio.shape[-1])
            reference = audio[..., :min_length].detach()
            estimate = codec_audio[..., :min_length].detach()
        else:
            raise NotImplementedError()
        result = self.metric(estimate, reference)
        return result.item()


class STFTMetric(BaseMetric):
    def __init__(self, audio_stft_config={}, version="codec_recon", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metric = MultiScaleSTFTLoss(**audio_stft_config)
        self.version = version

    def __call__(
        self,
        codec_audio: torch.Tensor,
        recon_audio: torch.Tensor,
        audio: torch.Tensor,
        **kwargs
    ):
        if self.version == "codec_recon":
            reference = codec_audio.detach()
            estimate = recon_audio.detach()
        elif self.version == "audio_recon":
            min_length = min(audio.shape[-1], recon_audio.shape[-1])
            reference = audio[..., :min_length].detach()
            estimate = recon_audio[..., :min_length].detach()
        elif self.version == "audio_codec":
            min_length = min(audio.shape[-1], codec_audio.shape[-1])
            reference = audio[..., :min_length].detach()
            estimate = codec_audio[..., :min_length].detach()
        else:
            raise NotImplementedError()
        result = self.metric(estimate, reference)
        return result.item()


class STOIMetric(BaseMetric):
    def __init__(self, *args, sampling_rate=16000, version="audio_recon", **kwargs):
        super().__init__(*args, **kwargs)
        self.metric = short_time_objective_intelligibility
        self.sampling_rate = sampling_rate
        self.version = version

    def __call__(
        self,
        codec_audio: torch.Tensor,
        recon_audio: torch.Tensor,
        audio: torch.Tensor,
        **kwargs
    ):
        if self.version == "codec_recon":
            reference = codec_audio.detach()
            estimate = recon_audio.detach()
        elif self.version == "audio_recon":
            min_length = min(audio.shape[-1], recon_audio.shape[-1])
            reference = audio[..., :min_length].detach()
            estimate = recon_audio[..., :min_length].detach()
        elif self.version == "audio_codec":
            min_length = min(audio.shape[-1], codec_audio.shape[-1])
            reference = audio[..., :min_length].detach()
            estimate = codec_audio[..., :min_length].detach()
        else:
            raise NotImplementedError()
        result = self.metric(estimate, reference, fs=self.sampling_rate)
        return result.item()


class PESQMetric(BaseMetric):
    def __init__(
        self, *args, sampling_rate=16000, mode="wb", version="audio_recon", **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.metric = perceptual_evaluation_speech_quality
        self.sampling_rate = sampling_rate
        self.mode = mode
        self.version = version

    def __call__(
        self,
        codec_audio: torch.Tensor,
        recon_audio: torch.Tensor,
        audio: torch.Tensor,
        **kwargs
    ):
        if self.version == "codec_recon":
            reference = codec_audio.detach()
            estimate = recon_audio.detach()
        elif self.version == "audio_recon":
            min_length = min(audio.shape[-1], recon_audio.shape[-1])
            reference = audio[..., :min_length].detach()
            estimate = recon_audio[..., :min_length].detach()
        elif self.version == "audio_codec":
            min_length = min(audio.shape[-1], codec_audio.shape[-1])
            reference = audio[..., :min_length].detach()
            estimate = codec_audio[..., :min_length].detach()
        else:
            raise NotImplementedError()
        result = self.metric(estimate, reference, fs=self.sampling_rate, mode=self.mode)
        return result.item()


class VISQOLMetric(BaseMetric):
    def __init__(
        self, *args, sampling_rate=16000, mode="speech", version="audio_recon", **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.metric = calc_visqol
        self.sampling_rate = sampling_rate
        self.mode = mode
        self.version = version
        self.target_rate = 16000 if mode == "speech" else 48000

    def __call__(
        self,
        codec_audio: torch.Tensor,
        recon_audio: torch.Tensor,
        audio: torch.Tensor,
        **kwargs
    ):
        if self.version == "codec_recon":
            reference = codec_audio.detach()
            estimate = recon_audio.detach()
        elif self.version == "audio_recon":
            min_length = min(audio.shape[-1], recon_audio.shape[-1])
            reference = audio[..., :min_length].detach()
            estimate = recon_audio[..., :min_length].detach()
        elif self.version == "audio_codec":
            min_length = min(audio.shape[-1], codec_audio.shape[-1])
            reference = audio[..., :min_length].detach()
            estimate = codec_audio[..., :min_length].detach()
        else:
            raise NotImplementedError()

        if self.sampling_rate != self.target_rate:
            resample = torchaudio.transforms.Resample(
                self.sampling_rate, self.target_rate
            )
            estimate = [resample(elem) for elem in estimate]
            estimate = torch.stack(estimate)
            reference = [resample(elem) for elem in reference]
            reference = torch.stack(reference)

        estimate = estimate.cpu().numpy().astype(float)
        reference = reference.cpu().numpy().astype(float)

        result = self.metric(estimate=estimate, reference=reference, mode=self.mode)
        return result.item()


class WERMetric(BaseMetric):
    def __init__(
        self,
        *args,
        model_id="openai/whisper-tiny",
        device="cpu",
        version="audio_recon",
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.device = device
        self.asr_pipeline = init_asr_model(model_id=model_id, device=device)
        self.version = version

    def __call__(
        self, text, recon_audio, codec_audio, recon_text=None, codec_text=None, **kwargs
    ):
        if self.version == "codec_recon":
            if codec_text is None:
                reference = run_asr_model(
                    self.asr_pipeline,
                    codec_audio.detach().cpu(),
                    normalize=True,
                )
            else:
                reference = codec_text
            if recon_text is None:
                estimate = run_asr_model(
                    self.asr_pipeline,
                    recon_audio.detach().cpu(),
                    normalize=True,
                )
            else:
                estimate = recon_text
        elif self.version == "audio_recon":
            reference = text

            if recon_text is None:
                estimate = run_asr_model(
                    self.asr_pipeline,
                    recon_audio.detach().cpu(),
                    normalize=True,
                )
            else:
                estimate = recon_text
        elif self.version == "audio_codec":
            reference = text
            if codec_text is None:
                estimate = run_asr_model(
                    self.asr_pipeline,
                    codec_audio.detach().cpu(),
                    normalize=True,
                )
            else:
                estimate = codec_text
        else:
            raise NotImplementedError()

        return wer(estimate, reference).item()


class SMAMetric(BaseMetric):
    def __init__(self, *args, version="audio_recon", threshold=0.7, **kwargs):
        super().__init__(*args, **kwargs)
        self.version = version
        self.threshold = threshold

    def __call__(self, recon_emb, codec_emb, audio_emb, **kwargs):
        if self.version == "codec_recon":
            reference = codec_emb
            estimate = recon_emb
        elif self.version == "audio_recon":
            reference = audio_emb
            estimate = recon_emb
        elif self.version == "audio_codec":
            reference = audio_emb
            estimate = codec_emb
        else:
            raise NotImplementedError()

        cos = torch.nn.functional.cosine_similarity(
            reference, estimate, dim=1, eps=1e-8
        )  # [-1, 1], shape [B]

        sim01 = (cos + 1.0) * 0.5

        # Fraction of matches
        match_ratio = (sim01 >= self.threshold).float().mean()

        return match_ratio.item()
