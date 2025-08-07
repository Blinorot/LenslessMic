import re

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


def init_asr_model(model_id="openai/whisper-tiny", device="cpu"):
    """
    Initialize speech recognition system.
    In this example, we use general class. The exact model can be provided
    via config.asr.model_id (id from HuggingFace).

    Args:
        model_id (str): model name on HuggingFace.
        device (str): model device.
    Returns:
        asr_pipeline: HF pipeline to convert speech into text.
    """
    torch_dtype = torch.float32 if device == "cpu" else torch.float16

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    asr_pipeline = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )
    return asr_pipeline


def run_asr_model(asr_pipeline, audio, normalize=True):
    """
    Get transcription for a speech input.

    Args:
        asr_pipeline: HF pipeline to get text from audio input.
        audio (torch.Tensor): input audio (B x 1 x T).
        normalize (bool): whether to normalize text.
    Returns:
        text (list[str]): text transcription of the given audio.
    """
    # pipeline expects numpy array of shape (T)
    # so we convert and take 0-th channel

    text = []
    for elem in audio:
        text_output = asr_pipeline(elem.numpy()[0])["text"]
        if normalize:
            text_output = normalize_text(text_output)
        text.append(text_output)

    return text


def normalize_text(text: str):
    text = text.lower().strip()
    text = re.sub(r"[^a-z ]", "", text)
    return text
