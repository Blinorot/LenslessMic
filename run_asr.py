import argparse
import re
from pathlib import Path

import nemo.collections.asr as nemo_asr


def normalize_text(text: str):
    text = text.lower().strip()
    text = re.sub(r"[^a-z ]", "", text)
    return text


def main(args):
    # Load model once
    model = nemo_asr.models.ASRModel.from_pretrained(args.model)

    audio_dir = Path(args.audio_dir).resolve()
    out_dir = audio_dir.parents[0] / "asr_text"
    out_dir.mkdir(parents=True, exist_ok=True)

    files = [str(p) for p in audio_dir.rglob("*.wav")]

    # Transcribe all in batches on GPU
    texts = model.transcribe(files, batch_size=args.batch_size, return_hypotheses=False)

    for f, text in zip(files, texts):
        out_txt = out_dir / (Path(f).stem + ".txt")
        out_txt.write_text(normalize_text(text), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run NeMo ASR.")
    parser.add_argument(
        "--model",
        default="nvidia/parakeet-tdt-0.6b-v2",
        help="NeMo model (default: 'nvidia/parakeet-tdt-0.6b-v2').",
    )
    parser.add_argument(
        "--audio-dir",
        required=True,
        help="Directory with .wav audio files (codec or recon).",
    )
    parser.add_argument(
        "--batch-size",
        default=16,
        help="Batch size.",
    )
    args = parser.parse_args()
    main(args)
