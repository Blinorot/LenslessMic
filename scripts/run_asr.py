import argparse
import re
from pathlib import Path

import nemo.collections.asr as nemo_asr


def normalize_text(text: str):
    text = text.lower().strip()
    text = re.sub(r"[^a-z ]", "", text)
    return text


def find_audio_dirs(root: Path, target_names=("audio", "codec_audio")):
    """
    Yield all directories whose name is in target_names
    under root (including root itself).
    """
    seen = set()

    if root.is_dir() and root.name in target_names:
        seen.add(root.resolve())
        yield root

    for p in root.rglob("*"):
        if p.is_dir() and p.name in target_names:
            rp = p.resolve()
            if rp not in seen:
                seen.add(rp)
                yield p


def transcribe(model, audio_dir, batch_size):
    out_dir = audio_dir.parent / "asr_text"
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted([str(p) for p in audio_dir.rglob("*.wav")])
    files = files + sorted([str(p) for p in audio_dir.rglob("*.flac")])

    # Transcribe all in batches on GPU
    hypotheses = model.transcribe(files, batch_size=batch_size)

    for f, hyp in zip(files, hypotheses):
        text = hyp.text
        out_txt = out_dir / (Path(f).stem + ".txt")
        out_txt.write_text(normalize_text(text), encoding="utf-8")


def main(args):
    # Load model once
    model = nemo_asr.models.ASRModel.from_pretrained(args.model)
    batch_size = args.batch_size

    audio_dir = Path(args.audio_dir).resolve()

    if not args.recursive:
        transcribe(model, audio_dir, batch_size)
    else:
        all_audio_dirs = find_audio_dirs(audio_dir)
        for audio_dir in all_audio_dirs:
            transcribe(model, audio_dir, batch_size)


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
        type=int,
        help="Batch size.",
    )
    parser.add_argument(
        "--recursive", action="store_true", help="Go over all subfolders named 'audio'."
    )
    args = parser.parse_args()
    main(args)
