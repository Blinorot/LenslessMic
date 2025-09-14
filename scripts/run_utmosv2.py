import argparse
import re
from pathlib import Path

import torch
import utmosv2


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


def get_mos(model, audio_dir, batch_size):
    out_dir = audio_dir.parent / "utmos_mos"
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted([str(p) for p in audio_dir.rglob("*.wav")])
    files = files + sorted([str(p) for p in audio_dir.rglob("*.flac")])

    # Transcribe all in batches on GPU
    mos = model.predict(input_dir=str(audio_dir), val_list=files, batch_size=batch_size)
    torch.save(mos, out_dir / "mos_output.pth")


def main(args):
    # Load model once
    model = utmosv2.create_model(pretrained=True)
    batch_size = args.batch_size

    audio_dir = Path(args.audio_dir).resolve()

    if not args.recursive:
        get_mos(model, audio_dir, batch_size)
    else:
        all_audio_dirs = find_audio_dirs(audio_dir)
        for audio_dir in all_audio_dirs:
            get_mos(model, audio_dir, batch_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run UTMOSv2.")
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
