import argparse
import re
from pathlib import Path

import nemo.collections.asr as nemo_asr
import torch
import torchaudio
from nemo.collections.asr.parts.utils.speaker_utils import embedding_normalize
from torch.utils.data import DataLoader, Dataset


class FileDataset(Dataset):
    def __init__(self, files):
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        audio_path = self.files[index]
        audio, sr = torchaudio.load(audio_path)
        audio_len = audio.shape[-1]
        return {
            "audio": audio,
            "audio_len": audio_len,
            "audio_path": audio_path,
        }


def collate_fn(dataset_items):
    all_audio = [elem["audio"][0] for elem in dataset_items]
    all_audio = torch.nn.utils.rnn.pad_sequence(all_audio, batch_first=True)
    all_lengths = torch.tensor(
        [elem["audio_len"] for elem in dataset_items], dtype=torch.long
    )
    all_paths = [elem["audio_path"] for elem in dataset_items]
    return all_audio, all_lengths, all_paths


def find_audio_dirs(root: Path):
    """
    Yield all directories named exactly 'audio'
    under root (including root if it's named 'audio').
    """
    seen = set()

    if root.is_dir() and root.name == "audio":
        seen.add(root.resolve())
        yield root

    for p in root.rglob("audio"):
        if p.is_dir():
            rp = p.resolve()
            if rp not in seen:
                seen.add(rp)
                yield p


@torch.no_grad()
def get_embeddings(model, audio_dir, batch_size, device):
    out_dir = audio_dir.parent / "speaker_emb"
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted([str(p) for p in audio_dir.rglob("*.wav")])
    dataset = FileDataset(files)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn,
    )

    # Transcribe all in batches on GPU
    for audio, audio_len, audio_paths in dataloader:
        audio, audio_len = audio.to(device), audio_len.to(device)
        _, emb = model(input_signal=audio, input_signal_length=audio_len)
        emb = emb.squeeze().detach().cpu().numpy()
        emb = embedding_normalize(emb)
        for j in range(emb.shape[0]):
            filepath = audio_paths[j]
            out_emb_path = out_dir / (Path(filepath).stem + ".pth")
            out_emb = torch.tensor(emb[j])
            torch.save(out_emb, out_emb_path)


def main(args):
    # Load model once
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(args.model)
    model.eval()
    model.to(device)
    batch_size = args.batch_size

    audio_dir = Path(args.audio_dir).resolve()

    if not args.recursive:
        get_embeddings(model, audio_dir, batch_size, device)
    else:
        all_audio_dirs = find_audio_dirs(audio_dir)
        for audio_dir in all_audio_dirs:
            get_embeddings(model, audio_dir, batch_size, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run NeMo Speaker.")
    parser.add_argument(
        "--model",
        default="nvidia/speakerverification_en_titanet_large",
        help="NeMo model (default: 'nvidia/speakerverification_en_titanet_large').",
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
