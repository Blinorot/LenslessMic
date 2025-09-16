import json
import os
import shutil
from pathlib import Path

import torchaudio
import wget
from tqdm import tqdm

from src.datasets.base_dataset import BaseDataset
from src.datasets.data_utils import normalize_text
from src.utils.io_utils import ROOT_PATH

URL_LINKS = {
    "dev-clean": "https://www.openslr.org/resources/12/dev-clean.tar.gz",
    "dev-other": "https://www.openslr.org/resources/12/dev-other.tar.gz",
    "test-clean": "https://www.openslr.org/resources/12/test-clean.tar.gz",
    "test-other": "https://www.openslr.org/resources/12/test-other.tar.gz",
    "train-clean-100": "https://www.openslr.org/resources/12/train-clean-100.tar.gz",
    "train-clean-360": "https://www.openslr.org/resources/12/train-clean-360.tar.gz",
    "train-other-500": "https://www.openslr.org/resources/12/train-other-500.tar.gz",
}


class LibrispeechDataset(BaseDataset):
    def __init__(self, part, data_dir=None, *args, **kwargs):
        if data_dir is None:
            data_dir = ROOT_PATH / "data" / "datasets" / "librispeech"
            data_dir.mkdir(exist_ok=True, parents=True)
        self._data_dir = data_dir
        if part == "train_all":
            index = sum(
                [
                    self._get_or_load_index(part)
                    for part in URL_LINKS
                    if "train" in part
                ],
                [],
            )
        elif isinstance(part, list):
            index = sum(
                [self._get_or_load_index(part_i) for part_i in part],
                [],
            )
        else:
            index = self._get_or_load_index(part)

        super().__init__(index, *args, **kwargs)

    def _load_part(self, part):
        arch_path = self._data_dir / f"{part}.tar.gz"
        print(f"Loading part {part}")
        wget.download(URL_LINKS[part], str(arch_path))
        shutil.unpack_archive(arch_path, self._data_dir)
        for fpath in (self._data_dir / "LibriSpeech").iterdir():
            shutil.move(str(fpath), str(self._data_dir / fpath.name))
        os.remove(str(arch_path))
        shutil.rmtree(str(self._data_dir / "LibriSpeech"))

    def _get_or_load_index(self, part):
        index_path = self._data_dir / f"{part}_index.json"
        if index_path.exists():
            with index_path.open() as f:
                index = json.load(f)
        else:
            index = self._create_index(part)
            with index_path.open("w") as f:
                json.dump(index, f, indent=2)
        return index

    def _create_index(self, part):
        index = []
        split_dir = self._data_dir / part
        if not split_dir.exists():
            self._load_part(part)

        # new dir for audio files
        new_audio_dir = split_dir / "audio"

        if new_audio_dir.exists():
            # librispeech was already processed
            for file in tqdm(
                os.listdir(str(new_audio_dir)),
                desc=f"Preparing librispeech folders: {part}",
            ):
                if file.endswith(".flac"):
                    new_flac_path = new_audio_dir / file
                    new_txt_path = new_flac_path.with_suffix(".txt")
                    f_text = new_txt_path.read_text()
                    t_info = torchaudio.info(str(new_flac_path))
                    length = t_info.num_frames / t_info.sample_rate

                    index.append(
                        {
                            "audio_path": str(new_flac_path),
                            "text": f_text,
                            "audio_len": length,
                        }
                    )

            return index

        new_audio_dir.mkdir(exist_ok=True, parents=True)

        flac_dirs = set()
        for dirpath, dirnames, filenames in os.walk(str(split_dir)):
            if any([f.endswith(".flac") for f in filenames]):
                flac_dirs.add(dirpath)
        for flac_dir in tqdm(
            list(flac_dirs), desc=f"Preparing librispeech folders: {part}"
        ):
            flac_dir = Path(flac_dir)
            trans_path = list(flac_dir.glob("*.trans.txt"))[0]
            with trans_path.open() as f:
                for line in f:
                    f_id = line.split()[0]
                    f_text = " ".join(line.split()[1:]).strip()
                    flac_path = (flac_dir / f"{f_id}.flac").absolute().resolve()
                    t_info = torchaudio.info(str(flac_path))
                    length = t_info.num_frames / t_info.sample_rate

                    new_flac_path = new_audio_dir / flac_path.name
                    shutil.move(str(flac_path), str(new_flac_path))

                    f_text = normalize_text(f_text)
                    new_txt_path = new_audio_dir / flac_path.with_suffix(".txt").name
                    with new_txt_path.open("w", encoding="utf-8") as text_file:
                        text_file.write(f_text)

                    index.append(
                        {
                            "audio_path": str(new_flac_path),
                            "text": f_text,
                            "audio_len": length,
                        }
                    )

        for p in split_dir.iterdir():
            if str(p) == str(new_audio_dir):
                continue
            if p.is_dir():
                shutil.rmtree(p)

        return index
