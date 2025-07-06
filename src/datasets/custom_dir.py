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


class CustomDirDataset(BaseDataset):
    """
    General class for any custom dir of the following format:

    DatasetName
    ├── part_name_1
    │   └── audio
    │       ├── audio_1.{wav, flac, mp3}
    │       ├── audio_1.txt                 # text transcription
    │       ├── audio_2.{wav, flac, mp3}
    │       ├── audio_1.txt                 # text transcription
    │       ├── ...
    │       ├── audio_m.{wav, flac, mp3}
    │       └── audio_m.txt                 # text transcription
    ├── part_name_2
    ├── ...
    └── part_name_n

    The data should be located in ROOT_PATH/data/datasets/DatasetName.
    """

    def __init__(self, part, dataset_name="example", *args, **kwargs):
        data_dir = ROOT_PATH / "data" / "datasets" / dataset_name
        assert data_dir.exists(), f"{data_dir} not found, create dataset first."
        self._data_dir = data_dir
        self.dataset_name = dataset_name
        index = self._get_or_load_index(part)

        super().__init__(index, *args, **kwargs)

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
        split_dir = self._data_dir / part / "audio"
        assert split_dir.exists(), f"{split_dir} not found, create partition first."

        for file in tqdm(
            os.listdir(str(split_dir)),
            desc=f"Preparing {self.dataset_name} files: {part}",
        ):
            file_path = split_dir / file
            txt_path = file_path.with_suffix(".txt")
            if file_path.suffix in [".wav", ".flac", ".mp3"]:
                t_info = torchaudio.info(file_path)
                length = t_info.num_frames / t_info.sample_rate
                text = normalize_text(txt_path.read_text(encoding="utf-8"))

                index.append(
                    {
                        "audio_path": str(file_path),
                        "text": text,
                        "audio_len": length,
                    }
                )

        return index
