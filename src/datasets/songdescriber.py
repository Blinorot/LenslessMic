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
    "audio": "https://zenodo.org/records/10072001/files/audio.zip?download=1",
    "license": "https://zenodo.org/records/10072001/files/audio_licenses.txt?download=1",
}


class SongDescriberDataset(BaseDataset):
    """
    The Song Describer Dataset: a Corpus of Audio
    Captions for Music-and-Language Evaluation

    https://zenodo.org/records/10072001
    """

    def __init__(
        self, part, target_sr=16000, crop_length=6, data_dir=None, *args, **kwargs
    ):
        if data_dir is None:
            data_dir = ROOT_PATH / "data" / "datasets" / "songdescriber"
            data_dir.mkdir(exist_ok=True, parents=True)
        self._data_dir = data_dir
        index = self._get_or_load_index(part, target_sr, crop_length)

        super().__init__(index, target_sr=target_sr, *args, **kwargs)

    def _load_part(self, part, target_sr, crop_length):
        arch_path = self._data_dir / "audio.zip"
        license_path = self._data_dir / "audio_licenses.txt"
        print(f"Loading dataset: {part}")
        wget.download(URL_LINKS["audio"], str(arch_path))
        wget.download(URL_LINKS["license"], str(license_path))
        shutil.unpack_archive(arch_path, self._data_dir / "raw_data")
        os.remove(arch_path)

        permitted_keywords = [
            "creativecommons.org/licenses/by/",  # CC-BY
            "creativecommons.org/licenses/by-sa/",  # CC-BY-SA
            "artlibre.org/licence/lal/",  # License Art Libre
        ]

        audio_dir = self._data_dir / part / "audio"
        audio_dir.mkdir(exist_ok=True, parents=True)

        license_list = license_path.read_text().split("\n")
        for line_ind in range(0, len(license_list), 4):
            filename = license_list[line_ind][:-4]
            full_filename = filename + ".2min.mp3"
            audio_path = self._data_dir / "raw_data" / "audio" / full_filename
            metadata = license_list[line_ind + 1]
            license = license_list[line_ind + 2]
            if any(pk in license for pk in permitted_keywords):
                audio, sr = torchaudio.load(audio_path)
                audio = torchaudio.functional.resample(audio, sr, target_sr)
                audio = audio[..., : target_sr * crop_length]
                new_filename = filename.replace("/", ".") + f".{crop_length}s.mp3"
                new_audio_path = audio_dir / new_filename
                torchaudio.save(new_audio_path, audio, target_sr)
                new_license_path = new_audio_path.with_suffix(".txt")
                full_license = metadata + "\n" + license
                with new_license_path.open("w") as f:
                    f.write(full_license)

        os.remove(license_path)
        shutil.rmtree(self._data_dir / "raw_data")

    def _get_or_load_index(self, part, target_sr, crop_length):
        index_path = self._data_dir / f"{part}_index.json"
        if index_path.exists():
            with index_path.open() as f:
                index = json.load(f)
        else:
            index = self._create_index(part, target_sr, crop_length)
            with index_path.open("w") as f:
                json.dump(index, f, indent=2)
        return index

    def _create_index(self, part, target_sr, crop_length):
        index = []
        split_dir = self._data_dir / part
        if not split_dir.exists():
            self._load_part(part, target_sr, crop_length)

        # new dir for audio files
        audio_dir = split_dir / "audio"

        # librispeech was already processed
        for file in tqdm(
            os.listdir(str(audio_dir)),
            desc=f"Preparing songdescriber folders: {part}",
        ):
            if file.endswith(".mp3"):
                audio_path = audio_dir / file
                t_info = torchaudio.info(str(audio_path))
                length = t_info.num_frames / t_info.sample_rate

                index.append(
                    {
                        "audio_path": str(audio_path),
                        "text": "",
                        "audio_len": length,
                    }
                )

        return index
