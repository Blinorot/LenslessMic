import json
import os
import shutil
from pathlib import Path

import numpy as np
import torch
import torchaudio
import wget
from tqdm import tqdm

from src.datasets.base_dataset import BaseDataset
from src.datasets.data_utils import save_grayscale_video_ffv1
from src.transforms import MinMaxNormalize
from src.utils.io_utils import ROOT_PATH


class RandomDataset(BaseDataset):
    def __init__(
        self,
        part,
        codec_name,
        length=200,
        data_length=150,
        dummy_audio_length=48000,
        data_dir=None,
        *args,
        **kwargs,
    ):
        if data_dir is None:
            data_dir = ROOT_PATH / "data" / "datasets" / "random"
            data_dir.mkdir(exist_ok=True, parents=True)
        self._data_dir = data_dir
        self.dummy_audio_length = dummy_audio_length

        assert codec_name is not None, "Provide codec_name"
        codec_size = codec_name[:5]
        index = self._get_or_load_index(part, codec_size, length, data_length)

        super().__init__(index, codec_name=codec_name, *args, **kwargs)

    def _get_or_load_index(self, part, codec_size, length, data_length):
        index_path = self._data_dir / f"{part}_{codec_size}_index.json"
        if index_path.exists():
            with index_path.open() as f:
                index = json.load(f)
        else:
            index = self._create_index(part, codec_size, length, data_length)
            with index_path.open("w") as f:
                json.dump(index, f, indent=2)
        return index

    def _create_index(self, part, codec_size, length, data_length):
        torch.manual_seed(1)

        normalizer = MinMaxNormalize(dim=0)

        index = []
        split_dir = self._data_dir / part / codec_size
        lensed_dir = split_dir / "lensed"

        if lensed_dir.exists():
            for file in tqdm(
                os.listdir(str(lensed_dir)),
                desc=f"Preparing random folders: {part}",
            ):
                if file.endswith(".mkv"):
                    video_path = lensed_dir / file

                    index.append(
                        {
                            "audio_path": "",
                            "text": "",
                            "lensed_codec_video_path": str(video_path),
                            "audio_len": -1,
                        }
                    )

            return index

        lensed_dir.mkdir(exist_ok=True, parents=True)

        image_size = int(codec_size[:2])

        for i in tqdm(
            range(length),
            desc=f"Preparing random folders: {part}",
        ):
            filename = f"{i:05d}"
            # normal distribution seems to be more aligned
            codec_video = torch.randn((data_length, image_size, image_size))
            codec_video = normalizer.normalize(codec_video)
            codec_video = (codec_video * 255).to(torch.uint8)

            video_path = lensed_dir / f"{filename}.mkv"

            save_grayscale_video_ffv1(codec_video.clone().numpy(), str(video_path))

            index.append(
                {
                    "audio_path": "",
                    "text": "",
                    "lensed_codec_video_path": str(video_path),
                    "audio_len": -1,
                }
            )

        return index


if __name__ == "__main__":
    # create dataset
    dataset = RandomDataset("train", "16x16_130_16khz")
    dataset = RandomDataset("train", "32x32_120_16khz_original")
