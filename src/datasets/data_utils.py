import re
import shutil
import subprocess
from itertools import repeat

import numpy as np
from hydra.utils import instantiate

from src.datasets.collate import collate_fn
from src.utils.init_utils import set_worker_seed


def save_grayscale_video_ffv1(array: np.ndarray, path: str):
    """
    Save a grayscale uint8 video using FFV1 codec and MKV container.

    Args:
        array (np.ndarray): of shape (frames, height, width), dtype=uint8
        path (str): output video path (should end with .mkv)
    """
    assert array.ndim == 3, "Input must be (frames, height, width)"
    assert array.dtype == np.uint8, "Input must be uint8"

    ffmpeg = shutil.which("ffmpeg")
    frames, height, width = array.shape
    pixel_format = "gray"

    ffmpeg_cmd = [
        ffmpeg,
        "-loglevel",
        "error",  # only show actual errors
        "-y",  # Overwrite output
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-s",
        f"{width}x{height}",
        "-pix_fmt",
        pixel_format,
        "-r",
        "1",
        "-i",
        "-",  # Read from stdin
        "-an",  # No audio
        "-vcodec",
        "ffv1",
        "-pix_fmt",
        pixel_format,
        path,
    ]

    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

    try:
        for frame in array:
            proc.stdin.write(frame.tobytes())
    finally:
        proc.stdin.close()
        proc.wait()


def load_grayscale_video_ffv1(path: str) -> np.ndarray:
    """
    Load a grayscale uint8 FFV1 .mkv video.

    Args:
        path (str): path to the FFV1-encoded .mkv video file.
    Returns:
        video (np.ndarray): loaded video. (TxHxW)
    """
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")

    # --- Step 1: Get resolution ---
    cmd_res = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "csv=p=0:s=x",
        path,
    ]
    res_out = subprocess.run(cmd_res, capture_output=True, text=True).stdout.strip()
    width, height = map(int, res_out.split("x"))

    # --- Step 2: Get frame count ---
    cmd_count = [
        ffprobe,
        "-v",
        "error",
        "-count_frames",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=nb_read_frames",
        "-of",
        "default=nw=1:nk=1",
        path,
    ]
    count_out = subprocess.run(cmd_count, capture_output=True, text=True).stdout.strip()
    num_frames = int(count_out)

    # --- Step 3: Read raw grayscale bytes from FFmpeg ---
    cmd_ffmpeg = [
        ffmpeg,
        "-loglevel",
        "error",  # only show actual errors
        "-i",
        path,
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "-",
    ]

    proc = subprocess.Popen(
        cmd_ffmpeg, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
    )

    expected_bytes = num_frames * height * width
    raw = proc.stdout.read(expected_bytes)
    proc.stdout.close()
    proc.wait()

    if len(raw) != expected_bytes:
        raise ValueError(f"Expected {expected_bytes} bytes, got {len(raw)}")

    array = (
        np.frombuffer(raw, dtype=np.uint8).copy().reshape((num_frames, height, width))
    )
    return array


def inf_loop(dataloader):
    """
    Wrapper function for endless dataloader.
    Used for iteration-based training scheme.

    Args:
        dataloader (DataLoader): classic finite dataloader.
    """
    for loader in repeat(dataloader):
        yield from loader


def move_batch_transforms_to_device(batch_transforms, device):
    """
    Move batch_transforms to device.

    Notice that batch transforms are applied on the batch
    that may be on GPU. Therefore, it is required to put
    batch transforms on the device. We do it here.

    Batch transforms are required to be an instance of nn.Module.
    If several transforms are applied sequentially, use nn.Sequential
    in the config (not torchvision.Compose).

    Args:
        batch_transforms (dict[Callable] | None): transforms that
            should be applied on the whole batch. Depend on the
            tensor name.
        device (str): device to use for batch transforms.
    """
    for transform_type in batch_transforms.keys():
        transforms = batch_transforms.get(transform_type)
        if transforms is not None:
            for transform_name in transforms.keys():
                transforms[transform_name] = transforms[transform_name].to(device)


def get_dataloaders(config, device, codec):
    """
    Create dataloaders for each of the dataset partitions.
    Also creates instance and batch transforms.

    Args:
        config (DictConfig): hydra experiment config.
        device (str): device to use for batch transforms.
        codec (nn.Module): audio codec for the dataset.
    Returns:
        dataloaders (dict[DataLoader]): dict containing dataloader for a
            partition defined by key.
        batch_transforms (dict[Callable] | None): transforms that
            should be applied on the whole batch. Depend on the
            tensor name.
    """
    # transforms or augmentations init
    batch_transforms = instantiate(config.transforms.batch_transforms)
    move_batch_transforms_to_device(batch_transforms, device)

    # dataloaders init
    dataloaders = {}
    for dataset_partition in config.datasets.keys():
        # dataset partitions init
        dataset = instantiate(
            config.datasets[dataset_partition], codec=codec
        )  # instance transforms are defined inside

        dataloader_type = "train" if dataset_partition == "train" else "inference"

        dataloader_config = config.dataloader[dataloader_type]

        assert dataloader_config["batch_size"] <= len(dataset), (
            f"The batch size ({dataloader_config['batch_size']}) cannot "
            f"be larger than the dataset length ({len(dataset)})"
        )

        partition_dataloader = instantiate(
            dataloader_config,
            dataset=dataset,
            collate_fn=collate_fn,
            drop_last=(dataloader_type == "train"),
            shuffle=(dataloader_type == "train"),
            worker_init_fn=set_worker_seed,
        )
        dataloaders[dataset_partition] = partition_dataloader

    return dataloaders, batch_transforms


def normalize_text(text: str):
    text = text.lower()
    text = re.sub(r"[^a-z ]", "", text)
    return text
