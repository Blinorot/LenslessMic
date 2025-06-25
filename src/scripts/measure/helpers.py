"""
Copy of some function from other src.package_name.
Used to avoid import errors on the Raspberry Pi that cannot install torch.
"""
import shutil
import subprocess

import numpy as np


def rgb2gray_np(rgb):
    """
    Convert RGB image to grayscale.

    Args:
        rgb (np.ndarray): array of shape (B, D, H, W, C)
    Returns:
        gray (np.ndarray): grayscale array of shape (B, D, H, W, 1)
    """
    if rgb.shape[-1] == 1:  # already grayscale
        return rgb
    assert len(rgb.shape) == 5, "Input must be of shape (B, D, H, W, C)"

    weights = np.array([0.2989, 0.5870, 0.1140], dtype=rgb.dtype)
    gray = np.einsum("bdhwc,c->bdhw", rgb, weights)
    return gray[..., None]


def patchify_gray_video_np(video, patch_height, patch_width, **kwargs):
    """
    Split video frames into patches and consider each patch as a frame.

    Args:
        video (np.array): tensor of shape (TxHxW)
        patch_height (int): height of each patch.
        patch_width (int): width of each patch.
    Returns:
        patchified_video (np.array): patchified video of shape
            (T_patchifiedxpatch_heightxpatch_width)
    """
    T, H, W = video.shape
    assert H % patch_height == 0, "Frame height shall be divisible by patch height."
    assert W % patch_width == 0, "Frame width shall be divisible by patch width."

    patchified_video = video.reshape(
        T, patch_height, H // patch_height, patch_width, W // patch_width
    )
    patchified_video = patchified_video.transpose(0, 2, 4, 1, 3)
    patchified_video = patchified_video.reshape(-1, patch_height, patch_width)

    return patchified_video


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
