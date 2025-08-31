"""
Copy of some function from other src.package_name.
Used to avoid import errors on the Raspberry Pi that cannot install torch.
"""
import shutil
import subprocess

import cv2
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
    patchified_video = patchified_video.transpose(2, 4, 0, 1, 3)
    patchified_video = patchified_video.reshape(-1, patch_height, patch_width)

    return patchified_video


def group_gray_frames_np(video, n_rows, n_cols, row_space, col_space, **kwargs):
    """
    Group several frames into one large frame.

    Args:
        video (np.array): tensor of shape (TxHxW)
        n_rows (int): number of frames per row of the large frame.
        n_cols (int): number of frames per column of the large frame.
        row_space (int): space between rows.
        col_space (int): space between columns.
    Returns:
        group_video (np.array): grouped video of shape
            (group_T x group_H x group_W).
    """
    n_frames = n_rows * n_cols
    n_diff = video.shape[0] % n_frames
    if n_diff != 0:
        new_video = np.zeros_like(video[:1])
        new_video = np.tile(new_video, (n_frames - n_diff, 1, 1))
        video = np.concatenate([video, new_video], axis=0)
    n_group_frames = video.shape[0] // n_frames

    _, H, W = video.shape
    group_H = n_rows * H + (n_rows - 1) * row_space
    group_W = n_cols * W + (n_cols - 1) * col_space

    group_video = np.zeros((n_group_frames, group_H, group_W), dtype=video.dtype)

    group_id = 0
    for frame_start in range(0, video.shape[0], n_frames):
        for row_id in range(n_rows):
            for col_id in range(n_cols):
                frame_id = row_id * n_cols + col_id
                frame = video[frame_start + frame_id]

                H_pos = row_id * H + row_id * row_space
                W_pos = col_id * W + col_id * col_space
                group_video[
                    group_id, H_pos : H_pos + H, W_pos : W_pos + W
                ] = frame.copy()

        group_id += 1

    return group_video


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


def format_img(
    img_og,
    pad,
    vshift,
    brightness,
    screen_res,
    hshift,
    rot90,
    landscape,
    image_res,
):
    interpolation = cv2.INTER_NEAREST

    # load image
    # if img_og.ndim == 3 and img_og.shape[2] == 3:
    #     img_og = cv2.cvtColor(img_og, cv2.COLOR_BGR2RGB)
    if landscape:
        if img_og.shape[0] > img_og.shape[1]:
            img_og = np.rot90(img_og)

    # rotate image
    if rot90:
        img_og = np.rot90(img_og, k=rot90)

        # if odd, swap hshift and vshift
        if rot90 % 2:
            vshift, hshift = hshift, vshift

    if screen_res is not None:
        image_height, image_width = img_og.shape[:2]
        img = np.zeros((screen_res[1], screen_res[0], 3), dtype=img_og.dtype)
        if img_og.ndim == 2:
            img = img[..., 0]

        if image_res is None:
            # set image with padding and correct aspect ratio
            if screen_res[0] < screen_res[1]:
                max_ratio = screen_res[1] / float(image_height)

                new_width = int(screen_res[0] / (1 + 2 * pad / 100))
                ratio = new_width / float(image_width)
                # new_height = int(ratio * image_height)

            else:
                max_ratio = screen_res[0] / float(image_width)

                new_height = int(screen_res[1] / (1 + 2 * pad / 100))
                ratio = new_height / float(image_height)
                # new_width = int(ratio * image_width)

            ratio = min(ratio, max_ratio)
            new_width = int(ratio * image_width)
            new_height = int(ratio * image_height)
            image_res = (new_width, new_height)

        # if negative value in image res
        elif image_res[0] < 0 or image_res[1] < 0:
            assert (
                image_res[0] > 0 or image_res[1] > 0
            ), "Both dimensions cannot be negative."
            # rescale according to non-negative value
            if image_res[0] < 0:
                new_height = image_res[1]
                ratio = new_height / float(image_height)
                image_res = (int(ratio * image_width), new_height)

            elif image_res[1] < 0:
                new_width = image_res[0]
                ratio = new_width / float(image_width)
                image_res = (new_width, int(ratio * image_height))

        # set image within screen
        img_og = cv2.resize(img_og, image_res, interpolation=interpolation)
        img[: image_res[1], : image_res[0]] = img_og

        # center
        img = np.roll(img, shift=int((screen_res[1] - image_res[1]) / 2), axis=0)
        img = np.roll(img, shift=int((screen_res[0] - image_res[0]) / 2), axis=1)

    else:
        # pad image
        if pad:
            padding_amount = np.array(img_og.shape[:2]) * pad / 100
            pad_width = (
                (int(padding_amount[0] // 2), int(padding_amount[0] // 2)),
                (int(padding_amount[1] // 2), int(padding_amount[1] // 2)),
                (0, 0),
            )
            img = np.pad(img_og, pad_width=pad_width[: img.ndim])
        else:
            img = img_og

    if vshift:
        nx = img.shape[0]
        img = np.roll(img, shift=int(vshift * nx / 100), axis=0)

    if hshift:
        ny = img.shape[1]
        img = np.roll(img, shift=int(hshift * ny / 100), axis=1)

    if brightness:
        img = (img.astype(np.float32) * brightness / 100).astype(np.uint8)

    if img.ndim == 2:
        # pygame requires 3 channels
        # convert to black and white
        img = np.stack([img] * 3, axis=-1)

    return img
