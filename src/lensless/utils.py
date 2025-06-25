import torch
import torchvision.transforms.v2 as T

from src.transforms import MinMaxNormalize


def get_roi_indexes(top_left, height, width, n_dim=4, axis=(-3, -2)):
    """
    Get indexes corresponding to region of interest (ROI).

    Args:
        top_left (tuple[int, int]): top left corner indexes.
        height (int): height of an object.
        width (int): width of an object.
        n_dim (int): number of dimensions.
        axis (tuple[int, int]): tuple with axis index for H and W.
    Returns:
        indexes (tuple[slice, slice, ...]): indexes of ROI.
            Slices for each dimension.
    """
    # extract according to axis
    index = [slice(None)] * n_dim
    index[axis[0]] = slice(top_left[0], top_left[0] + height)
    index[axis[1]] = slice(top_left[1], top_left[1] + width)

    return tuple(index)


def fix_video_perspective(video, corners_list, roi_kwargs):
    """
    Fix perspective using ROI region and corner points.

    Args:
        video (Tensor): video (BxDxHxWxCxT).
        corners_list (None | list): list of coordinates for matching corners.
        roi_kwargs (None | dict): kwargs for a rectangular box. Shows ROI.
    Returns:
        fixed_video (Tensor): fixed video (BxDxHxWxCxT).
    """
    fixed_video = torch.zeros_like(video)
    for frame_ind in range(video.shape[-1]):
        frame = video[..., frame_ind]
        frame = fix_perspective(frame, corners_list, roi_kwargs)
        fixed_video[..., frame_ind] = frame
    return fixed_video


def fix_perspective(img, corners_list, roi_kwargs):
    """
    Fix perspective using ROI region and corner points.

    Args:
        img (Tensor): image (BxDxHxWxC).
        corners_list (None | list): list of coordinates for matching corners.
        roi_kwargs (None | dict): kwargs for a rectangular box. Shows ROI.
    Returns:
        fixed_img (Tensor): fixed image (BxDxHxWxC).
    """
    top_left_corner = roi_kwargs["top_left"]
    top_right_corner = (top_left_corner[0], top_left_corner[1] + roi_kwargs["width"])
    bot_left_corner = (top_left_corner[0] + roi_kwargs["height"], top_left_corner[1])
    bot_right_corner = (
        top_left_corner[0] + roi_kwargs["height"],
        top_left_corner[1] + roi_kwargs["width"],
    )

    corners_output = [
        top_left_corner,
        top_right_corner,
        bot_left_corner,
        bot_right_corner,
    ]

    B, D, H, W, C = img.shape
    img = img.permute(0, 1, 4, 2, 3)
    img = img.reshape(B * D, C, H, W)
    img = T.functional.perspective(
        img, startpoints=corners_list, endpoints=corners_output
    )
    img = img.reshape(B, D, C, H, W)
    img = img.permute(0, 1, 3, 4, 2)
    return img


def normalize_video(
    video,
    min_vals=None,
    max_vals=None,
    return_min_max_values=False,
    normalize_dims=(0, 4),
):
    """
    Normalize video frames.

    Args:
        video (Tensor): video (BxDxHxWxCxT).
        min_vals (None | float | Tensor): min_vals for original video.
        max_vals (None | float | Tensor): max_vals for original video.
        return_min_max_values (bool): whether to return min/max vals.
        normalize_dims (int | tuple): dims for normalization. Use 0 for Batch-only.
            Use (0, 4) for batch and channel-wise normalization.
    Returns:
        normalized_video (Tensor): normalized video
        min_vals_list (list[Tensor|float]): list of min_vals for each frame (Optional).
        max_vals_list (list[Tensor|float]): list of max_vals for each frame (Optional).
    """
    min_max_normalizer = MinMaxNormalize(min=min_vals, max=max_vals, dim=normalize_dims)
    min_vals_list = []
    max_vals_list = []
    normalized_video = torch.zeros_like(video)
    for frame_index in range(video.shape[-1]):
        frame = video[..., frame_index]

        if return_min_max_values:
            frame, min_vals, max_vals = min_max_normalizer.normalize(
                frame, return_min_max_values
            )
            min_vals_list.append(min_vals)
            max_vals_list.append(max_vals)
        else:
            frame = min_max_normalizer.normalize(frame)
        normalized_video[..., frame_index] = frame.clone()

    if return_min_max_values:
        return normalized_video, min_vals_list, max_vals_list

    return normalized_video


def group_frames(video, n_rows, n_cols, row_space, col_space, **kwargs):
    """
    Group several frames into one large frame.

    Args:
        video (Tensor): tensor of shape (BxDxHxWxCxT)
        n_rows (int): number of frames per row of the large frame.
        n_cols (int): number of frames per column of the large frame.
        row_space (int): space between rows.
        col_space (int): space between columns.
    Returns:
        group_video (Tensor): grouped video of shape
            (B x D x group_H x group_W x C x group_T).
    """
    n_frames = n_rows * n_cols
    n_diff = video.shape[-1] % n_frames
    if n_diff != 0:
        new_video = torch.zeros_like(video[..., :1])
        new_video = new_video.repeat(1, 1, 1, 1, 1, n_frames - n_diff)
        video = torch.cat([video, new_video], dim=-1)
    n_group_frames = video.shape[-1] // n_frames

    B, D, H, W, C, _ = video.shape
    group_H = n_rows * H + (n_rows - 1) * row_space
    group_W = n_cols * W + (n_cols - 1) * col_space

    group_video = torch.zeros((B, D, group_H, group_W, C, n_group_frames))

    group_id = 0
    for frame_start in range(0, video.shape[-1], n_frames):
        for row_id in range(n_rows):
            for col_id in range(n_cols):
                frame_id = row_id * n_cols + col_id
                frame = video[..., frame_start + frame_id]

                H_pos = row_id * H + row_id * row_space
                W_pos = col_id * W + col_id * col_space
                group_video[
                    :, :, H_pos : H_pos + H, W_pos : W_pos + W, :, group_id
                ] = frame.clone()

        group_id += 1

    return group_video


def ungroup_frames(group_video, n_rows, n_cols, row_space, col_space, n_orig_frames):
    """
    Ungroup large frame into several frames. The opposite of group frames.

    Args:
        group_video (Tensor): tensor of shape
            (B x D x group_H x group_W x C x group_T).
        n_rows (int): number of frames per row of the large frame.
        n_cols (int): number of frames per column of the large frame.
        row_space (int): space between rows.
        col_space (int): space between columns.
    Returns:
        video (Tensor): ungrouped video of shape (BxDxHxWxCxT).
    """

    B, D, group_H, group_W, C, T = group_video.shape
    H = (group_H - (n_rows - 1) * row_space) // n_rows
    W = (group_W - (n_cols - 1) * col_space) // n_cols

    n_frames = n_rows * n_cols

    video = torch.zeros((B, D, H, W, C, n_frames * T))

    group_id = 0
    for frame_start in range(0, video.shape[-1], n_frames):
        for row_id in range(n_rows):
            for col_id in range(n_cols):
                frame_id = row_id * n_cols + col_id

                H_pos = row_id * H + row_id * row_space
                W_pos = col_id * W + col_id * col_space
                frame = group_video[
                    :, :, H_pos : H_pos + H, W_pos : W_pos + W, :, group_id
                ]
                video[..., frame_start + frame_id] = frame.clone()

        group_id += 1

    video = video[..., :n_orig_frames].clone()

    return video


def patchify_video(video, patch_height, patch_width, **kwargs):
    """
    Split video frames into patches and consider each patch as a frame.

    Args:
        video (Tensor): tensor of shape (BxDxHxWxCxT)
        patch_height (int): height of each patch.
        patch_width (int): width of each patch.
    Returns:
        patchified_video (Tensor): patchified video of shape
            (BxDxpatch_heightxpatch_widthxCxT_patchified)
    """
    B, D, H, W, C, T = video.shape
    assert H % patch_height == 0, "Frame height shall be divisible by patch height."
    assert W % patch_width == 0, "Frame width shall be divisible by patch width."

    patchified_video = video.reshape(
        B, D, patch_height, H // patch_height, patch_width, W // patch_width, C, T
    )
    patchified_video = patchified_video.permute(0, 1, 2, 4, 6, 3, 5, 7)
    patchified_video = patchified_video.reshape(B, D, patch_height, patch_width, C, -1)

    return patchified_video


def unpatchify_video(
    patchified_video, patch_height, patch_width, orig_height, orig_width
):
    """
    The inverse of patchify video function. Extracts patches from the
    time dimension and combines them into single frames.

    Args:
        patchified_video (Tensor): patchified video of shape
            (BxDxpatch_heightxpatch_widthxCxT_patchified)
        patch_height (int): height of each patch.
        patch_width (int): width of each patch.
        orig_height (int): height of original frame.
        orig_width (int): width of original frame.
    Returns:
        video (Tensor): tensor of shape (BxDxHxWxCxT)
    """
    B, D, _, _, C, T_patchified = patchified_video.shape

    height_patches = orig_height // patch_height
    width_patches = orig_width // patch_width
    T = T_patchified // (height_patches * width_patches)

    video = patchified_video.reshape(
        B, D, patch_height, patch_width, C, height_patches, width_patches, T
    )
    video = video.permute(0, 1, 2, 5, 3, 6, 4, 7)
    video = video.reshape(B, D, orig_height, orig_width, C, T)
    return video
