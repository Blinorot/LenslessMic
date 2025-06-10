import torch


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
