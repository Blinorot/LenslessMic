import torch
import torchvision.transforms.v2 as TV2

from lensless.recon.rfft_convolve import RealFFTConvolve2D
from src.lensless.utils import get_roi_indexes, group_frames
from src.transforms import MinMaxNormalize


def simulate_lensless(lensed, psf, roi_kwargs, normalize=True, normalize_dims=(0, 4)):
    """
    Simulate lensless image, given lensed image, PSF, and ROI information.

    Args:
        lensed (Tensor): lensed version of image (BxDxHxWxC).
        psf (Tensor): PSF for the system.
        roi_kwargs (dict): top_left, height, and width for ROI.
        normalize (bool): whether to rescale lensless output via peak-normalization.
        normalize_dims (int | tuple): dims for normalization. Use 0 for Batch-only.
            Use (0, 4) for batch and channel-wise normalization.
    Returns:
        lensless (Tensor): simulated lensless image.
        resized_lensed (Tensor): resized (according to ROI) lensed image.
    """
    roi_indexes = get_roi_indexes(n_dim=len(lensed.shape), axis=(-3, -2), **roi_kwargs)
    output_shape = lensed.shape[0:1] + psf.shape
    resized_lensed = torch.zeros(output_shape)
    resized_lensed[roi_indexes] = lensed

    # create the convolution object
    convolver = RealFFTConvolve2D(psf=psf)

    lensless = convolver.convolve(resized_lensed)

    if normalize:
        # set min = 0 to only change max.
        min_max_normalizer = MinMaxNormalize(min=0, dim=normalize_dims)
        lensless = min_max_normalizer.normalize(lensless)

    return lensless, resized_lensed


def simulate_lensless_codec(
    codec_video,
    psf,
    roi_kwargs,
    resize_coef=1,
    min_vals=None,
    max_vals=None,
    return_min_max_values=False,
    normalize=True,
    normalize_dims=(0, 4),
    group_frames_kwargs=None,
):
    """
    Simulate lensless codec video, given lensed codec video,
    PSF, and ROI information. Resize if needed.

    Args:
        codec_video (Tensor): lensed version of codec video (BxDxHxWxCxT).
        psf (Tensor): PSF for the system.
        roi_kwargs (dict): top_left, height, and width for ROI.
        resize_coef (int): the scaling factor for resize.
        min_vals (float | Tensor): min_vals for original lensed codec video.
            Used for normalization.
        max_vals (float | Tensor): max_vals for original lensed codec video.
            Used for normalization.
        return_min_max_values (bool): whether to return min/max vals.
        normalize (bool): whether to rescale lensless output via peak-normalization.
        normalize_dims (int | tuple): dims for normalization. Use 0 for Batch-only.
            Use (0, 4) for batch and channel-wise normalization.
        group_frames_kwargs (dict | None): configuration for group_frames function.
            See src.lensless.utils.group_frames. Ignored if None.
    Returns:
        lensless_codec_video (Tensor): simulated lensless video.
        resized_codec_video (Tensor): resized (according to ROI) lensed video.
    """

    B, D, H, W, C, T = codec_video.shape

    assert C == psf.shape[-1], "PSF and video must have the same number of channels."

    new_H = resize_coef * H
    new_W = resize_coef * W

    if resize_coef > 1:
        transform = TV2.Resize(
            (new_H, new_W),
            interpolation=TV2.InterpolationMode.NEAREST,
        )
    else:
        transform = torch.nn.Identity()

    min_max_normalizer = MinMaxNormalize(min=min_vals, max=max_vals, dim=(0, 4))
    # use dim = 0, 4 to support multi-object and multi-channel input

    resized_codec_video = []
    lensless_codec_video = []

    min_vals_list = []
    max_vals_list = []

    transformed_lensed = torch.zeros((B, D, new_H, new_W, C, T))

    # normalize and resize
    for frame_index in range(T):
        frame = codec_video[..., frame_index]

        if return_min_max_values:
            frame, min_vals, max_vals = min_max_normalizer.normalize(
                frame, return_min_max_values
            )
            min_vals_list.append(min_vals)
            max_vals_list.append(max_vals)
        else:
            frame = min_max_normalizer.normalize(frame)

        # apply transform on H and W
        frame = frame.permute(0, 1, 4, 2, 3)
        frame = transform(frame)
        frame = frame.permute(0, 1, 3, 4, 2)

        transformed_lensed[..., frame_index] = frame

    # group if needed
    if group_frames_kwargs is not None:
        transformed_lensed = group_frames(transformed_lensed, **group_frames_kwargs)

    T = transformed_lensed.shape[-1]

    # get lensless
    for frame_index in range(T):
        frame = transformed_lensed[..., frame_index]

        lensless, resized_lensed = simulate_lensless(
            frame, psf, roi_kwargs, normalize, normalize_dims
        )

        lensless_codec_video.append(lensless.unsqueeze(-1))
        resized_codec_video.append(resized_lensed.unsqueeze(-1))

    lensless_codec_video = torch.cat(lensless_codec_video, dim=-1)
    resized_codec_video = torch.cat(resized_codec_video, dim=-1)

    if return_min_max_values:
        return lensless_codec_video, resized_codec_video, min_vals_list, max_vals_list

    return lensless_codec_video, resized_codec_video
