import torch.nn.functional as F

from src.lensless.utils import ungroup_frames
from src.transforms import MinMaxNormalize


def reconstruct_codec(
    lensless_codec_video,
    recon_model,
    min_vals,
    max_vals,
    resize_coef=1,
    roi_kwargs=None,
    corners_list=None,
    group_frames_kwargs=None,
    normalize_lensless=False,
):
    """
    Reconstruct lensed codec video from a lensless one.

    Args:
        lensless_codec_video (Tensor): lensless codec video (BxDxHxWxCxT_latent).
        recon_model (nn.Module): reconstruction model with
            reconstruct_video method.
        min_vals (float | Tensor): min_vals for original lensed codec video.
            Used for normalization.
        max_vals (float | Tensor): max_vals for original lensed codec video.
            Used for normalization.
        resize_coef (int): the scaling factor for the original lensed
            codec video.
        roi_kwargs (dict | optional): top_left, height, and width for ROI.
        corners_list (None | list): list of coordinates for matching corners.
        group_frames_kwargs (dict | None): configuration for ungroup_frames function.
            See src.lensless.utils.ungroup_frames. Ignored if None.
        normalize_lensless (bool): whether to peak-normalize lensless video.
    Returns:
        recon_lensed_video (Tensor): reconstructed lensed codec video.
            If roi_kwargs are provided, the ROI part is returned.
    """
    if normalize_lensless:
        # set min = 0 to only change max.
        min_max_normalizer = MinMaxNormalize(min=0, dim=(0, 4, 5))
        lensless_codec_video = min_max_normalizer.normalize(lensless_codec_video)

    recon_codec_video = recon_model.reconstruct_video(
        lensless_codec_video, roi_kwargs, corners_list
    )

    if group_frames_kwargs is not None:
        recon_codec_video = ungroup_frames(recon_codec_video, **group_frames_kwargs)

    min_max_normalizer = MinMaxNormalize(min=None, max=None, dim=(0, 4))
    # use dim = 0, 4 to support multi-object and multi-channel input

    if resize_coef > 1:
        # apply avg_pool on H and W
        B, D, H, W, C, T = recon_codec_video.shape
        recon_codec_video = recon_codec_video.permute(0, 1, 5, 4, 2, 3)
        recon_codec_video = recon_codec_video.reshape(B * D * T, C, H, W)
        recon_codec_video = F.avg_pool2d(
            recon_codec_video, kernel_size=resize_coef, stride=resize_coef
        )
        _, _, H, W = recon_codec_video.shape
        recon_codec_video = recon_codec_video.reshape(B, D, T, C, H, W)
        recon_codec_video = recon_codec_video.permute(0, 1, 4, 5, 3, 2)

    if isinstance(min_vals, float):
        min_vals = [min_vals] * recon_codec_video.shape[-1]
    if isinstance(max_vals, float):
        max_vals = [max_vals] * recon_codec_video.shape[-1]

    for i in range(recon_codec_video.shape[-1]):
        # normalize each frame to [0, 1]
        recon_codec_video[..., i] = min_max_normalizer.normalize(
            recon_codec_video[..., i]
        )
        # denormalize to codec range
        recon_codec_video[..., i] = min_max_normalizer.denormalize(
            recon_codec_video[..., i], min_val=min_vals[i], max_val=max_vals[i]
        )

    return recon_codec_video
