import torch
import torch.nn.functional as F

from src.lensless.utils import ungroup_frames, unpatchify_video
from src.transforms import MinMaxNormalize


def reconstruct_codec(
    lensless_codec_video,
    recon_model,
    min_vals,
    max_vals,
    lensless_psf=None,
    resize_coef=1,
    roi_kwargs=None,
    corners_list=None,
    group_frames_kwargs=None,
    patchify_video_kwargs=None,
    normalize_lensless=False,
    return_raw=False,
    **kwargs,
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
        lensless_psf (None | Tensor): optional PSF instead of default one.
        resize_coef (int): the scaling factor for the original lensed
            codec video.
        roi_kwargs (dict | optional): top_left, height, and width for ROI.
        corners_list (None | list): list of coordinates for matching corners.
        group_frames_kwargs (dict | None): configuration for ungroup_frames function.
            See src.lensless.utils.ungroup_frames. Ignored if None.
        patchify_video_kwargs (dict | None): configuration for unpatchify_video function.
            See src.lensless.utils.unpatchify_video. Ignored if None.
        normalize_lensless (bool): whether to peak-normalize lensless video.
        return_raw (bool): also return non-averaged reconstruction.
    Returns:
        recon_lensed_video (Tensor): reconstructed lensed codec video.
            If roi_kwargs are provided, the ROI part is returned.
    """
    if normalize_lensless:
        # set min = 0 to only change max.
        min_max_normalizer = MinMaxNormalize(min=0, dim=(0, 4, 5))
        lensless_codec_video = min_max_normalizer.normalize(lensless_codec_video)

    raw_recon_codec_video = recon_model.reconstruct_video(
        lensless_codec_video,
        lensless_psf=lensless_psf,
        roi_kwargs=roi_kwargs,
        corners_list=corners_list,
    )

    if group_frames_kwargs is not None:
        n_orig_frames = group_frames_kwargs.get("n_orig_frames", None)
        pad_mask = None
        if n_orig_frames is None:
            pad_mask = kwargs["pad_mask"]
        elif n_orig_frames == -1:
            n_orig_frames = kwargs["n_orig_frames"]
        raw_recon_codec_video = ungroup_frames(
            raw_recon_codec_video,
            n_orig_frames=n_orig_frames,
            pad_mask=pad_mask,
            **group_frames_kwargs,
        )

    min_max_normalizer = MinMaxNormalize(min=None, max=None, dim=(0, 4))
    # use dim = 0, 4 to support multi-object and multi-channel input

    recon_codec_video = raw_recon_codec_video.clone()

    if resize_coef > 1:
        # apply avg_pool on H and W
        B, D, H, W, C, T = recon_codec_video.shape
        recon_codec_video = recon_codec_video.permute(0, 1, 5, 4, 2, 3).contiguous()
        recon_codec_video = recon_codec_video.reshape(B * D * T, C, H, W)
        recon_codec_video = F.avg_pool2d(
            recon_codec_video, kernel_size=resize_coef, stride=resize_coef
        )
        _, _, H, W = recon_codec_video.shape
        recon_codec_video = recon_codec_video.reshape(B, D, T, C, H, W)
        recon_codec_video = recon_codec_video.permute(0, 1, 4, 5, 3, 2).contiguous()

    if patchify_video_kwargs is not None:
        recon_codec_video = unpatchify_video(recon_codec_video, **patchify_video_kwargs)

    if isinstance(min_vals, float):
        min_vals = torch.tensor(min_vals, device=recon_codec_video.device).reshape(
            1, 1, 1, 1, 1
        )
        min_vals = min_vals.repeat(1, 1, 1, 1, 1, recon_codec_video.shape[-1])
    if isinstance(max_vals, float):
        max_vals = torch.tensor(max_vals, device=recon_codec_video.device).reshape(
            1, 1, 1, 1, 1
        )
        max_vals = max_vals.repeat(1, 1, 1, 1, 1, recon_codec_video.shape[-1])

    normalized_frames = []

    for i in range(recon_codec_video.shape[-1]):
        # normalize each frame to [0, 1]
        normalized_frame = min_max_normalizer.normalize(recon_codec_video[..., i])
        # denormalize to codec range
        normalized_frame = min_max_normalizer.denormalize(
            normalized_frame,
            min_val=min_vals[..., i],
            max_val=max_vals[..., i],
        )

        normalized_frames.append(normalized_frame.unsqueeze(-1))

    recon_codec_video = torch.cat(normalized_frames, dim=-1)

    if return_raw:
        return recon_codec_video, raw_recon_codec_video

    return recon_codec_video
