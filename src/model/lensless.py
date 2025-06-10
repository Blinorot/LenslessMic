import torch
from torch import nn

import lensless
from lensless.recon.model_dict import download_model, load_model
from src.lensless.utils import get_roi_indexes


class LenslessWrapper(nn.Module):
    def __init__(
        self,
        recon_name,
        recon_kwargs,
        use_loader=False,
        loader_kwargs=None,
        use_batch_video_version=False,
        freeze_weights=False,
    ):
        """
        Args:
            recon_name (str): reconstruction method name from lensless package.
            recon_kwargs (Any): kwargs for the reconstruction method.
            use_loader (bool): whether to use loader from lensless package.
                Use for trainable reconstruction methods from lensless package.
            loader_kwargs (dict | None): kwargs for loaded trainable model.
            use_batch_video_version (bool): whether to use batch or for-loop
                for video reconstruction. Set to True if VRAM is big.
            freeze_weights (bool): whether to freeze weights and avoid
                gradient calculation.
        """
        super().__init__()

        if use_loader:
            model_path = download_model(**loader_kwargs)
            self.recon = load_model(model_path, **recon_kwargs)
        else:
            self.recon = getattr(lensless, recon_name)(**recon_kwargs)

        self.use_batch_video_version = use_batch_video_version

        if freeze_weights:
            self.change_requires_grad(requires_grad=False)

    def change_requires_grad(self, requires_grad):
        for param in self.parameters():
            param.requires_grad = requires_grad

    def forward(self, lensless, roi_kwargs=None):
        """
        Reconstruct lensless image.

        Args:
            lensless (Tensor): lensless image (BxDxHxWxC).
            roi_kwargs (dict | optional): top_left, height, and width for ROI.
        Returns:
            recon_lensed (Tensor): reconstructed lensed image. If roi_kwargs
                are provided, the ROI part is returned.
        """
        # apply
        if isinstance(self.recon, nn.Module):
            # Trainable Recon can work with batch
            recon_lensed = self.recon(lensless)
        else:
            recon_list = []
            for i in range(lensless.shape[0]):
                self.recon.set_data(lensless[i : i + 1])
                # non-trainable methods work only with batch_size = 1
                # and remove it in the end
                recon_list.append(self.recon.apply(plot=False).unsqueeze(0))
            recon_lensed = torch.cat(recon_list, dim=0)

        if roi_kwargs is not None:
            roi_indexes = get_roi_indexes(
                n_dim=len(recon_lensed.shape), axis=(-3, -2), **roi_kwargs
            )

            recon_lensed = recon_lensed[roi_indexes]

        return recon_lensed

    def reconstruct_video(self, lensless_video, roi_kwargs=None):
        """
        Reconstruct lensless video.

        Args:
            lensless_video (Tensor): lensless video (BxDxHxWxCxT_latent).
            roi_kwargs (dict | optional): top_left, height, and width for ROI.
        Returns:
            recon_lensed_video (Tensor): reconstructed lensed video. If roi_kwargs
                are provided, the ROI part is returned.
        """
        if self.use_batch_video_version:
            # make the dims adjacent to proper reshape
            lensless_video = lensless_video.permute(0, 5, 1, 2, 3, 4)
            B, T, D, H, W, C = lensless_video.shape
            lensless_video = lensless_video.reshape(B * T, D, H, W, C)
            recon_lensed_video = self.forward(lensless_video, roi_kwargs)
            _, _, H, W, _ = recon_lensed_video.shape
            recon_lensed_video = recon_lensed_video.reshape(B, T, D, H, W, C)
            recon_lensed_video = recon_lensed_video.permute(0, 2, 3, 4, 5, 1)
            return recon_lensed_video

        # for-loop version

        recon_lensed_video = []
        for frame_index in range(lensless_video.shape[-1]):
            frame = lensless_video[..., frame_index]
            recon_lensed = self.forward(frame, roi_kwargs).unsqueeze(-1)
            recon_lensed_video.append(recon_lensed)
        recon_lensed_video = torch.cat(recon_lensed_video, dim=-1)
        return recon_lensed_video
