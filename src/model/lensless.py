import torch
from torch import nn

import lensless
from lensless.recon.model_dict import download_model, load_model
from lensless.utils.image import rgb2gray
from lensless.utils.io import load_psf
from src.lensless.utils import fix_perspective, get_roi_indexes


class LenslessWrapper(nn.Module):
    def __init__(
        self,
        recon_name,
        recon_kwargs,
        use_loader=False,
        loader_kwargs=None,
        use_batch_video_version=False,
        freeze_weights=False,
        psf_path=None,
        psf_loader_kwargs=None,
        grayscale_psf=False,
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
            psf_path (str | None): load psf for initialization if psf
                is not provided in recon_kwargs.
            psf_loader_kwargs (dict | None): kwargs for loading psf.
            grayscale_psf (bool): whether to convert psf to grayscale.
        """
        super().__init__()

        if psf_path is not None:
            psf = load_psf(psf_path, **psf_loader_kwargs)
            if grayscale_psf:
                psf = rgb2gray(psf)
            psf = torch.from_numpy(psf)
            recon_kwargs["psf"] = psf

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

    def forward(self, lensless, roi_kwargs=None, corners_list=None):
        """
        Reconstruct lensless image.

        Args:
            lensless (Tensor): lensless image (BxDxHxWxC).
            roi_kwargs (dict | optional): top_left, height, and width for ROI.
            corners_list (None | list): list of coordinates for matching corners.
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

        if corners_list is not None:
            assert roi_kwargs is not None, "Define ROI kwargs to fix perspective"
            recon_lensed = fix_perspective(recon_lensed, corners_list, roi_kwargs)

        if roi_kwargs is not None:
            roi_indexes = get_roi_indexes(
                n_dim=len(recon_lensed.shape), axis=(-3, -2), **roi_kwargs
            )

            recon_lensed = recon_lensed[roi_indexes]

        return recon_lensed

    def reconstruct_video(self, lensless_video, roi_kwargs=None, corners_list=None):
        """
        Reconstruct lensless video.

        Args:
            lensless_video (Tensor): lensless video (BxDxHxWxCxT_latent).
            roi_kwargs (dict | optional): top_left, height, and width for ROI.
            corners_list (None | list): list of coordinates for matching corners.
        Returns:
            recon_lensed_video (Tensor): reconstructed lensed video. If roi_kwargs
                are provided, the ROI part is returned.
        """
        if self.use_batch_video_version:
            # make the dims adjacent to proper reshape
            lensless_video = lensless_video.permute(0, 5, 1, 2, 3, 4).contiguous()
            B, T, D, H, W, C = lensless_video.shape
            lensless_video = lensless_video.reshape(B * T, D, H, W, C)
            recon_lensed_video = self.forward(lensless_video, roi_kwargs, corners_list)
            _, _, H, W, _ = recon_lensed_video.shape
            recon_lensed_video = recon_lensed_video.reshape(B, T, D, H, W, C)
            recon_lensed_video = recon_lensed_video.permute(
                0, 2, 3, 4, 5, 1
            ).contiguous()
            return recon_lensed_video

        # for-loop version

        recon_lensed_video = []
        for frame_index in range(lensless_video.shape[-1]):
            frame = lensless_video[..., frame_index]
            recon_lensed = self.forward(frame, roi_kwargs, corners_list).unsqueeze(-1)
            recon_lensed_video.append(recon_lensed)
        recon_lensed_video = torch.cat(recon_lensed_video, dim=-1)
        return recon_lensed_video

    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        if not isinstance(self.recon, nn.Module):
            return self.recon.__class__.__name__

        all_parameters = sum([p.numel() for p in self.recon.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.recon.parameters() if p.requires_grad]
        )

        result_info = str(self.recon)
        result_info = result_info + f"\nAll parameters: {all_parameters}"

        if hasattr(self.recon, "psf_network_model"):
            psf_network_parameters = sum(
                [p.numel() for p in self.recon.psf_network_model.parameters()]
            )
            result_info = (
                result_info + f"\nPSF Network parameters: {psf_network_parameters}"
            )

        if hasattr(self.recon, "pre_process_model"):
            pre_process_parameters = sum(
                [p.numel() for p in self.recon.pre_process_model.parameters()]
            )
            result_info = (
                result_info
                + f"\nPre Process Network parameters: {pre_process_parameters}"
            )

        if hasattr(self.recon, "post_process_model"):
            post_process_parameters = sum(
                [p.numel() for p in self.recon.post_process_model.parameters()]
            )
            result_info = (
                result_info
                + f"\nPost Process Network parameters: {post_process_parameters}"
            )

        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info
