import torch
from torch import nn


class ReconstructionLoss(nn.Module):
    """
    Reconstruction loss for lensless images.
    """

    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()

    def forward(self, lensed_codec_video, recon_codec_video, **batch):
        loss = self.loss(recon_codec_video, lensed_codec_video)
        return {"loss": loss}
