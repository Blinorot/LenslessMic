import dac
import torch
from torch import nn


class Discriminator(nn.Module):
    """
    Wrapper over DAC discriminator
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.discriminator = dac.model.Discriminator(**kwargs)

    def forward(self, recon_audio, codec_audio):
        d_fmaps_recon = self.discriminator(recon_audio)
        d_fmaps_codec = self.discriminator(codec_audio)
        return {"d_fmaps_recon": d_fmaps_recon, "d_fmaps_codec": d_fmaps_codec}
