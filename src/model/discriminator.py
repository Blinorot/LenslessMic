import dac
import torch
from torch import nn


class Discriminator(nn.Module):
    """
    Wrapper over DAC discriminator
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.discriminator = dac.model.Discriminator(**kwargs)

    def forward(self, recon_audio, codec_audio):
        d_fmaps_recon = self.discriminator(recon_audio)
        d_fmaps_codec = self.discriminator(codec_audio)
        return {"d_fmaps_recon": d_fmaps_recon, "d_fmaps_codec": d_fmaps_codec}

    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info
