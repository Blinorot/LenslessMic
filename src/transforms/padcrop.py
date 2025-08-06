import numpy as np
from torch import nn


class PadCrop(nn.Module):
    """
    Pad or crop video to the fixed length.
    Pad or crop audio accordingly.
    """

    def __init__(self, length, pad_format, random_crop=False):
        """
        Args:
            length (int): number of video frames to keep
            pad_format (str): replicated, zeros, etc.
            random_crop (bool): whether to use random crop.
        """
        super().__init__()

        self.length = length
        self.pad_format = pad_format
        self.random_crop = random_crop

    def forward(self, instance_data):
        audio = instance_data["audio"]
        lensed_codec_video = instance_data["lensed_codec_video"]
        lensless_codec_video = instance_data["lensless_codec_video"]

        ratio = audio.shape[-1] / lensed_codec_video.shape[-1]

        if lensed_codec_video.shape[-1] < self.length:
            if self.pad_format == "replicated":
                pad_repeat_times = self.length // lensed_codec_video.shape[-1]
                if self.length % lensed_codec_video.shape[-1] != 0:
                    pad_repeat_times += 1
                repeats = [1] * len(lensed_codec_video.shape)
                repeats[-1] = pad_repeat_times

                lensed_codec_video = lensed_codec_video.repeat(*repeats)
                lensless_codec_video = lensless_codec_video.repeat(*repeats)
                audio = audio.repeat(1, pad_repeat_times)

        len_difference = self.length - lensed_codec_video.shape[-1]
        if self.random_crop and len_difference > 0:
            start = np.random.choice(len_difference + 1)
        else:
            start = 0

        end = start + self.length

        lensed_codec_video = lensed_codec_video[..., start:end]
        lensless_codec_video = lensless_codec_video[..., start:end]

        audio_start = int(start * ratio)
        audio_end = int(end * ratio)

        audio = audio[..., audio_start:audio_end]

        instance_data.update(
            {
                "audio": audio,
                "lensed_codec_video": lensed_codec_video,
                "lensless_codec_video": lensless_codec_video,
            }
        )

        return instance_data
