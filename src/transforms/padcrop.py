import numpy as np
import torch
from torch import nn


class PadCrop(nn.Module):
    """
    Pad or crop video to the fixed length.
    Pad or crop audio accordingly.
    """

    def __init__(
        self, length, pad_format, random_crop=False, ratio=None, frames_per_lensless=1
    ):
        """
        Args:
            length (int): number of video frames to keep
            pad_format (str): replicated, zeros, etc.
            random_crop (bool): whether to use random crop.
            ratio (int | None): ratio between audio and codec time.
            frames_per_lensless (int): number of frames per lensless frame.
                Use when grouped.
        """
        super().__init__()

        self.length = length
        self.pad_format = pad_format
        self.random_crop = random_crop
        self.ratio = ratio
        self.frames_per_lensless = frames_per_lensless

    def forward(self, instance_data):
        audio = instance_data["audio"]
        lensed_codec_video = instance_data["lensed_codec_video"]
        lensless_codec_video = instance_data["lensless_codec_video"]
        min_vals = instance_data["min_vals"]
        max_vals = instance_data["max_vals"]

        if self.ratio is not None:
            ratio = self.ratio
        else:
            ratio = audio.shape[-1] / lensed_codec_video.shape[-1]

        # before doing repeat, we need to pad lensed and audio
        if self.frames_per_lensless > 1:
            # note that, for this complicated padding, we will assume that
            # n_orig_frames will include padding.
            # During inference, we do not use padcrop, so we will
            # reconstruct original audio.
            all_frames = lensless_codec_video.shape[-1] * self.frames_per_lensless
            lensed_frames = lensed_codec_video.shape[-1]
            lensed_shape = list(lensed_codec_video.shape)
            lensed_shape[-1] = all_frames
            pad_lensed_codec_video = torch.zeros(*lensed_shape)
            pad_lensed_codec_video[..., :lensed_frames] = lensed_codec_video
            lensed_codec_video = pad_lensed_codec_video

            # we will assume that audio is also zero, which is not exactly true.
            # This may negatively impact audio-based losses if we use them
            # for grouped setup
            new_audio_len = int(all_frames * ratio)
            audio_len = audio.shape[-1]
            audio_shape = list(audio.shape)
            audio_shape[-1] = new_audio_len
            new_audio = torch.zeros(*audio_shape)
            new_audio[..., :audio_len] = audio
            audio = new_audio

        if lensless_codec_video.shape[-1] < self.length:
            if self.pad_format == "replicated":
                pad_repeat_times = self.length // lensless_codec_video.shape[-1]
                if self.length % lensless_codec_video.shape[-1] != 0:
                    pad_repeat_times += 1
                repeats = [1] * len(lensless_codec_video.shape)
                repeats[-1] = pad_repeat_times

                lensed_codec_video = lensed_codec_video.repeat(*repeats)
                lensless_codec_video = lensless_codec_video.repeat(*repeats)
                audio = audio.repeat(1, pad_repeat_times)

        len_difference = lensless_codec_video.shape[-1] - self.length
        if self.random_crop and len_difference > 0:
            start = np.random.choice(len_difference + 1)
        else:
            start = 0

        end = start + self.length

        lensless_codec_video = lensless_codec_video[..., start:end]

        # 1 lensless frame has frames_per_lenless lensed ones
        start = start * self.frames_per_lensless
        end = end * self.frames_per_lensless
        lensed_codec_video = lensed_codec_video[..., start:end]
        min_vals = min_vals[..., start:end]
        max_vals = max_vals[..., start:end]

        audio_start = int(start * ratio)
        audio_end = int(end * ratio)

        audio = audio[..., audio_start:audio_end]

        instance_data.update(
            {
                "audio": audio,
                "lensed_codec_video": lensed_codec_video,
                "lensless_codec_video": lensless_codec_video,
                "min_vals": min_vals,
                "max_vals": max_vals,
                "n_orig_frames": lensed_codec_video.shape[-1],
            }
        )

        return instance_data
