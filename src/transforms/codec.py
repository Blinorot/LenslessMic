import inspect

import torch
from torch import nn


class CodecEncoderDecoder(nn.Module):
    def __init__(
        self, codec_cls, codec_weights_path, codec_kwargs=None, codec_name="codec"
    ):
        """
        Args:
            codec_cls (nn.Module): class
            codec_weights_path (str): path to codec weights.pth
            codec_kwargs (dict | None): kwargs for codec class.
            codec_name (str): tag to identify codec.
        """
        super().__init__()
        checkpoint = torch.load(codec_weights_path, map_location="cpu")

        if codec_kwargs is None and "metadata" in checkpoint.keys():
            # try to get from checkpoint
            # metadata is a dict: "kwargs" key with kwargs dict as value
            codec_kwargs = checkpoint["metadata"]
            codec_kwargs["kwargs"]

            sig = inspect.signature(codec_cls)
            class_keys = list(sig.parameters.keys())
            for k in list(codec_kwargs["kwargs"].keys()):
                if k not in class_keys:
                    codec_kwargs["kwargs"].pop(k)
        else:
            error = "Codec kwargs are not provided."
            error += "Provide kwargs or use checkpoint with metadata field."
            raise ValueError(error)

        self.codec = codec_cls(**codec_kwargs["kwargs"])
        self.codec.load_state_dict(checkpoint["state_dict"])
        self.codec.metadata = codec_kwargs
        self.codec_name = codec_name

    def forward(self, audio):
        return self.codec(audio)

    def audio_to_video(self, audio):
        """
        Converts audio to img representation.
        Args:
            audio (Tensor): audio in [-1, 1] (B x 1 x T).
        Returns:
            codec_video (Tensor): latent representation of audio
                (B x 1 x sqrt(D) x sqrt(D) x 1 x T_latent).
        """
        latent_audio = self.codec.encoder(audio)
        image_shape = int(latent_audio.shape[1] ** 0.5)
        codec_video = latent_audio.reshape(
            latent_audio.shape[0], image_shape, image_shape, latent_audio.shape[-1]
        )
        # add plane and channel dim
        codec_video = codec_video.unsqueeze(1).unsqueeze(-2)
        return codec_video

    def video_to_audio(self, codec_video):
        """
        Converts video to audio representation.
        Args:
            codec_video (Tensor): latent representation of audio
                (B x 1 x sqrt(D) x sqrt(D) x 1 x T_latent).
        Returns:
            audio (Tensor): audio in [-1, 1] (B x 1 x T).
                Note: T might be a bit shorter than in the original audio.
        """
        image_shape = codec_video.shape[-3]
        latent_audio = codec_video.reshape(
            codec_video.shape[0], image_shape**2, codec_video.shape[-1]
        )
        z, _, _, _, _ = self.codec.quantizer(latent_audio, n_quantizers=None)
        audio = self.codec.decode(z)
        return audio
