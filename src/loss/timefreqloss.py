import functools
import math
import typing
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

# Based on DAC repo
# https://github.com/descriptinc/descript-audio-codec/blob/main/dac/nn/loss.py


class MultiScaleSTFTLoss(nn.Module):
    """Computes the multi-scale STFT loss from [1].

    Parameters
    ----------
    window_lengths : List[int], optional
        Length of each window of each STFT, by default [2048, 512]
    loss_fn : typing.Callable, optional
        How to compare each loss, by default nn.L1Loss()
    clamp_eps : float, optional
        Clamp on the log magnitude, below, by default 1e-5
    mag_weight : float, optional
        Weight of raw magnitude portion of loss, by default 1.0
    log_weight : float, optional
        Weight of log magnitude portion of loss, by default 1.0
    pow : float, optional
        Power to raise magnitude to before taking log, by default 2.0
    weight : float, optional
        Weight of this loss, by default 1.0
    match_stride : bool, optional
        Whether to match the stride of convolutional layers, by default False

    References
    ----------

    1.  Engel, Jesse, Chenjie Gu, and Adam Roberts.
        "DDSP: Differentiable Digital Signal Processing."
        International Conference on Learning Representations. 2019.

    Implementation copied from: https://github.com/descriptinc/lyrebird-audiotools/blob/961786aa1a9d628cca0c0486e5885a457fe70c1a/audiotools/metrics/spectral.py
    """

    def __init__(
        self,
        window_lengths: List[int] = [2048, 512],
        loss_fn: typing.Callable = nn.L1Loss(),
        clamp_eps: float = 1e-5,
        mag_weight: float = 1.0,
        log_weight: float = 1.0,
        pow: float = 2.0,
        weight: float = 1.0,
        match_stride: bool = False,
        window_type: str = "hann",
        padding_type: str = "reflect",
    ):
        super().__init__()
        self.stft_params = [
            {
                "window_length": w,
                "hop_length": w // 4,
                "match_stride": match_stride,
                "window_type": window_type,
                "padding_type": padding_type,
            }
            for w in window_lengths
        ]
        self.loss_fn = loss_fn
        self.log_weight = log_weight
        self.mag_weight = mag_weight
        self.clamp_eps = clamp_eps
        self.weight = weight
        self.pow = pow

    def forward(self, x, y):
        """Computes multi-scale STFT between an estimate and a reference
        signal.

        Parameters
        ----------
        x : Tensor
            Estimate signal
        y : Tensor
            Reference signal

        Returns
        -------
        torch.Tensor
            Multi-scale STFT loss.
        """
        loss = 0.0
        for s in self.stft_params:
            x_stft = stft(x, **s).abs()
            y_stft = stft(y, **s).abs()
            loss += self.log_weight * self.loss_fn(
                x_stft.clamp(self.clamp_eps).pow(self.pow).log10(),
                y_stft.clamp(self.clamp_eps).pow(self.pow).log10(),
            )
            loss += self.mag_weight * self.loss_fn(x_stft, y_stft)
        return loss


class MelSpectrogramLoss(nn.Module):
    """Compute distance between mel spectrograms. Can be used
    in a multi-scale way.

    Parameters
    ----------
    n_mels : List[int]
        Number of mels per STFT, by default [150, 80],
    window_lengths : List[int], optional
        Length of each window of each STFT, by default [2048, 512]
    loss_fn : typing.Callable, optional
        How to compare each loss, by default nn.L1Loss()
    clamp_eps : float, optional
        Clamp on the log magnitude, below, by default 1e-5
    mag_weight : float, optional
        Weight of raw magnitude portion of loss, by default 1.0
    log_weight : float, optional
        Weight of log magnitude portion of loss, by default 1.0
    pow : float, optional
        Power to raise magnitude to before taking log, by default 2.0
    weight : float, optional
        Weight of this loss, by default 1.0
    match_stride : bool, optional
        Whether to match the stride of convolutional layers, by default False

    Implementation copied from: https://github.com/descriptinc/lyrebird-audiotools/blob/961786aa1a9d628cca0c0486e5885a457fe70c1a/audiotools/metrics/spectral.py
    """

    def __init__(
        self,
        n_mels: List[int] = [150, 80],
        window_lengths: List[int] = [2048, 512],
        loss_fn: typing.Callable = nn.L1Loss(),
        clamp_eps: float = 1e-5,
        mag_weight: float = 1.0,
        log_weight: float = 1.0,
        pow: float = 2.0,
        weight: float = 1.0,
        match_stride: bool = False,
        mel_fmin: List[float] = [0.0, 0.0],
        mel_fmax: List[float] = [None, None],
        window_type: str = "hann",
        padding_type: str = "reflect",
        sample_rate: int = 16000,
    ):
        super().__init__()
        self.stft_params = [
            {
                "window_length": w,
                "hop_length": w // 4,
                "match_stride": match_stride,
                "window_type": window_type,
                "padding_type": padding_type,
            }
            for w in window_lengths
        ]
        self.n_mels = n_mels
        self.loss_fn = loss_fn
        self.clamp_eps = clamp_eps
        self.log_weight = log_weight
        self.mag_weight = mag_weight
        self.weight = weight
        self.mel_fmin = mel_fmin
        self.mel_fmax = mel_fmax
        self.pow = pow
        self.sample_rate = sample_rate

    def forward(self, x, y):
        """Computes mel loss between an estimate and a reference
        signal.

        Parameters
        ----------
        x : Tensor
            Estimate signal
        y : Tensor
            Reference signal

        Returns
        -------
        torch.Tensor
            Mel loss.
        """
        loss = 0.0
        for n_mels, fmin, fmax, s in zip(
            self.n_mels, self.mel_fmin, self.mel_fmax, self.stft_params
        ):
            x_mels = mel_spectrogram(
                x,
                sample_rate=self.sample_rate,
                n_mels=n_mels,
                mel_fmin=fmin,
                mel_fmax=fmax,
                **s
            )
            y_mels = mel_spectrogram(
                y,
                sample_rate=self.sample_rate,
                n_mels=n_mels,
                mel_fmin=fmin,
                mel_fmax=fmax,
                **s
            )

            loss += self.log_weight * self.loss_fn(
                x_mels.clamp(self.clamp_eps).pow(self.pow).log10(),
                y_mels.clamp(self.clamp_eps).pow(self.pow).log10(),
            )
            loss += self.mag_weight * self.loss_fn(x_mels, y_mels)
        return loss


@functools.lru_cache(None)
def get_window(window_type: str, window_length: int, device: str):
    """Wrapper around scipy.signal.get_window so one can also get the
    popular sqrt-hann window. This function caches for efficiency
    using functools.lru_cache.

    Parameters
    ----------
    window_type : str
        Type of window to get
    window_length : int
        Length of the window
    device : str
        Device to put window onto.

    Returns
    -------
    torch.Tensor
        Window returned by scipy.signal.get_window, as a tensor.
    """
    from scipy import signal

    if window_type == "average":
        window = np.ones(window_length) / window_length
    elif window_type == "sqrt_hann":
        window = np.sqrt(signal.get_window("hann", window_length))
    else:
        window = signal.get_window(window_type, window_length)
    window = torch.from_numpy(window).to(device).float()
    return window


def compute_stft_padding(
    signal_length: int, window_length: int, hop_length: int, match_stride: bool
):
    """Compute how the STFT should be padded, based on match_stride.

    Parameters
    ----------
    signal_length: int
        Signal length
    window_length : int
        Window length of STFT.
    hop_length : int
        Hop length of STFT.
    match_stride : bool
        Whether or not to match stride, making the STFT have the same alignment as
        convolutional layers.

    Returns
    -------
    tuple
        Amount to pad on either side of audio.
    """
    length = signal_length

    if match_stride:
        assert (
            hop_length == window_length // 4
        ), "For match_stride, hop must equal n_fft // 4"
        right_pad = math.ceil(length / hop_length) * hop_length - length
        pad = (window_length - hop_length) // 2
    else:
        right_pad = 0
        pad = 0

    return right_pad, pad


def stft(
    audio,
    window_length: int,
    hop_length: int,
    window_type: str,
    match_stride: bool,
    padding_type: str,
):
    """Computes the short-time Fourier transform of the audio data,
    with specified STFT parameters.

    Parameters
    ----------
    audio: Tensor
        Audio input.
    window_length : int
        Window length of STFT, by default ``0.032 * self.sample_rate``.
    hop_length : int
        Hop length of STFT, by default ``window_length // 4``.
    window_type : str
        Type of window to use, by default ``hann``.
    match_stride : bool
        Whether to match the stride of convolutional layers, by default False
    padding_type : str
        Type of padding to use, by default 'reflect'

    Returns
    -------
    torch.Tensor
        STFT of audio data.

    Examples
    --------
    Compute the STFT of an AudioSignal:

    >>> signal = AudioSignal(torch.randn(44100), 44100)
    >>> signal.stft()

    Vary the window and hop length:

    >>> stft_params = [STFTParams(128, 32), STFTParams(512, 128)]
    >>> for stft_param in stft_params:
    >>>     signal.stft_params = stft_params
    >>>     signal.stft()

    """
    window_length = int(window_length)
    hop_length = int(hop_length)

    window = get_window(window_type, window_length, audio.device)
    window = window.to(audio.device)

    audio_data = audio
    batch_size, num_channels, signal_length = audio_data.shape
    right_pad, pad = compute_stft_padding(
        signal_length, window_length, hop_length, match_stride
    )
    audio_data = torch.nn.functional.pad(
        audio_data, (pad, pad + right_pad), padding_type
    )
    stft_data = torch.stft(
        audio_data.reshape(-1, audio_data.shape[-1]),
        n_fft=window_length,
        hop_length=hop_length,
        window=window,
        return_complex=True,
        center=True,
    )
    _, nf, nt = stft_data.shape
    stft_data = stft_data.reshape(batch_size, num_channels, nf, nt)

    if match_stride:
        # Drop first two and last two frames, which are added
        # because of padding. Now num_frames * hop_length = num_samples.
        stft_data = stft_data[..., 2:-2]

    return stft_data


def mel_spectrogram(
    audio,
    sample_rate=16000,
    n_mels: int = 80,
    mel_fmin: float = 0.0,
    mel_fmax: float = None,
    **kwargs
):
    """Computes a Mel spectrogram.

    Parameters
    ----------
    audio: Tensor
        Input audio signal
    sample_rate: int, optional
        Sample rate, by default 16000
    n_mels : int, optional
        Number of mels, by default 80
    mel_fmin : float, optional
        Lowest frequency, in Hz, by default 0.0
    mel_fmax : float, optional
        Highest frequency, by default None
    kwargs : dict, optional
        Keyword arguments to stft().

    Returns
    -------
    torch.Tensor [shape=(batch, channels, mels, time)]
        Mel spectrogram.
    """
    stft_output = stft(audio, **kwargs)
    magnitude = torch.abs(stft_output)

    nf = magnitude.shape[2]
    mel_basis = get_mel_filters(
        sr=sample_rate,
        n_fft=2 * (nf - 1),
        n_mels=n_mels,
        fmin=mel_fmin,
        fmax=mel_fmax,
    )
    mel_basis = torch.from_numpy(mel_basis).to(audio.device)

    mel_spectrogram = magnitude.transpose(2, -1) @ mel_basis.T
    mel_spectrogram = mel_spectrogram.transpose(-1, 2)
    return mel_spectrogram


@functools.lru_cache(None)
def get_mel_filters(
    sr: int, n_fft: int, n_mels: int, fmin: float = 0.0, fmax: float = None
):
    """Create a Filterbank matrix to combine FFT bins into Mel-frequency bins.

    Parameters
    ----------
    sr : int
        Sample rate of audio
    n_fft : int
        Number of FFT bins
    n_mels : int
        Number of mels
    fmin : float, optional
        Lowest frequency, in Hz, by default 0.0
    fmax : float, optional
        Highest frequency, by default None

    Returns
    -------
    np.ndarray [shape=(n_mels, 1 + n_fft/2)]
        Mel transform matrix
    """
    from librosa.filters import mel as librosa_mel_fn

    return librosa_mel_fn(
        sr=sr,
        n_fft=n_fft,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
    )
