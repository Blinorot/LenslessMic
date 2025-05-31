import torch
from torch import nn


class MeanStdNormalize(nn.Module):
    """
    Mean and Std based tensor normalization.
    """

    def __init__(self, mean, std):
        """
        Args:
            mean (float): mean used in the normalization.
            std (float): std used in the normalization.
        """
        super().__init__()

        self.mean = mean
        self.std = std

    def forward(self, x):
        """
        Args:
            x (Tensor): input tensor.
        Returns:
            x (Tensor): normalized tensor.
        """
        x = (x - self.mean) / self.std
        return x


class MinMaxNormalize(nn.Module):
    """
    Min and Max normalization to [0, 1].
    """

    def __init__(self, min=None, max=None, dim=None):
        """
        If min/max is None, it is calculated during forward operation.

        Args:
            min (float | Tensor | None): min used in the normalization.
            max (float | Tensor | None): max used in the normalization.
            dim (int | tuple | None): whether to do normalization across dim
                instead of general scalar normalization. Min/max should
                be None or Tensors in this case. For example, used when we
                want individual max/min for B and C dims.
        """
        super().__init__()

        self.min = min
        self.max = max
        self.dim = dim

    def set_min_max(self, min, max):
        """
        Set new min/max.

        Args:
            min (float | Tensor | None): min used in the normalization.
            max (float | Tensor | None): max used in the normalization.
        """
        self.min = min
        self.max = max

    def forward(self, x, return_min_max_values=False):
        """
        Normalize input.

        Args:
            x (Tensor): input tensor.
            return_min_max_values (bool): whether to return min/max vals.
        Returns:
            x (Tensor): normalized tensor.
            min_val (float | Tensor): only if return_min_max_values=True.
            max_val (float | Tensor): only if return_min_max_values=True.
        """
        return self.normalize(x, return_min_max_values)

    def normalize(self, x, return_min_max_values=False):
        """
        Normalize input.

        Args:
            x (Tensor): input tensor.
            return_min_max_values (bool): whether to return min/max vals.
        Returns:
            x (Tensor): normalized tensor.
            min_val (float | Tensor): only if return_min_max_values=True.
            max_val (float | Tensor): only if return_min_max_values=True.
        """
        if self.min is None:
            if self.dim is None:
                min_val = x.amin().item()
            else:
                dims = tuple(set(range(x.ndim)) - set(self.dim))
                min_val = x.amin(dim=dims, keepdim=True)
        else:
            min_val = self.min

        if self.max is None:
            if self.dim is None:
                max_val = x.amax().item()
            else:
                dims = tuple(set(range(x.ndim)) - set(self.dim))
                max_val = x.amax(dim=dims, keepdim=True)
        else:
            max_val = self.max

        x = (x - min_val) / (max_val - min_val)
        if return_min_max_values:
            return x, min_val, max_val
        return x

    def denormalize(self, x, min_val=None, max_val=None):
        """
        Denormalize input.

        Args:
            x (Tensor): normalized tensor.
            min_val (float | Tensor | None): min values to denormalize.
            max_val (float | Tensor | None): max values to denormalize.
        Returns:
            x (Tensor): denormalized tensor.
        """
        if min_val is None:
            min_val = self.min
        if max_val is None:
            max_val = self.max
        if isinstance(min_val, torch.Tensor):
            min_val = min_val.to(x.device)
        if isinstance(max_val, torch.Tensor):
            max_val = max_val.to(x.device)
        x = x * (max_val - min_val) + min_val
        return x
