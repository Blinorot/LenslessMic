import io

import matplotlib.animation as animation
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import PIL
import torch
from torchvision.transforms.v2 import ToTensor

from lensless.utils.plot import plot_image as lensless_plot_image

plt.switch_backend("agg")  # fix RuntimeError: main thread is not in main loop


def plot_tensor_as_video(
    tensor, cmap="gray", interval=100, roi_kwargs=None, corners_list=None
):
    """
    Plot video corresponding to the tensor.

    Args:
        tensor (Tensor): tensor of shape (T, H, W, C).
        interval (int): interval in ms between frames.
        cmap (str): colormap for grayscale images.
        roi_kwargs (None | dict): kwargs for a rectangular box. Shows ROI.
        corners_list (None | list): list of coordinates for plotting corners.
    Returns:
        ani: animation video in the HTML5 format.
    """
    fig, ax = plt.subplots(1, 1)
    im = plt.imshow(tensor[0], cmap=cmap)

    if roi_kwargs is not None:
        # Add rectangle (adjust Y to convert top-left to bottom-left)
        y, x = roi_kwargs["top_left"]  # x correspond to width, image is H x W
        height = roi_kwargs["height"]
        width = roi_kwargs["width"]
        rect = patches.Rectangle(
            (x, y), width, height, linewidth=2, edgecolor="red", facecolor="none"
        )
        ax.add_patch(rect)

    scatter_dots = []
    if corners_list is not None:
        for py, px in corners_list:
            (dot,) = ax.plot(px, py, "go", markersize=5)  # green dot
            scatter_dots.append(dot)

    def update(frame):
        im.set_array(tensor[frame])
        if roi_kwargs is not None:
            return [im, rect] + scatter_dots
        return [im] + scatter_dots

    ani = animation.FuncAnimation(
        fig, update, frames=len(tensor), interval=interval
    )  # interval in ms

    plt.close(fig)

    return ani.to_html5_video()


# https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.convert
def rgb2gray(rgb):
    """
    Convert RGB image to grayscale.

    Args:
        rgb (Tensor): tensor of shape B x D x H x W x C.
    Returns:
        gray (Tensor): grayscale version of shape B x D x H x W x 1.
    """
    if rgb.shape[-1] == 1:  # already grayscale
        return rgb
    assert len(rgb.shape) == 5, "Input must be of shape (B, D, H, W, C)"

    weights = torch.tensor([0.2989, 0.5870, 0.1140], device=rgb.device, dtype=rgb.dtype)
    return torch.einsum("bdhwc,c->bdhw", rgb, weights).unsqueeze(-1)


def plot_triplet(axs, lensed, lensless, reconstructed, tag="", index=0):
    """
    Plot lensed, lensless, and reconstructed images next to each other.

    Args:
        lensed (Tensor): lensed image (BxDxHxWxC).
        lensless (Tensor): lensless image.
        reconstructed (Tensor): reconstructed image.
        tag (str): optional tag.
        index (int): which object from the batch to choose.
    """
    if len(tag) > 0:
        tag = tag + " "
    lensless_plot_image(lensed[index], ax=axs[0])
    axs[0].set_title(tag + "Lensed")
    lensless_plot_image(lensless[index], ax=axs[1])
    axs[1].set_title(tag + "Lensless")
    lensless_plot_image(reconstructed[index], ax=axs[2])
    axs[2].set_title(tag + "Reconstructed")


def plot_images(imgs, config, permute=True):
    """
    Combine several images into one figure.

    Args:
        imgs (Tensor): array of images (B X C x H x W)
            or (B x H x W x C).
        config (DictConfig): hydra experiment config.
        permute (bool): permute dimensions (if channels-first).
    Returns:
        image (Tensor): a single figure with imgs plotted side-to-side.
    """
    # name of each img in the array
    names = config.writer.names
    # figure size
    figsize = config.writer.figsize
    n_imgs = min(len(names), imgs.shape[0])
    fig, axes = plt.subplots(1, n_imgs, figsize=figsize)
    for i in range(n_imgs):
        # channels must be in the last dim
        img = imgs[i]
        if permute:
            img = img.permute(1, 2, 0)
        axes[i].imshow(img)
        axes[i].set_title(names[i])
        axes[i].axis("off")  # we do not need axis
    # To create a tensor from matplotlib,
    # we need a buffer to save the figure
    buf = io.BytesIO()
    fig.tight_layout()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    # convert buffer to Tensor
    image = ToTensor()(PIL.Image.open(buf))

    plt.close()

    return image
