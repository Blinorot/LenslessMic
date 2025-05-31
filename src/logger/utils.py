import io

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import PIL
import torch
from torchvision.transforms import ToTensor

from lensless.utils.plot import plot_image as lensless_plot_image

plt.switch_backend("agg")  # fix RuntimeError: main thread is not in main loop


def plot_tensor_as_video(tensor, cmap="grey", interval=100):
    """
    Plot video corresponding to the tensor.

    Args:
        tensor (Tensor): tensor of shape (T, H, W, C).
        interval (int): interval in ms between frames.
        cmap (str): colormap for grayscale images.
    Returns:
        ani: animation video in the HTML5 format.
    """
    fig = plt.figure()
    im = plt.imshow(tensor[0], cmap=cmap)

    def update(frame):
        im.set_array(tensor[frame])
        return [im]

    ani = animation.FuncAnimation(
        fig, update, frames=len(tensor), interval=interval
    )  # interval in ms

    plt.close(fig)

    return ani.to_html5_video()


# https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.convert
def rgb2gray(rgb):
    """
    Convert rgb image to grayscale one.

    Args:
        rgb (Tensor): tensor of shape B x D x H x W x C.
    Returns:
        gray (Tensor): grayscale version of shape B x D x H x W x 1.
    """
    if rgb.shape[-1] == 1:  # already grayscale
        return rgb
    assert len(rgb.shape) == 5
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


def plot_images(imgs, config):
    """
    Combine several images into one figure.

    Args:
        imgs (Tensor): array of images (B X C x H x W).
        config (DictConfig): hydra experiment config.
    Returns:
        image (Tensor): a single figure with imgs plotted side-to-side.
    """
    # name of each img in the array
    names = config.writer.names
    # figure size
    figsize = config.writer.figsize
    fig, axes = plt.subplots(1, len(names), figsize=figsize)
    for i in range(len(names)):
        # channels must be in the last dim
        img = imgs[i].permute(1, 2, 0)
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
