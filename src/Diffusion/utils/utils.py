import torch
import torchvision
from PIL import Image
import matplotlib.pyplot as plt

def plot_images(images):
    """
    Plots a grid of images in a single matplotlib figure.

    This function takes a batch of images, concatenates them into a single large image,
    and then plots this image. It is assumed that the images are PyTorch tensors.

    Parameters:
        images (torch.Tensor): A batch of images as a PyTorch tensor. The tensor shape
                               is expected to be (N, C, H, W), where N is the number of images,
                               C is the number of channels, H is the height, and W is the width.
    """
    plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu())
    plt.show()

def save_images(images, path, **kwargs):
    """
    Saves a grid of images as a single image file.

    This function takes a batch of images, arranges them into a grid using torchvision's
    make_grid function, and then saves the grid to a specified path. The images are
    expected to be PyTorch tensors.

    Parameters:
        images (torch.Tensor): A batch of images as a PyTorch tensor. The tensor shape
                               is expected to be (N, C, H, W), where N is the number of images,
                               C is the number of channels, H is the height, and W is the width.
        path (str): The file path where the image will be saved.
        **kwargs: Additional keyword arguments passed to torchvision.utils.make_grid function.
    """
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(path)