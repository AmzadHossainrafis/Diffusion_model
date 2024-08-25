import torch
import torch.nn as nn


class Discriminator(nn.Module):
    """
    A Discriminator class for a Generative Adversarial Network (GAN).

    This class defines a convolutional neural network (CNN) based discriminator
    which is used to distinguish between real and generated images. The network
    consists of several convolutional layers followed by a LeakyReLU activation
    function.

    Attributes:
        im_channels (int): Number of input image channels (default is 3 for RGB images).
        conv_channels (list of int): List of output channels for each convolutional layer.
        kernels (list of int): List of kernel sizes for each convolutional layer.
        strides (list of int): List of stride values for each convolutional layer.
        paddings (list of int): List of padding values for each convolutional layer.
        layers (nn.ModuleList): List of sequential layers containing Conv2d and LeakyReLU.
    """

    def __init__(
        self,
        im_channels=3,
        conv_channels=[64, 128, 256],
        kernels=[4, 4, 4, 4],
        strides=[2, 2, 2, 1],
        paddings=[1, 1, 1, 1],
    ):
        """
        Initializes the Discriminator with the given parameters.

        Args:
            im_channels (int): Number of input image channels (default is 3).
            conv_channels (list of int): List of output channels for each convolutional layer.
            kernels (list of int): List of kernel sizes for each convolutional layer.
            strides (list of int): List of stride values for each convolutional layer.
            paddings (list of int): List of padding values for each convolutional layer.
        """
        super().__init__()
        self.im_channels = im_channels
        activation = nn.LeakyReLU(0.2)
        layers_dim = [self.im_channels] + conv_channels + [1]
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        layers_dim[i],
                        layers_dim[i + 1],
                        kernel_size=kernels[i],
                        stride=strides[i],
                        padding=paddings[i],
                        bias=False if i != 0 else True,
                    ),
                    activation,
                )
                for i in range(len(conv_channels) + 1)
            ]
        )

    def forward(self, x):
        """
        Defines the forward pass of the Discriminator.

        Args:
            x (torch.Tensor): Input tensor representing a batch of images.

        Returns:
            torch.Tensor: Output tensor representing the discriminator's prediction.
        """
        for layer in self.layers:
            x = layer(x)
        return x
