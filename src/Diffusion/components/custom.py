import torch
import torch.nn as nn
import torch.nn.functional as F


class EMA:
    """'

    Exponential Moving Average (EMA) is a method to average the weights of a model with the weights of another model.
    This is useful for model ensembling, where the weights of the model are averaged with the weights of another model.

    Args:
    beta: float
        The beta value for the EMA. This is the weight of the new model's weights in the average.

    return: None

    """

    def __init__(self, beta):
        super().__init__()

        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(
            current_model.parameters(), ma_model.parameters()
        ):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())


class SelfAttention(nn.Module):
    """
    The SelfAttention class implements a self-attention mechanism using multi-head attention, specifically designed for processing 2D data (e.g., images) in a format compatible with convolutional neural networks. This class enhances the model's ability to focus on different parts of the input image by computing attention across different positions in the input feature map.

    Attributes:
        channels (int): The number of channels in the input feature map.
        size (int): The height and width of the input feature map (assuming a square shape).
        mha (nn.MultiheadAttention): A multi-head attention layer that allows the model to jointly attend to information from different representation subspaces at different positions.
        ln (nn.LayerNorm): A layer normalization applied before the multi-head attention to stabilize and accelerate training.
        ff_self (nn.Sequential): A feed-forward network applied after the attention mechanism. It consists of layer normalization, a linear layer, a GELU activation function, and another linear layer. This network further processes the attention output.

    Parameters:
        channels (int): The number of channels in the input feature map.
        size (int): The height and width of the input feature map.

    Methods:
        forward(x): Defines the forward pass of the module. It reshapes the input tensor to a 2D format (flattening the spatial dimensions), applies layer normalization, computes self-attention using the multi-head attention layer, adds the input (residual connection), processes the result through the feed-forward network, and finally reshapes the output back to its original spatial dimensions.
    """

    def __init__(self, channels, size):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        # Reshape input and apply layer normalization
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        x_ln = self.ln(x)
        # Compute self-attention
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        # Add residual connection and process through feed-forward network
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        # Reshape output to original spatial dimensions
        return attention_value.swapaxes(2, 1).view(
            -1, self.channels, self.size, self.size
        )


class DoubleConv(nn.Module):
    """
    A DoubleConv module represents a sequence of two convolutional blocks used in U-Net architectures. Each block consists of a convolutional layer followed by batch normalization and a ReLU activation function.
    Optionally, a residual connection can be included to add the input directly to the output after the second convolutional block, which can help with gradient flow and model performance for deeper networks.

    This module is typically used in both the downsampling (contracting path) and upsampling (expansive path) parts of a U-Net architecture, providing a way to process and refine feature maps while maintaining the network's depth.

    Attributes:
        double_conv (nn.Sequential): A sequential container that holds the layers of the two convolutional blocks. Each block includes a convolutional layer, batch normalization, and ReLU activation.
        use_residual (bool): Indicates whether a residual connection is used. If True, the input is added to the output of the double_conv sequence before returning.

    Parameters:
        in_channels (int): The number of input channels to the first convolutional layer.
        out_channels (int): The number of output channels for the second convolutional layer.
        mid_channels (int, optional): The number of output channels for the first convolutional layer and input channels for the second. If not specified, it defaults to the same number as out_channels, making both convolutional blocks identical in terms of channel dimensions.
        use_residual (bool, optional): Specifies whether to include a residual connection that adds the input to the output. Defaults to False.

    Methods:
        forward(x): Defines the forward pass of the module. Takes an input tensor `x` and passes it through the two convolutional blocks. If use_residual is True, the original input is added to the output of these blocks before returning.
    """

    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class Down(nn.Module):
    """
    A Down module for a U-Net architecture, performing downsampling followed by a double convolution.

    This module first applies a max pooling operation to reduce the spatial dimensions of the input tensor by a factor of 2.
    Then, it applies two consecutive convolutional blocks. The first block uses a residual connection if specified,
    and the second block further processes the output to produce the final feature map of the desired output channels.

    Additionally, an embedding layer processes a separate input tensor `t`, which is intended to be a time or condition embedding.
    This embedding is expanded and added to the output of the convolutional blocks, allowing the module to integrate external information
    into the spatial feature maps.

    Attributes:
        maxpool_conv (nn.Sequential): A sequential container of layers comprising a 2D max pooling operation followed by
                                      two double convolution blocks. The first double convolution block can optionally
                                      include a residual connection.
        emb_layer (nn.Sequential): A sequential container of layers for processing the embedding tensor `t`. It includes
                                   a SiLU activation function followed by a linear layer that matches the output channels
                                   of the convolutional blocks.

    Parameters:
        in_channels (int): The number of input channels to the module.
        out_channels (int): The number of output channels after processing through the convolutional blocks.
        emb_dim (int, optional): The dimensionality of the input embedding tensor `t`. Defaults to 256.

    Methods:
        forward(x, t): Defines the computation performed at every call. It takes an input tensor `x` and an embedding tensor `t`,
                       applies the max pooling and convolutional operations to `x`, processes `t` through the embedding layer,
                       and adds the expanded embedding to the output feature map.
    """

    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels),
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class Up(nn.Module):
    """
    The Up module is designed for the upsampling part of a U-Net architecture. It increases the spatial dimensions of the input tensor, merges it with a skip connection from the downsampling path, and then processes the combined tensor through convolutional blocks. Additionally, it integrates an external embedding into the feature map, allowing the network to utilize additional information (e.g., time or condition embeddings) in generating the output.

    Attributes:
        up (nn.Upsample): An upsampling layer that increases the spatial dimensions of the input tensor by a factor of 2 using bilinear interpolation.
        conv (nn.Sequential): A sequential container that holds two DoubleConv modules. The first DoubleConv applies a residual connection, and the second processes the concatenated input (from the skip connection and the upsampled tensor) to refine the features.
        emb_layer (nn.Sequential): A sequential container for processing the embedding tensor `t`. It includes a SiLU activation function followed by a linear layer that matches the output channels of the convolutional blocks.

    Parameters:
        in_channels (int): The number of input channels to the module. This should match the number of channels in the tensor to be upsampled.
        out_channels (int): The number of output channels after processing through the convolutional blocks. This defines the channel size of the output tensor.
        emb_dim (int, optional): The dimensionality of the input embedding tensor `t`. Defaults to 256.

    Methods:
        forward(x, skip_x, t): Defines the computation performed at every call. It takes an input tensor `x`, a skip connection tensor `skip_x`, and an embedding tensor `t`. The method upsamples `x`, concatenates it with `skip_x`, processes the result through convolutional blocks, and integrates the processed embedding into the final feature map.
    """

    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels),
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb
