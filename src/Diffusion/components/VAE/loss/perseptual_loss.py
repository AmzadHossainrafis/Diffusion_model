import torch
import torch.nn as nn
import torch.nn
import torchvision
import torchvision.models as models
from collections import namedtuple
import warnings
import inspect
import os

warnings.filterwarnings("ignore")


class PerceptualLoss(nn.Module):
    def __init__(self, layers=[0, 5, 10, 19, 28]):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg19(pretrained=True).features
        self.layers = layers
        self.vgg = nn.Sequential(*[vgg[i] for i in range(max(layers) + 1)])
        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, x, y):
        loss = 0
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            y = layer(y)
            if i in self.layers:
                loss += nn.functional.mse_loss(x, y)
        return loss


# Taken from https://github.com/richzhang/PerceptualSimilarity/blob/master/lpips/lpips.py

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def spatial_average(in_tens, keepdim=True):
    return in_tens.mean([2, 3], keepdim=keepdim)


class vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True):
        super(vgg16, self).__init__()
        # Load pretrained vgg model from torchvision
        vgg_pretrained_features = torchvision.models.vgg16(
            pretrained=pretrained
        ).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])

        # Freeze vgg model
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        # Return output of vgg features
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        vgg_outputs = namedtuple(
            "VggOutputs", ["relu1_2", "relu2_2", "relu3_3", "relu4_3", "relu5_3"]
        )
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)
       

        return out


# Learned perceptual metric
class LPIPS(nn.Module):
    def __init__(self, net="vgg", version="0.1", use_dropout=True):
        super(LPIPS, self).__init__()
        self.version = version
        # Imagenet normalization
        self.scaling_layer = ScalingLayer()
        ########################

        # Instantiate vgg model
        self.chns = [64, 128, 256, 512, 512]
        self.L = len(self.chns)
        self.net = vgg16(pretrained=True, requires_grad=False)

        # Add 1x1 convolutional Layers
        self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
        self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
        self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
        self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
        self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
        self.lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
        self.lins = nn.ModuleList(self.lins)
        ########################

        # Load the weights of trained LPIPS model

        model_path = os.path.abspath(
            os.path.join(
                inspect.getfile(self.__init__),
                "..",
                "vgg.pth",
            )
        )
        print("Loading model from: %s" % model_path)
        self.load_state_dict(torch.load(model_path, map_location=device), strict=False)
        ########################

        # Freeze all parameters
        self.eval()
        for param in self.parameters():
            param.requires_grad = False
        ########################

    def forward(self, in0, in1, normalize=False):
        # Scale the inputs to -1 to +1 range if needed
        if (
            normalize
        ):  # turn on this flag if input is [0,1] so it can be adjusted to [-1, +1]
            in0 = 2 * in0 - 1
            in1 = 2 * in1 - 1
        ########################

        # Normalize the inputs according to imagenet normalization
        in0_input, in1_input = self.scaling_layer(in0), self.scaling_layer(in1)
        ########################

        # Get VGG outputs for image0 and image1
        outs0, outs1 = self.net.forward(in0_input), self.net.forward(in1_input)
        feats0, feats1, diffs = {}, {}, {}
        ########################

        # Compute Square of Difference for each layer output
        for kk in range(self.L):
            feats0[kk], feats1[kk] = torch.nn.functional.normalize(
                outs0[kk], dim=1
            ), torch.nn.functional.normalize(outs1[kk])
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2
        ########################

        # 1x1 convolution followed by spatial average on the square differences
        res = [
            spatial_average(self.lins[kk](diffs[kk]), keepdim=True)
            for kk in range(self.L)
        ]
        val = 0

        # Aggregate the results of each layer
        for l in range(self.L):
            val += res[l]
        return val


class ScalingLayer(nn.Module):
    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.register_buffer(
            "shift", torch.Tensor([-0.030, -0.088, -0.188])[None, :, None, None]
        )
        self.register_buffer(
            "scale", torch.Tensor([0.458, 0.448, 0.450])[None, :, None, None]
        )

    def forward(self, inp):
        return (inp - self.shift) / self.scale


class NetLinLayer(nn.Module):
    """A single linear layer which does a 1x1 conv"""

    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()

        layers = (
            [
                nn.Dropout(),
            ]
            if (use_dropout)
            else []
        )
        layers += [
            nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        out = self.model(x)
        return out


if __name__ == "__main__":
    # Test code
    # Create a random tensor
    torch.manual_seed(0)
    x = torch.rand(1, 3, 256, 256)
    y = torch.rand(1, 3, 256, 256)
    #set seed 

    # Create the perceptual loss object
    perceptual_loss = PerceptualLoss()

    # Compute the loss
    loss = perceptual_loss(x, y)

    # Create the LPIPS object
    print(f"Perceptual Loss: {loss}")
    # Create the LPIPS object with version 0.1
    lpips = LPIPS(version="0.1")
    # Compute the loss
    loss = lpips(x, y)
    print(f"LPIPS Loss: {loss}")