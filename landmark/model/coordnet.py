import sys
import torch
import torch.nn as nn

sys.path.append("../")
sys.path.append("./")
from utils.util import dsnt, flat_softmax, linear_expectation, normalized_linspace
from model import resnet, unet


class CoordRegressionNetwork(nn.Module):
    def __init__(self, n_locations=1, net="unet", conv_depths=None, midch=8):
        super().__init__()
        if net == "unet":
            self.fcn = unet.UNet3D(1, midch, conv_depths=conv_depths)
        elif net == "resnet":
            self.fcn = resnet.resnet50()
        else:
            raise NotImplementedError("Please input `unet` or `resnet`")
        self.hm_conv = nn.Conv3d(midch, n_locations, kernel_size=1, bias=False)

    def forward(self, img):
        # 1. Run the images through our FCN
        fcn_out = self.fcn(img)
        # 2. Use a 1x1 conv to get one unnormalized heatmap per location
        unnormalized_heatmaps = self.hm_conv(fcn_out)
        # 3. Normalize the heatmaps
        heatmaps = flat_softmax(unnormalized_heatmaps)
        # 4. Calculate the coordinates
        coords = dsnt(heatmaps)

        return coords, heatmaps


if __name__ == "__main__":
    device = "cuda:0"
    x = torch.randn(1, 1, 128, 128, 128).to(device)
    net = CoordRegressionNetwork(n_locations=1,
                                 net="unet",
                                 conv_depths=(4, 8, 16, 32)).to(device)
    y = net(x)
    print(y[0])