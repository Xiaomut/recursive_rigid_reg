import torch
import torch.nn as nn
from torch.nn import ReLU, LeakyReLU, Linear, Dropout


def convolveLeakyReLU(in_channels, out_channels, kernel_size, stride):
    return nn.Sequential(
        LeakyReLU(0.1),
        # ReLU(True),
        nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding=1),
        nn.InstanceNorm3d(out_channels),  # BatchNorm3d
    )


class VTNAffineStem(nn.Module):
    def __init__(self, channels=16):
        super(VTNAffineStem, self).__init__()
        self.channels = channels

        # Network architecture
        # The first convolution's input is the concatenated image
        self.conv1 = nn.Conv3d(2, channels, 3, 2)
        self.bn1 = nn.InstanceNorm3d(channels)
        self.conv2 = convolveLeakyReLU(channels, 2 * channels, 3, 2)
        self.conv3 = convolveLeakyReLU(2 * channels, 4 * channels, 3, 2)
        self.conv3_1 = convolveLeakyReLU(4 * channels, 4 * channels, 3, 1)
        self.conv4 = convolveLeakyReLU(4 * channels, 8 * channels, 3, 2)
        self.conv4_1 = convolveLeakyReLU(8 * channels, 8 * channels, 3, 1)
        self.conv5 = convolveLeakyReLU(8 * channels, 16 * channels, 3, 2)
        self.conv5_1 = convolveLeakyReLU(16 * channels, 16 * channels, 3, 1)
        self.conv6 = convolveLeakyReLU(16 * channels, 32 * channels, 3, 2)
        self.conv6_1 = convolveLeakyReLU(32 * channels, 32 * channels, 3, 1)
        self.conv7 = convolveLeakyReLU(32 * channels, 32 * channels, 3, 2)

        self.bottleneck = nn.Sequential(
            LeakyReLU(0.1), nn.Conv3d(64 * channels, 32 * channels, 1, 1, 0),
            nn.InstanceNorm3d(32 * channels), LeakyReLU(0.1))

        self.fc_loc = nn.Sequential(
            Linear(channels * 8 * 32, 1024),
            ReLU(True),
            Dropout(0.4),
            Linear(1024, 256),
            ReLU(True),
            Dropout(0.4),
            Linear(256, 6),
        )

        for name, w in self.named_parameters():
            if "conv" in name:
                if 'weight' in name:
                    nn.init.xavier_normal_(w.unsqueeze(0))
                    # nn.init.kaiming_normal_(w.unsqueeze(0))
                    # nn.init.normal_(w.unsqueeze(0))
                elif 'bias' in name:
                    nn.init.constant_(w, 0)
                else:
                    pass
            if "fc" in name:
                if 'weight' in name:
                    nn.init.normal_(w, std=0.001)
                    # nn.init.xavier_normal_(w.unsqueeze(0))
                elif "bias" in name:
                    nn.init.zeros_(w)
            else:
                pass

    def forward(self, imgA, imgB, imgA_gt, imgB_gt):
        concat_image_A = torch.cat((imgA, imgA_gt), dim=1)
        x1_A = self.conv1(concat_image_A)  # [b, channal, 70, 128, 128]
        x1_A = self.bn1(x1_A)
        x2_A = self.conv2(x1_A)  # [b, channal * 2, 35, 64, 64]
        x3_A = self.conv3(x2_A)  # [b, channal * 4, 18, 32, 32]
        x3_1_A = self.conv3_1(x3_A)  # [b, channal * 4, 18, 32, 32]
        x4_A = self.conv4(x3_1_A)  # [b, channal * 8, 9, 16, 16]
        x4_1_A = self.conv4_1(x4_A)  # [b, channal * 8, 9, 16, 16]
        x5_A = self.conv5(x4_1_A)  # [b, channal * 16, 5, 8, 8]
        x5_1_A = self.conv5_1(x5_A)  # [b, channal * 16, 5, 8, 8]
        x6_A = self.conv6(x5_1_A)  # [b, channal * 32, 3, 4, 4]
        x6_1_A = self.conv6_1(x6_A)  # [b, channal * 32, 3, 4, 4]
        x7_A = self.conv7(x6_1_A)  # [b, channal * 32, 2, 2, 2]

        concat_image_B = torch.cat((imgB, imgB_gt), dim=1)
        x1_B = self.conv1(concat_image_B)  # [b, channal, 70, 128, 128]
        x1_B = self.bn1(x1_B)
        x2_B = self.conv2(x1_B)  # [b, channal * 2, 35, 64, 64]
        x3_B = self.conv3(x2_B)  # [b, channal * 4, 18, 32, 32]
        x3_1_B = self.conv3_1(x3_B)  # [b, channal * 4, 18, 32, 32]
        x4_B = self.conv4(x3_1_B)  # [b, channal * 8, 9, 16, 16]
        x4_1_B = self.conv4_1(x4_B)  # [b, channal * 8, 9, 16, 16]
        x5_B = self.conv5(x4_1_B)  # [b, channal * 16, 5, 8, 8]
        x5_1_B = self.conv5_1(x5_B)  # [b, channal * 16, 5, 8, 8]
        x6_B = self.conv6(x5_1_B)  # [b, channal * 32, 3, 4, 4]
        x6_1_B = self.conv6_1(x6_B)  # [b, channal * 32, 3, 4, 4]
        x7_B = self.conv7(x6_1_B)  # [b, channal * 32, 2, 2, 2]

        x_input = torch.cat((x7_A, x7_B), dim=1)
        # Affine transformation
        x_fc_input = self.bottleneck(x_input)
        xs = x_fc_input.view(imgA.size(0), -1)  # x6_1
        rt = self.fc_loc(xs)

        return rt
