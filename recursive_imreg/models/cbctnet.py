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


class FeatureExtraction(nn.Module):
    def __init__(self, channels):
        super(FeatureExtraction, self).__init__()
        self.channels = channels

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

        for name, w in self.named_parameters():
            if "conv" in name:
                if 'weight' in name:
                    nn.init.xavier_normal_(w.unsqueeze(0))
                elif 'bias' in name:
                    nn.init.constant_(w, 0)

    def forward(self, imgA, imgA_gt):
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
        # x7_A = self.conv7(x6_1_A)  # [b, channal * 32, 2, 2, 2]

        return x6_1_A


class FeatureL2Norm(nn.Module):
    def __init__(self):
        super(FeatureL2Norm, self).__init__()

    def forward(self, feature):
        epsilon = 1e-6
        norm = torch.pow(torch.sum(torch.pow(feature, 2), 1) + epsilon,
                         0.5).unsqueeze(1).expand_as(feature)
        return torch.div(feature, norm)


class FeatureCorrelation(nn.Module):
    """https://github.com/pimed/ProsRegNet/blob/master/model/ProsRegNet_model.py"""
    def __init__(self):
        super(FeatureCorrelation, self).__init__()

    def forward(self, feature_A, feature_B):

        b, c, d, h, w = feature_A.size()
        feature_A = feature_A.permute(0, 1, 4, 3,
                                      2).contiguous().view(b, c, d * h * w)
        feature_B = feature_B.view(b, c, d * h * w).transpose(1, 2)
        feature_mul = torch.bmm(feature_B, feature_A)
        correlation_tensor = feature_mul.view(b, d, h, w, d * h * w).permute(
            0, 4, 1, 2, 3)

        return correlation_tensor


class FeatureRegression(nn.Module):
    def __init__(self, mid_ch=32):
        super(FeatureRegression, self).__init__()
        self.bottleneck = nn.Sequential(
            LeakyReLU(0.1),
            # 3 * 4 * 4 = 48, 5 * 8 * 8 = 320, 2 * 2 * 2 = 8, 3 * 6 * 4 = 72
            nn.Conv3d(48, mid_ch, 1, 1, 0),
            nn.InstanceNorm3d(mid_ch),
            LeakyReLU(0.1))
        self.fc_loc = nn.Sequential(Linear(48 * mid_ch, 256), LeakyReLU(0.1),
                                    Dropout(0.3), Linear(256, 6))

        for name, w in self.named_parameters():
            if "fc" in name:
                if 'weight' in name:
                    nn.init.normal_(w, std=0.0001)
                elif "bias" in name:
                    nn.init.zeros_(w)

    def forward(self, x):

        x = self.bottleneck(x)
        x = x.view(x.size(0), -1)
        x = self.fc_loc(x)
        return x


class FeatureConcat(nn.Module):
    def __init__(self,
                 mid_ch1,
                 mid_ch2=32,
                 normalize_features=False,
                 normalize_matches=False):
        super(FeatureConcat, self).__init__()

        self.normalize_features = normalize_features
        self.normalize_matches = normalize_matches

        self.FeatureExtraction = FeatureExtraction(channels=mid_ch1)
        self.FeatureL2Norm = FeatureL2Norm()
        self.FeatureCorrelation = FeatureCorrelation()
        self.FeatureRegression = FeatureRegression(mid_ch=mid_ch2)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, imgA, imgB, imgA_gt, imgB_gt):

        feature_A = self.FeatureExtraction(imgA, imgA_gt)
        feature_B = self.FeatureExtraction(imgB, imgB_gt)  # [1, 256, 3, 4, 4]

        if self.normalize_features:
            feature_A = self.FeatureL2Norm(feature_A)
            feature_B = self.FeatureL2Norm(feature_B)

        correlation = self.FeatureCorrelation(feature_A, feature_B)
        # normalize
        if self.normalize_matches:
            correlation = self.FeatureL2Norm(self.relu(correlation))

        rt = self.FeatureRegression(correlation)
        return rt


if __name__ == "__main__":
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    x1 = torch.rand(1, 1, 140, 256, 256).to(device)
    # x2 = torch.rand(1, 1, 2, 2, 2)
    net = FeatureConcat(8).to(device)
    # print(net)
    y = net(x1, x1, x1, x1)
    print(y)
