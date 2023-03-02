import torch
import torch.nn as nn
from torch.nn import ReLU, LeakyReLU, Linear, Dropout, InstanceNorm3d, Conv3d


def convolveLeakyReLU(in_c, out_c, k, s, p=1):
    return nn.Sequential(LeakyReLU(0.1), Conv3d(in_c, out_c, k, s, padding=p),
                         InstanceNorm3d(out_c))


def convolveReLU(in_c, out_c, k, s, p=1):
    return nn.Sequential(Conv3d(in_c, out_c, k, s, padding=p),
                         InstanceNorm3d(out_c), ReLU(True))


def convolve(in_c, out_c, k, s, p=1):
    return nn.Sequential(Conv3d(in_c, out_c, k, s, padding=p),
                         InstanceNorm3d(out_c))


class FeaExAndCorr(nn.Module):
    def __init__(self,
                 in_c,
                 neckin_c=None,
                 neckout_c=None,
                 norm_features=True,
                 norm_matches=True):
        super(FeaExAndCorr, self).__init__()

        self.normalize_features = norm_features
        self.normalize_matches = norm_matches

        self.conv1 = convolveLeakyReLU(in_c, in_c, 3, 1)
        self.conv2 = convolveLeakyReLU(in_c, in_c * 2, 3, 2)

        # self.neck1 = convolveLeakyReLU(neck_c, 1, 1, 1, 0)
        self.neck2 = convolve(neckin_c, neckout_c, 3, 1)

        self.FeatureL2Norm = FeatureL2Norm()
        self.FeatureCorrelation = FeatureCorrelation()

        self.relu = nn.ReLU(True)

        for name, w in self.named_parameters():
            if "conv" in name:
                if 'weight' in name:
                    nn.init.xavier_normal_(w.unsqueeze(0))
                elif 'bias' in name:
                    nn.init.constant_(w, 0)

    def forward(self, imgA, imgB):
        feature_A1 = self.conv1(imgA)
        feature_A2 = self.conv2(feature_A1)
        feature_B1 = self.conv1(imgB)
        feature_B2 = self.conv2(feature_B1)

        if self.normalize_features:
            feature_A = self.FeatureL2Norm(feature_A2)
            feature_B = self.FeatureL2Norm(feature_B2)

        correlation = self.FeatureCorrelation(feature_A, feature_B)
        # normalize
        if self.normalize_matches:
            correlation = self.FeatureL2Norm(self.relu(correlation))
            # correlation = self.FeatureL2Norm(self.relu(correlation))

        # corr = self.neck2(self.neck1(correlation))
        correlation = self.neck2(correlation)
        return correlation, feature_A2, feature_B2


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


class FeatureConcat(nn.Module):
    def __init__(self,
                 mid_ch1=8,
                 mid_ch2=32,
                 norm_features=True,
                 norm_matches=True):
        super(FeatureConcat, self).__init__()

        self.norm_features = norm_features
        self.norm_matches = norm_matches
        self.FeatureL2Norm = FeatureL2Norm()

        self.conv1 = nn.Conv3d(2, mid_ch1, 3, 2)
        self.bn1 = nn.InstanceNorm3d(mid_ch1)
        self.conv2 = convolveLeakyReLU(mid_ch1, 2 * mid_ch1, 3, 2)
        self.conv3 = convolveLeakyReLU(2 * mid_ch1, 4 * mid_ch1, 3, 2)

        self.ext1 = FeaExAndCorr(4 * mid_ch1, 2304, 64)
        self.ext2 = FeaExAndCorr(8 * mid_ch1, 320, 128)
        self.ext3 = FeaExAndCorr(16 * mid_ch1, 48, 256)

        # self.avgpool1 = nn.AvgPool3d((3, 4, 4))
        # self.avgpool2 = nn.AvgPool3d((1, 2, 2), (2, 2, 2))
        self.fea_conv1 = convolveReLU(8 * mid_ch1, 32 * mid_ch1, (7, 9, 9),
                                      (2, 3, 3))
        self.fea_conv2 = convolveReLU(16 * mid_ch1, 32 * mid_ch1, 3, 2)

        # self.fea_conv1 = convolveLeakyReLU(8 * mid_ch1, 8 * mid_ch1, 3, 1)
        # self.fea_conv2 = convolveLeakyReLU(16 * mid_ch1, 16 * mid_ch1, 3, 2)
        # self.fea_conv3 = convolveLeakyReLU(32 * mid_ch1, 32 * mid_ch1, 3, 2)
        # self.fea_conv4 = convolveLeakyReLU(64 * mid_ch1, mid_ch2, 1, 1, 0)
        self.fea_conv = convolveReLU(32 * mid_ch1, mid_ch2, 1, 1, 0)

        # add
        # self.fea_conv1 = convolveReLU(64, 64, 3, 2)
        # self.fea_conv2 = convolveReLU(64, 64, 3, 2)
        # self.fea_conv3 = convolveReLU(64, 64, 3, 2)
        # self.fea_conv4 = convolveReLU(64, mid_ch2, 1, 1, 0)

        self.active = ReLU(True)
        self.fc_loc = nn.Sequential(
            Linear(48 * mid_ch2, 256),
            LeakyReLU(0.1),
            Dropout(0.3),
            # Linear(256, 32),
            # ReLU(True),
            # Dropout(0.3),
            Linear(256, 6),
        )
        for name, w in self.named_parameters():
            if "conv" in name:
                if 'weight' in name:
                    nn.init.xavier_normal_(w.unsqueeze(0))
                elif 'bias' in name:
                    nn.init.constant_(w, 0)
            if "fc" in name:
                if 'weight' in name:
                    nn.init.normal_(w, std=0.0001)
                elif "bias" in name:
                    nn.init.zeros_(w)

    def forward(self, imgA, imgB, imgA_gt, imgB_gt):
        concat_A = torch.cat((imgA, imgA_gt), dim=1)
        prefea_A1 = self.bn1(self.conv1(concat_A))  # [b, 8, 70, 128, 128]
        prefea_A2 = self.conv3(self.conv2(prefea_A1))  # [b, 32, 18, 32, 32]

        concat_B = torch.cat((imgB, imgB_gt), dim=1)
        prefea_B1 = self.bn1(self.conv1(concat_B))  # [b, 8, 70, 128, 128]
        prefea_B2 = self.conv3(self.conv2(prefea_B1))  # [b, 32, 18, 32, 32]

        fea_corr1, prefea_A3, prefea_B3 = self.ext1(prefea_A2, prefea_B2)
        fea_corr2, prefea_A4, prefea_B4 = self.ext2(prefea_A3, prefea_B3)
        fea_corr3, prefea_A5, prefea_B5 = self.ext3(prefea_A4, prefea_B4)

        fea_corr1_down = self.FeatureL2Norm(self.fea_conv1(fea_corr1))
        fea_corr2_down = self.FeatureL2Norm(self.fea_conv2(fea_corr2))
        fea_corr_sum = fea_corr1_down + fea_corr2_down + fea_corr3
        fea_corr_sum = self.FeatureL2Norm(self.active(fea_corr_sum))
        x = self.fea_conv(fea_corr_sum)

        # fea_1 = self.fea_conv1(fea_corr1)
        # fea_cat_1 = torch.cat((fea_1, fea_corr1), dim=1)
        # # fea_cat_1 = fea_1 + fea_corr2
        # fea_2 = self.fea_conv2(self.FeatureL2Norm(fea_cat_1))
        # fea_cat_2 = torch.cat((fea_2, fea_corr2), dim=1)
        # # fea_cat_2 = fea_2 + fea_corr3
        # fea_3 = self.fea_conv3(self.FeatureL2Norm(fea_cat_2))
        # fea_cat_3 = torch.cat((fea_3, fea_corr3), dim=1)
        # x = self.active(self.fea_conv4(self.FeatureL2Norm(fea_cat_3)))

        x = x.view(x.size(0), -1)
        x = self.fc_loc(x)
        return x


if __name__ == "__main__":
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    x1 = torch.rand(1, 1, 140, 256, 256).to(device)
    # x2 = torch.rand(1, 1, 2, 2, 2)
    net = FeatureConcat(8).to(device)
    print(net)
    # net = FeaExAndCorr(2).to(device)
    y = net(x1, x1, x1, x1)
    print(y.shape)
