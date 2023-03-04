import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialTransform(nn.Module):
    def __init__(self, mode="bilinear"):
        super(SpatialTransform, self).__init__()
        self.mode = mode

    def get_theta(self, rt, i):
        device_ori = rt.device
        rt = rt.to(torch.device("cpu"))
        rx = torch.cos(rt[i, 0]).repeat(4, 4) * torch.tensor(
            [[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]],
            dtype=float) + torch.sin(rt[i, 0]).repeat(4, 4) * torch.tensor(
                [[0, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 0]],
                dtype=float) + torch.tensor(
                    [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]],
                    dtype=float)

        ry = torch.cos(rt[i, 1]).repeat(4, 4) * torch.tensor(
            [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]],
            dtype=float) + torch.sin(rt[i, 1]).repeat(4, 4) * torch.tensor(
                [[0, 0, 1, 0], [0, 0, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 0]],
                dtype=float) + torch.tensor(
                    [[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]],
                    dtype=float)

        rz = torch.cos(rt[i, 2]).repeat(4, 4) * torch.tensor(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            dtype=float) + torch.sin(rt[i, 2]).repeat(4, 4) * torch.tensor(
                [[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                dtype=float) + torch.tensor(
                    [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                    dtype=float)

        # translation x
        d = rt[0, 3:].unsqueeze(1).repeat(1, 4)
        d = d * torch.tensor([[0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1]],
                             dtype=float)

        # transform matrix
        R = torch.mm(torch.mm(rx, ry), rz)
        theta = R[0:3, :] + d
        return theta.view(1, 3, 4).to(device_ori)

    def get_thetas(self, rt):
        thetas = self.get_theta(rt, 0)
        for i in range(1, rt.size(0)):
            theta = self.get_theta(rt, i)
            thetas = torch.cat([thetas, theta])
        return thetas

    def forward(self, src, rt):

        thetas = self.get_thetas(rt)

        flow = F.affine_grid(thetas, src.size(), align_corners=False).float()
        return F.grid_sample(src, flow, align_corners=False,
                             mode=self.mode)  # bilinear, nearest


def conv_block(inc, outc, k=3, s=1, p=1):
    layer = nn.Sequential(
        nn.Conv3d(inc, outc, kernel_size=k, stride=s, padding=p, bias=False),
        nn.InstanceNorm3d(outc),
        nn.ReLU(True),
    )
    return layer


def transition(inc, outc, k=2, s=2, p=0):
    trans_layer = nn.Sequential(
        nn.Conv3d(inc, outc, 1),
        nn.InstanceNorm3d(outc),
        nn.ReLU(True),
        nn.AvgPool3d(kernel_size=k, stride=s, padding=p),
    )
    return trans_layer


class Denseblock(nn.Module):
    def __init__(self, inc, growth_rate, num_layers):
        super(Denseblock, self).__init__()
        block = []
        for i in range(num_layers):
            block.append(conv_block(inc, growth_rate))
            inc += growth_rate
        self.net = nn.Sequential(*block)

    def forward(self, x):
        for layer in self.net:
            out = layer(x)
            x = torch.cat((out, x), dim=1)
        return x


class Net(nn.Module):
    def __init__(self, mid_ch=8, growth_rate=8, layers=[1, 2, 4, 8, 16, 32]):
        super(Net, self).__init__()

        # self.conv1 = conv_block(2, mid_ch, k=7, s=1, p=3)
        # self.max_pool = nn.MaxPool3d(3, 2, 1)
        self.trans0 = transition(2, mid_ch)
        self.dense1 = Denseblock(mid_ch, growth_rate, layers[0])
        self.trans1 = transition(mid_ch + growth_rate * layers[0], mid_ch * 2)

        self.dense2 = Denseblock(mid_ch * 2, growth_rate, layers[1])
        self.trans2 = transition(mid_ch * 2 + growth_rate * layers[1],
                                 mid_ch * 4)

        self.dense3 = Denseblock(mid_ch * 4, growth_rate, layers[2])
        self.trans3 = transition(mid_ch * 4 + growth_rate * layers[2],
                                 mid_ch * 8)

        self.dense4 = Denseblock(mid_ch * 8, growth_rate, layers[3])
        self.trans4 = transition(mid_ch * 8 + growth_rate * layers[3],
                                 mid_ch * 16)

        self.dense5 = Denseblock(mid_ch * 16, growth_rate, layers[4])
        self.trans5 = transition(mid_ch * 16 + growth_rate * layers[4],
                                 mid_ch * 32, 3, (1, 2, 2), 1)

        self.dense6 = Denseblock(mid_ch * 32, growth_rate, layers[5])
        self.trans6 = transition(mid_ch * 32 + growth_rate * layers[5],
                                 mid_ch * 64)

        self.fc = nn.Sequential(
            nn.Linear(4096, 1024),  # 8192
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(128, 6),
        )

        for name, param in self.named_parameters():
            if "fc" not in name:
                param.requires_grad = False

    def forward(self, img_A, img_B):
        x = torch.cat([img_A, img_B], dim=1)

        x1 = self.trans1(self.dense1(self.trans0(x)))
        x2 = self.trans2(self.dense2(x1))
        x3 = self.trans3(self.dense3(x2))
        x4 = self.trans4(self.dense4(x3))
        x5 = self.trans5(self.dense5(x4))
        x6 = self.trans6(self.dense6(x5))

        out = x6.view(x.size(0), -1)
        rt = self.fc(out)

        return rt


if __name__ == "__main__":
    x = torch.rand(1, 1, 140, 256, 256)

    model = Net()
    print(model)
