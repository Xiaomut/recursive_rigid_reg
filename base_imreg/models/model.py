import torch
import torch.nn as nn


def conv_block(in_channel, out_channel):
    layer = nn.Sequential(
        nn.Conv3d(in_channel, out_channel, 3, padding=1, bias=False),
        # nn.ReLU(True), nn.LeakyReLU(0.2, True),
        nn.ReLU(True),
        nn.InstanceNorm3d(out_channel),
    )
    return layer


def transition(in_channel, out_channel):
    trans_layer = nn.Sequential(
        nn.Conv3d(in_channel, out_channel, 1),
        nn.ReLU(True),
        nn.InstanceNorm3d(out_channel),
        nn.AvgPool3d(2, 2),
    )
    return trans_layer


class Denseblock(nn.Module):
    def __init__(self, in_channel, growth_rate, num_layers):
        super(Denseblock, self).__init__()
        block = []
        for i in range(num_layers):
            block.append(conv_block(in_channel, growth_rate))
            in_channel += growth_rate
        self.net = nn.Sequential(*block)

    def forward(self, x):
        for layer in self.net:
            out = layer(x)
            x = torch.cat((out, x), dim=1)
        return x


class Net(nn.Module):
    def __init__(self, mid_ch=8, growth_rate=8, layers=[1, 2, 4, 8, 16, 24]):
        super(Net, self).__init__()

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
                                 mid_ch * 32)

        self.dense6 = Denseblock(mid_ch * 32, growth_rate, layers[5])
        self.trans6 = transition(mid_ch * 32 + growth_rate * layers[5],
                                 mid_ch * 64)

        self.fc = nn.Sequential(
            nn.Linear(4096, 1024),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(1024, 256),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(256, 32),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(32, 6),
        )

    def forward(self, img_A, img_B):
        x = torch.cat([img_A, img_B], dim=1)

        x1 = self.trans1(self.dense1(self.trans0(x)))
        x2 = self.trans2(self.dense2(x1))
        x3 = self.trans3(self.dense3(x2))
        x4 = self.trans4(self.dense4(x3))
        x5 = self.trans5(self.dense5(x4))
        x6 = self.trans6(self.dense6(x5))

        out = x6.view(x.size(0), -1)
        out = self.fc(out)

        return out


if __name__ == "__main__":

    size = 256
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    shape = (size, size, size)
    x = torch.randn((1, 1, *shape)).to(device)
    x2 = torch.randn((1, 1, *shape)).to(device)

    net = Net(mid_ch=8, growth_rate=4).to(device)
    # print(net)
    y = net(x, x2)
    print(y.cpu().detach().shape)
