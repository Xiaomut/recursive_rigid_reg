import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

sys.path.append("../")
sys.path.append("./")
from models import base_network_split, base_network_4img


class SpatialTransform(nn.Module):
    def __init__(self, mode="bilinear"):
        super(SpatialTransform, self).__init__()
        self.mode = mode

    def get_Rx(self, theta_x):
        return torch.FloatTensor([[1, 0, 0],
                                  [0,
                                   torch.cos(theta_x),
                                   torch.sin(theta_x)],
                                  [0, -torch.sin(theta_x),
                                   torch.cos(theta_x)]])

    def get_Ry(self, theta_y):
        return torch.FloatTensor([[torch.cos(theta_y), 0, -torch.sin(theta_y)],
                                  [0, 1, 0],
                                  [torch.sin(theta_y), 0,
                                   torch.cos(theta_y)]])

    def get_Rz(self, theta_z):
        return torch.FloatTensor([[torch.cos(theta_z),
                                   torch.sin(theta_z), 0],
                                  [-torch.sin(theta_z),
                                   torch.cos(theta_z), 0], [0, 0, 1]])

    # def get_theta(self, rt, i):
    #     rx = self.get_Rx(rt[i, 0])
    #     ry = self.get_Ry(rt[i, 1])
    #     rz = self.get_Rz(rt[i, 2])

    #     R = torch.FloatTensor(rx @ ry @ rz).to(rt.device)
    #     T = rt[i, -3:]

    #     theta = torch.cat([R, T.view(-1, 1)], dim=1).view(1, 3, 4)
    #     return theta

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


class RecursiveCascadeNetwork(nn.Module):
    def __init__(self,
                 n_cascades=1,
                 midch=8,
                 device=None,
                 state_dict=None,
                 testing=False):
        super(RecursiveCascadeNetwork, self).__init__()

        self.stems = []
        # See note in base_networks.py about the assumption in the image shape
        for i in range(n_cascades):
            # self.stems.append(base_network_split.VTNAffineStem(channels=midch))
            self.stems.append(base_network_4img.VTNAffineStem(channels=midch))

        for model in self.stems:
            model.to(device)

        if state_dict:
            for i, m in enumerate(self.stems):
                m.load_state_dict(state_dict[f'cascade_{i}'])

        self.reconstruction1 = SpatialTransform(mode="bilinear").to(device)

        if testing:
            for m in self.stems:
                m.eval()
            self.reconstruction1.eval()

    def forward(self, fixed, moving):

        # Affine registration
        theta = self.stems[0](fixed, moving)
        stem_results = [self.reconstruction1(moving, theta)]
        thetas = [theta]
        for model in self.stems[1:]:  # cascades
            # registration between the fixed and the warped from last cascade
            theta = model(fixed, stem_results[-1])
            stem_results.append(self.reconstruction1(stem_results[-1], theta))
            thetas.append(theta)

        return stem_results, thetas
