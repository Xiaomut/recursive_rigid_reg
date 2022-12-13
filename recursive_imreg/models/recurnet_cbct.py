import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

sys.path.append("../")
sys.path.append("./")
from models.cbctnet import FeatureConcat


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


class RecursiveCascadeNetwork(nn.Module):
    def __init__(self,
                 n_cascades=1,
                 midch1=8,
                 midch2=32,
                 device=None,
                 normalize_features=False,
                 normalize_matches=False,
                 state_dict=None,
                 testing=False):
        super(RecursiveCascadeNetwork, self).__init__()

        self.stems = []
        # See note in base_networks.py about the assumption in the image shape
        for i in range(n_cascades):
            self.stems.append(
                FeatureConcat(midch1, midch2, normalize_features,
                              normalize_matches))

        for model in self.stems:
            model.to(device)

        if state_dict:
            for i, m in enumerate(self.stems):
                m.load_state_dict(state_dict[f'cascade_{i}'])

        self.reconstruction1 = SpatialTransform(mode="bilinear").to(device)
        self.reconstruction2 = SpatialTransform(mode="nearest").to(device)

        if testing:
            for m in self.stems:
                m.eval()
            self.reconstruction1.eval()
            self.reconstruction2.eval()

    def forward(self, fixed, moving, fixed_gt, moving_gt):

        # Affine registration
        theta = self.stems[0](fixed, moving, fixed_gt, moving_gt)
        stem_results = [self.reconstruction1(moving, theta)]
        stem_results_gt = [self.reconstruction2(moving_gt, theta)]
        thetas = [theta]
        for model in self.stems[1:]:  # cascades
            # registration between the fixed and the warped from last cascade
            theta = model(fixed, stem_results[-1], fixed_gt,
                          stem_results_gt[-1])
            stem_results.append(self.reconstruction1(stem_results[-1], theta))
            stem_results_gt.append(
                self.reconstruction2(stem_results_gt[-1], theta))
            thetas.append(theta)

        return stem_results, thetas, stem_results_gt
