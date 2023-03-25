# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.functional as f
import numpy as np

from sklearn import metrics

class DICELossMultiClass(nn.Module):

    def __init__(self):
        super(DICELossMultiClass, self).__init__()

    def forward(self, output, mask):

        probs = torch.squeeze(output, 0)
        mask = torch.squeeze(mask, 0)

        num = probs * mask
        num = torch.sum(num)

        den1 = probs * probs
        den1 = torch.sum(den1)

        den2 = mask * mask
        den2 = torch.sum(den2)

        eps = 0.0000001
        dice = 2 * ((num + eps) / (den1 + den2 + eps))

        loss = 1 - dice
        return loss


class DICELoss(nn.Module):

    def __init__(self):
        super(DICELoss, self).__init__()

    def forward(self, output, mask):

        probs = torch.squeeze(output)
        mask = torch.squeeze(mask)

        intersection = probs * mask
        intersection = torch.sum(intersection)

        den1 = probs * probs
        den1 = torch.sum(den1)

        den2 = mask * mask
        den2 = torch.sum(den2)

        eps = 0.0000001
        dice = 2 * ((intersection + eps) / (den1 + den2 + eps))

        loss = 1 - dice
        return loss


class EUCLoss(nn.Module):

    def __init__(self):
        super(EUCLoss, self).__init__()

    def forward(self, actual, target):

        assert actual.size() == target.size(), 'input tensors must have the same size'

        # Calculate Euclidean distances between actual and target locations
        diff = actual - target
        dist_sq = diff.pow(2).sum(-1, keepdim=False)
        
        losses = dist_sq.sqrt()

        return losses


class NMILoss(nn.Module):
    
    def __init__(self, device, num_bins, sigma_ratio=0.5, max_clip=1.0):
        super(NMILoss, self).__init__()
        self.num_bins = num_bins
        self.sigma_ratio = sigma_ratio
        self.max_clip = max_clip
        self.device = device

    def diff(self, t):

        # different of input
        num_t = t.size()[0]
        t1 = t[0:num_t-1]
        t2 = t[1:num_t]
        diff = t2 - t1

        return diff

    def forward(self, actual, target):

        assert actual.size() == target.size(), 'input tensors must have the same size'

        # Calculate Normalized Mutual Information betweent acutal and target images

        # bin numbers
        bin_centers = torch.linspace(0, 1, self.num_bins)
        sigma = torch.mean(self.diff(bin_centers)) * self.sigma_ratio
        preterm = 1 / (2*sigma.pow(2))

        # image clamp
        actual = torch.clamp(actual, 0, self.max_clip)
        target = torch.clamp(target, 0, self.max_clip)

        # reshape: flatten image into shape (batch_size, h*w*d*c, 1)
        actual = torch.reshape(actual, (-1, torch.prod(torch.tensor(actual.size()))))
        actual = actual.unsqueeze(2)
        target = torch.reshape(target, (-1, torch.prod(torch.tensor(target.size()))))
        target = target.unsqueeze(2)

        # voxel number
        nb_voxels = actual.size()[1]

        # reshape bin_centers to be (1, 1, B)
        o = (1, 1, self.num_bins)
        vbc = torch.reshape(bin_centers, o)
        vbc = vbc.to(device = self.device)

        # compute image terms
        I_a = torch.exp(-preterm * (actual - vbc).pow(2))
        I_an = I_a / torch.sum(I_a, dim=-1, keepdim=True)

        I_b = torch.exp(-preterm * (target - vbc).pow(2))
        I_bn = I_b / torch.sum(I_b, dim=-1, keepdim=True)

        # compute porbabilites
        I_a_permute = I_an.permute(0, 2, 1)
        pab = torch.matmul(I_a_permute, I_b) / nb_voxels
        
        # probabilities
        pa = torch.mean(I_an, dim=1, keepdim=True)
        pb = torch.mean(I_bn, dim=1, keepdim=True)

        # probabilites
        eps = 1.4e-45
        papb = torch.matmul(pa.permute(0, 2, 1), pb) + eps
        
        # mutual information
        mi = torch.sum(torch.sum((pab * torch.log(pab / papb + eps)), dim=1), dim=1)

        # mutual information loss
        losses = -mi

        return losses


class Normal_MILoss(nn.Module):
    
    def __init__(self, device, num_bins, sigma_ratio=0.5, max_clip=1.0):
        super(Normal_MILoss, self).__init__()
        self.num_bins = num_bins
        self.sigma_ratio = sigma_ratio
        self.max_clip = max_clip
        self.device = device

    def diff(self, t):

        # different of input
        num_t = t.size()[0]
        t1 = t[0:num_t-1]
        t2 = t[1:num_t]
        diff = t2 - t1

        return diff

    def forward(self, actual, target):

        assert actual.size() == target.size(), 'input tensors must have the same size'

        # Calculate Normalized Mutual Information betweent acutal and target images

        # bin numbers
        bin_centers = torch.linspace(0, 1, self.num_bins)
        sigma = torch.mean(self.diff(bin_centers)) * self.sigma_ratio
        preterm = 1 / (2*sigma.pow(2))

        # image clamp
        actual = torch.clamp(actual, 0, self.max_clip)
        target = torch.clamp(target, 0, self.max_clip)

        # reshape: flatten image into shape (batch_size, h*w*d*c, 1)
        actual = torch.reshape(actual, (-1, torch.prod(torch.tensor(actual.size()))))
        actual = actual.unsqueeze(2)
        target = torch.reshape(target, (-1, torch.prod(torch.tensor(target.size()))))
        target = target.unsqueeze(2)

        # voxel number
        nb_voxels = actual.size()[1]

        # reshape bin_centers to be (1, 1, B)
        o = (1, 1, self.num_bins)
        vbc = torch.reshape(bin_centers, o)
        vbc = vbc.to(device = self.device)

        # compute image terms
        I_a = torch.exp(-preterm * (actual - vbc).pow(2))
        I_an = I_a / torch.sum(I_a, dim=-1, keepdim=True)

        I_b = torch.exp(-preterm * (target - vbc).pow(2))
        I_bn = I_b / torch.sum(I_b, dim=-1, keepdim=True)

        # compute porbabilites
        I_a_permute = I_an.permute(0, 2, 1)
        pab = torch.matmul(I_a_permute, I_b) / nb_voxels
        
        # probabilities
        pa = torch.mean(I_an, dim=1, keepdim=True)
        pb = torch.mean(I_bn, dim=1, keepdim=True)

        # information
        eps = 1.4e-45
        ha = torch.sum(pa * torch.log(pa))
        hb = torch.sum(pb * torch.log(pb))
        hab = torch.sum(torch.sum(pab * torch.log(pab + eps), dim=1), dim=1)

        # normalized mutual information
        nmi = (ha + hb) / (hab + eps)

        # mutual information loss
        losses = -1 * nmi

        return losses
