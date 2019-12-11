from __future__ import division
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp
import math

#############################################################################
#
# SSIM & SI metrics
#
#############################################################################
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size)).cuda()
    return window


def SSIM(img1, img2):
    (_, channel, _, _) = img1.size()
    window_size = 11
    window = create_window(window_size, channel)
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    si_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)

    return ssim_map.mean(), si_map.mean()
#############################################################################
#
# NCC metrics
#
#############################################################################
def NCC(img1, img2):
    m1_sq = torch.sum(torch.pow(img1, 2))
    m2_sq = torch.sum(torch.pow(img2, 2))
    m1_m2 = torch.sum(img1*img2)
    return m1_m2 / torch.sqrt(m1_sq * m2_sq)

#############################################################################
#
# sLMSE metrics
#
#############################################################################
def ssq_error(correct, estimate):
    thr = torch.sum(estimate**2).data.cpu().numpy()
    if thr > 1e-5:
        alpha = torch.sum(correct * estimate) / torch.sum(estimate**2)
    else:
        alpha = 0.0
    return torch.sum((correct - alpha*estimate) ** 2)

def local_error(correct, estimate, window_size, window_shift):
    (_, _, M, N) = correct.size()
    ssq = total = 0.0
    for i in range(0, M - window_size + 1, window_shift):
        for j in range(0, N - window_size + 1, window_shift):
            correct_curr = correct[:, :, i:i+window_size, j:j+window_size]
            estimate_curr = estimate[:, :, i:i+window_size, j:j+window_size]
            ssq += ssq_error(correct_curr, estimate_curr)
            total += torch.sum(correct_curr**2)

    return ssq / total

def sLMSE(true_LT, estimate_LT, window_size=20):
    sLMSE = 1.0 - local_error(true_LT, estimate_LT, window_size, window_size//2)
    return sLMSE

