"""
Loss functions.
"""


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


class SimSiamLoss(nn.Module):
    def __init__(self):
        super(SimSiamLoss, self).__init__()

    def forward(self, p1, p2, z1, z2):
        return (self.neg_cosine_sim(p1, z2) + self.neg_cosine_sim(p2, z1)) * 0.5

    @staticmethod
    def neg_cosine_sim(p, z, version='simplified'):
        """
        Negative cosine similarity between two normalized vectors.
        """
        if version == 'original':
            p = F.normalize(p, dim=1)  # l2-normalize
            z = F.normalize(z, dim=1)  # l2-normalize
            return -(p * z).sum(dim=1).mean()

        elif version == 'simplified':
            return -F.cosine_similarity(p, z, dim=1).mean()

        else:
            raise Exception
