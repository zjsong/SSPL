"""
Construct network parts based on existing network classes.
"""


import torch
import torch.nn as nn

from .frame_net import VGG16
from .sound_net import VGGish128
from .pc_net import PCNet
from .simsiam_head import SimSiam
from .criterions import SimSiamLoss


class ModelBuilder():

    def build_frame(self, arch='vgg16', train_from_scratch=False, fine_tune=False):

        if arch == 'vgg16':
            net_frame = VGG16(train_from_scratch, fine_tune)

        else:
            raise Exception('Architecture undefined!')

        return net_frame

    def build_sound(self, arch='vggish', weights_vggish=None, out_dim=512):

        if arch == 'vggish':
            net_sound = VGGish128(weights_vggish, out_dim)
            for p in net_sound.features.parameters():
                p.requires_grad = False
            for p in net_sound.embeddings.parameters():
                p.requires_grad = False

        else:
            raise Exception('Architecture undefined!')

        return net_sound

    def build_feat_fusion_pc(self, cycs_in=4, dim_audio=128, n_fm_out=512):
        return PCNet(cycs_in=cycs_in, dim_audio=dim_audio, n_fm_out=n_fm_out)

    def build_selfsuperlearn_head(self, arch='simsiam', in_dim_proj=512):
        if arch == 'simsiam':
            net_ssl_head = SimSiam(in_dim_proj)
        else:
            raise Exception('SSL method undefined')

        return net_ssl_head

    def build_criterion(self, args):
        if args.arch_ssl_method == 'simsiam':
            loss_ssl = SimSiamLoss()
        else:
            raise Exception('Loss function should be consistent with SSL method')

        return loss_ssl
