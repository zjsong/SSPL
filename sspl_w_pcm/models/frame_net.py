"""
Frame net.
"""


import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F


# VGG16
class VGG16(nn.Module):
    def __init__(self, train_from_scratch=False, fine_tune=False):
        super(VGG16, self).__init__()

        if train_from_scratch:
            original_model = torchvision.models.vgg16(pretrained=False)

        else:
            original_model = torchvision.models.vgg16(pretrained=True)
            if not fine_tune:
                for param in original_model.parameters():
                    param.requires_grad = False

        layers = list(original_model.children())[0][0:29]
        self.feat_extractor = nn.Sequential(*layers)

    def forward(self, x):
        """
        Output: B x 512 x 14 x 14, for input of size B x 3 x 224 x 224
        Output: B x 512 x 20 x 20, for input of size B x 3 x 320 x 320
        """
        return self.feat_extractor(x)
