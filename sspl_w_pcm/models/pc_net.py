"""
Predictive coding module (PCM) for audio and visual feature alignment.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F


# top-down prediction process
class pred_module(nn.Module):
    def __init__(self, inchan, outchan, downsample=False):
        super(pred_module, self).__init__()
        self.conv2d = nn.Conv2d(inchan, outchan, kernel_size=3, stride=1, padding=1, bias=False)
        self.downsample = downsample
        if self.downsample:
            self.Downsample = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv2d(x)
        if self.downsample:
            x = self.Downsample(x)
        return x


# bottom-up error propagation process
class error_module(nn.Module):
    def __init__(self, inchan, outchan, upsample=False, scale_factor=2):
        super(error_module, self).__init__()
        self.convtrans2d = nn.ConvTranspose2d(inchan, outchan, kernel_size=3, stride=1, padding=1, bias=False)
        self.upsample = upsample
        if self.upsample:
            self.Upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)

    def forward(self, x):
        if self.upsample:
            x = self.Upsample(x)
        x = self.convtrans2d(x)
        return x


# input feature map: 14 x 14
class PCNet(nn.Module):
    def __init__(self, cycs_in=4, dim_audio=128, n_fm_out=512):
        super(PCNet, self).__init__()
        self.cycs_in = cycs_in

        self.in_channels =  [dim_audio, 512,      512,        512]
        self.out_channels = [512,       512,      512,        n_fm_out]
        # in->out            3x3->3x3,  3x3->7x7, 7x7->14x14, 14x14->14x14
        sample_flag = [False, True, True, False]
        self.num_layers = len(self.in_channels)

        # -------------------------------------------------------
        # feedback prediction process
        # -------------------------------------------------------
        pred_bottom_layer = [nn.Conv2d(self.out_channels[0], self.in_channels[0],
                                       kernel_size=1, stride=1, padding=0)]
        self.PredProcess = nn.ModuleList(pred_bottom_layer +
                                         [pred_module(self.out_channels[i], self.in_channels[i], downsample=sample_flag[i])
                                          for i in range(1, self.num_layers - 1)])

        # -------------------------------------------------------
        # feedforward error propagation process
        # -------------------------------------------------------
        error_bottom_layer1 = [nn.ConvTranspose2d(self.in_channels[0], self.out_channels[0],
                                                  kernel_size=1, stride=1, padding=0, bias=False)]
        error_bottom_layer2 = [nn.ConvTranspose2d(self.in_channels[1], self.out_channels[1],
                                                  kernel_size=3, stride=2, padding=0, bias=False)]
        error_output_layer = [nn.ConvTranspose2d(self.in_channels[-1], self.out_channels[-1],
                                                 kernel_size=1, stride=1, padding=0)]
        self.ErrorProcess = nn.ModuleList(error_bottom_layer1 + error_bottom_layer2 +
                                          [error_module(self.in_channels[i], self.out_channels[i], upsample=sample_flag[i])
                                           for i in range(2, self.num_layers - 1)] + error_output_layer)

        # -------------------------------------------------------
        # two kinds of scalars
        # -------------------------------------------------------
        self.b0 = nn.ParameterList([nn.Parameter(torch.zeros(1, self.in_channels[i], 1, 1) + 0.5)
                                    for i in range(1, self.num_layers)])
        self.a0 = nn.ParameterList([nn.Parameter(torch.zeros(1, self.out_channels[i], 1, 1) + 1.0)
                                    for i in range(self.num_layers - 1)])

        # -------------------------------------------------------
        # batch norm
        # -------------------------------------------------------
        # for representation initialization
        self.BNPred = nn.ModuleList([nn.BatchNorm2d(self.in_channels[i]) for i in range(1, self.num_layers - 1)])
        self.BNError = nn.ModuleList([nn.BatchNorm2d(self.out_channels[i]) for i in range(self.num_layers - 1)])
        # for representation updates at each time step
        BNPred_step = []
        BNError_step = []
        for t in range(cycs_in):
            BNPred_step = BNPred_step + [nn.BatchNorm2d(self.in_channels[i]) for i in range(1, self.num_layers - 1)]
            BNError_step = BNError_step + [nn.BatchNorm2d(self.out_channels[i]) for i in range(self.num_layers - 1)]
        self.BNPred_step = nn.ModuleList(BNPred_step)
        self.BNError_step = nn.ModuleList(BNError_step)

    def forward(self, vis_fm, audio_feature_orig):

        # representation initialization (feedback process)
        r_pred = [vis_fm]
        for i in range(self.num_layers - 2, 0, -1):
            r_pred = [F.gelu(self.BNPred[i - 1](self.PredProcess[i](r_pred[0])))] + r_pred
        # predict audio feature
        audio_feature_pred = F.gelu(self.PredProcess[0](r_pred[0]))   # B x C_audio x 3 x 3
        audio_feature_pred = F.adaptive_avg_pool2d(audio_feature_pred, 1).view(audio_feature_pred.size(0), -1)

        # representation initialization (feedforward process)
        pred_error_audio = audio_feature_orig - audio_feature_pred    # B x C_audio
        pred_error_audio = pred_error_audio.unsqueeze(-1).unsqueeze(-1)
        pred_error_audio = pred_error_audio.expand(-1, -1, 3, 3)
        a0 = F.relu(self.a0[0]).expand_as(r_pred[0])
        r_update = [F.gelu(self.BNError[0](r_pred[0] + a0 * self.ErrorProcess[0](pred_error_audio)))]
        for i in range(1, self.num_layers - 1):
            pred_error = r_update[i - 1] - r_pred[i - 1]
            a0 = F.relu(self.a0[i]).expand_as(r_pred[i])
            r_update.append(F.gelu(self.BNError[i](r_pred[i] + a0 * self.ErrorProcess[i](pred_error))))

        for t in range(self.cycs_in):

            # representation updates (feedback process)
            b0 = F.relu(self.b0[-1]).expand_as(r_update[-1])
            r_update[-1] = F.gelu((1 - b0) * r_update[-1] + b0 * r_pred[-1])
            for i in range(self.num_layers - 2, 0, -1):
                r_pred[i - 1] = self.PredProcess[i](r_update[i])
                b0 = F.relu(self.b0[i - 1]).expand_as(r_update[i - 1])
                r_update[i - 1] = F.gelu(
                    self.BNPred_step[(self.num_layers-2)*t+i-1]((1 - b0) * r_update[i - 1] + b0 * r_pred[i - 1]))
            # predict audio feature
            audio_feature_pred = F.gelu(self.PredProcess[0](r_update[0]))
            audio_feature_pred = F.adaptive_avg_pool2d(audio_feature_pred, 1).view(audio_feature_pred.size(0), -1)

            # representation updates (feedforward process)
            pred_error_audio = audio_feature_orig - audio_feature_pred    # B x C_audio
            pred_error_audio = pred_error_audio.unsqueeze(-1).unsqueeze(-1)
            pred_error_audio = pred_error_audio.expand(-1, -1, 3, 3)
            a0 = F.relu(self.a0[0]).expand_as(r_update[0])
            r_update[0] = F.gelu(
                self.BNError_step[(self.num_layers-1)*t](r_update[0] + a0 * self.ErrorProcess[0](pred_error_audio)))
            for i in range(1, self.num_layers - 1):
                pred_error = r_update[i - 1] - r_pred[i - 1]
                a0 = F.relu(self.a0[i]).expand_as(r_update[i])
                r_update[i] = F.gelu(
                    self.BNError_step[(self.num_layers-1)*t+i](r_update[i] + a0 * self.ErrorProcess[i](pred_error)))

        # transformed feature
        feat_trans = self.ErrorProcess[-1](r_update[-1])   # B x C_vis x 14 x 14

        return feat_trans
