"""
Structure of self-supervised learning head: SimSiam.
"""


import torch.nn as nn


class projection_mlp(nn.Module):
    def __init__(self, in_dim, hidden_dim=2048, out_dim=2048):
        super().__init__()
        """
        Projection MLP. The projection MLP (in f) has BN applied 
        to each fully-connected (fc) layer, including its 
        output fc. Its output fc has no ReLU. The hidden fc is 2048-d. 
        This MLP has 3 layers.
        """
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim, bias=False),
            nn.BatchNorm1d(out_dim, affine=False)
        )

    def forward(self, x):
        if len(x.size()) != 2:
            x = x.view(x.size(0), -1)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        return x


class prediction_mlp(nn.Module):
    def __init__(self, in_dim=2048, hidden_dim=512, out_dim=2048):  # bottleneck structure
        super().__init__()
        """
        Prediction MLP. The prediction MLP (h) has BN applied 
        to its hidden fc layers. Its output fc does not have BN
        (ablation in Sec. 4.4) or ReLU. This MLP has 2 layers. 
        The dimension of h’s input and output (z and p) is d = 2048, 
        and h’s hidden layer’s dimension is 512, making h a 
        bottleneck structure (ablation in supplement). 
        """
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)

        return x


class SimSiam(nn.Module):
    def __init__(self, in_dim_proj=512):
        super(SimSiam, self).__init__()

        self.projector = projection_mlp(in_dim_proj)
        self.predictor = prediction_mlp()

    def forward(self, x1, x2):
        z1 = self.projector(x1)
        z2 = self.projector(x2)
        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        return p1, p2, z1.detach(), z2.detach()
