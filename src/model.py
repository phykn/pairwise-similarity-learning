import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, in_channels = 1, embed_dim = 256):
        super().__init__()
        self.embed_dim = embed_dim

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, 1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, 1, stride = 2),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, 3, padding = 1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 1, stride = 2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 3, padding = 1),
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )

        self.mlp = nn.Linear(64, embed_dim)

    def forward(self, x):
        x = self.net(x)
        x = x.mean(dim = [-2, -1])
        x = self.mlp(x)
        return x


class SimPLE(nn.Module):
    def __init__(
        self, 
        b_theta = 0.3,
        alpha = 0.0001, 
        r = 1.,
        m = 0., 
        lw = 1000.,
        init_bias = -10.
    ):
        super().__init__()
        self.b_theta = b_theta
        self.alpha = alpha
        self.r = r
        self.m = m
        self.lw = lw
        self.bias = nn.Parameter(init_bias + 0. * torch.Tensor(1))

    @staticmethod
    def to_polarlike(x):
        m = F.softplus(x[..., :1], beta = 1)
        return m * F.normalize(x[..., 1:], dim = -1)

    @staticmethod
    def inner_product(x1, x2, b_theta = 0.):
        m1 = torch.norm(x1, p = 2, dim = 1, keepdim = True)
        m2 = torch.norm(x2, p = 2, dim = 1, keepdim = True)
        return x1.mm(x2.t()) - (b_theta * m1).mm(m2.t())

    def forward(self, x, y, x_bank, y_bank):
        # mask
        mask_p = y.view(-1, 1).eq(y_bank.view(1, -1))
        mask_n = mask_p.logical_not()
        mask_p[:, :len(x)].fill_diagonal_(False)

        # logits
        x, x_bank = self.to_polarlike(x), self.to_polarlike(x_bank)
        logits = self.inner_product(x, x_bank, b_theta = self.b_theta)

        logits_p = torch.masked_select(logits, mask_p)
        logits_p = (logits_p - self.m + self.bias) / self.r
        logits_n = torch.masked_select(logits, mask_n)
        logits_n = (logits_n + self.m + self.bias) * self.r

        # loss
        loss_p = F.binary_cross_entropy_with_logits(logits_p, torch.ones_like(logits_p))
        loss_n = F.binary_cross_entropy_with_logits(logits_n, torch.zeros_like(logits_n))
        loss = self.alpha * loss_p + (1. - self.alpha) * loss_n

        return self.lw * loss