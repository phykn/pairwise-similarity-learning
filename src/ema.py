import torch
from torch import nn
from copy import deepcopy
from collections import OrderedDict


class EMA(nn.Module):
    def __init__(self, model: nn.Module, decay = 0.999):
        super().__init__()
        self.decay = decay

        self.model = model
        self.model_ema = deepcopy(self.model)

        for param in self.model_ema.parameters():
            param.detach_()

    @torch.no_grad()
    def update(self):
        if not self.training:
            raise Exception("EMA update should only be called during training")

        model_params = OrderedDict(self.model.named_parameters())
        model_ema_params = OrderedDict(self.model_ema.named_parameters())

        for name, param in model_params.items():
            model_ema_params[name].sub_((1. - self.decay) * (model_ema_params[name] - param))

        model_buffers = OrderedDict(self.model.named_buffers())
        model_ema_buffers = OrderedDict(self.model_ema.named_buffers())

        for name, buffer in model_buffers.items():
            model_ema_buffers[name].copy_(buffer)

    def train(self):
        self.model_ema.train()

    def eval(self):
        self.model_ema.eval()

    def forward(self, x):
        if self.training:
            return self.model(x)
        else:
            return self.model_ema(x)
