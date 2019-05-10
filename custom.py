import torch
import torch.nn as nn
import torch.nn.functional as F

from .dfxp import Conv2d_q, Linear_q, BatchNorm2d_q

__all__ = ['custom']


def conv5x5(bits, in_channels, out_channels, stride=1):
    return Conv2d_q(bits, in_channels, out_channels, kernel_size=5,
        stride=stride, padding=1, bias=False)


class CUSTOM_MNIST(nn.Module):
    cfg = {
        'custom': [
            6, 'M',
            16, 'M',
            120],
    }

    def __init__(self, bits, custom_name):
        super().__init__()

        self.bits = bits
        self.features = self._make_layers(self.cfg[custom_name])
        # self.fc = nn.Linear(84, 10)
        self.fc1 = Linear_q(bits, 120, 84)
        self.fc2 = Linear_q(bits, 84, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = nn.ReLU(out)
        out = self.fc2(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 1
        for x in cfg:
            if x == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers += [conv5x5(self.bits, in_channels, x),
                        nn.ReLU()]
                in_channels = x
        return nn.Sequential(*layers)

def custom(bits):
    return CUSTOM_MNIST(bits, 'custom')
