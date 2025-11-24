# Build on top of: https://github.com/kuangliu/pytorch-cifar/blob/master/models/vgg.py

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from .activation import EvoReLU, EvoReLU_Fuse
from datasets import DATASETS

cfg = {
    'vgg11_autofhe': [16, 16, 16, 'S', 32, 32, 32, 'S', 64, 64, 64],
    'vgg16_autofhe': [16, 16, 16, 16, 'S', 32, 32, 32, 32, 32, 'S', 64, 64, 64, 64, 64],
    'vgg19_autofhe': [16, 16, 16, 16, 16, 'S', 32, 32, 32, 32, 32, 32, 'S', 64, 64, 64, 64, 64, 64],
}


def _weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")


class VGGNet(nn.Module):
    def __init__(self, arch, coeffs: list, res=32, num_classes=10):
        super(VGGNet, self).__init__()
        stride = 1 if res == 32 else 2
        layers = []
        layers += [nn.Conv2d(3, 16, kernel_size=3, stride=stride, padding=1, bias=False),
                   nn.BatchNorm2d(16),
                   EvoReLU(16, coeffs.pop(0))]
        self.features = self._make_layers(layers, cfg[arch], coeffs)
        self.linear = nn.Linear(64, num_classes)
        self.apply(_weights_init)

    def forward(self, x):
        out = self.features(x)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def enable_track_relu(self):
        for m in self.modules():
            if isinstance(m, EvoReLU):
                m.track_relu = True

    def disable_track_relu(self):
        for m in self.modules():
            if isinstance(m, EvoReLU):
                m.track_relu = False

    def _make_layers(self, layers, cfg, coeffs):
        in_channels = 16
        stride = 1
        for x in cfg:
            if x == 'S':
                stride = 2
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, stride=stride, padding=1, bias=False),
                           nn.BatchNorm2d(x),
                           EvoReLU(x, coeffs.pop(0))]
                in_channels = x
                stride = 1
        return nn.Sequential(*layers)


def fuse(model: nn.Module):
    for name, m in model.named_children():
        if isinstance(m, EvoReLU):
            setattr(model, name, EvoReLU_Fuse(m))
        else:
            fuse(m)

def vgg_autofhe(arch: str, dataset: str, coeffs):
    assert arch in cfg.keys(), f"=> Unknown architecture: {arch}"
    assert dataset in DATASETS.keys(), f"=> Unknown dataset: {dataset}"
    res, n_class = DATASETS[dataset.lower()]
    model = VGGNet(arch, coeffs, res, n_class)
    return model
