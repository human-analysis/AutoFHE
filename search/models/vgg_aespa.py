# Build on top of: https://github.com/kuangliu/pytorch-cifar/blob/master/models/vgg.py

import torch.nn as nn
import torch.nn.functional as F
from .activation import HerPN, HerPN_Fuse
from datasets import DATASETS

cfg = {
    'vgg11_aespa': [16, 16, 16, 'S', 32, 32, 32, 'S', 64, 64, 64],
    'vgg16_aespa': [16, 16, 16, 16, 'S', 32, 32, 32, 32, 32, 'S', 64, 64, 64, 64, 64],
    'vgg19_aespa': [16, 16, 16, 16, 16, 'S', 32, 32, 32, 32, 32, 32, 'S', 64, 64, 64, 64, 64, 64],
}


class VGGNet(nn.Module):
    def __init__(self, arch, res=32, num_classes=10):
        super(VGGNet, self).__init__()
        stride = 1 if res == 32 else 2
        layers = []
        layers += [nn.Conv2d(3, 16, kernel_size=3, stride=stride, padding=1, bias=False),
                   HerPN(16)]
        self.features = self._make_layers(layers, cfg[arch])
        self.linear = nn.Linear(64, num_classes)

    def forward(self, x):
        out = self.features(x)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def enable_track_herpn(self):
        for m in self.modules():
            if isinstance(m, HerPN_Fuse):
                m.track_herpn = True

    def disable_track_herpn(self):
        for m in self.modules():
            if isinstance(m, HerPN_Fuse):
                m.track_herpn = False

    def _make_layers(self, layers, cfg):
        in_channels = 16
        stride = 1
        for x in cfg:
            if x == 'S':
                stride = 2
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, stride=stride, padding=1, bias=False),
                           HerPN(x)]
                in_channels = x
                stride = 1
        return nn.Sequential(*layers)


def vgg_aespa(arch: str, dataset: str):
    assert arch in cfg.keys(), f"=> Unknown architecture: {arch}"
    assert dataset in DATASETS.keys(), f"=> Unknown dataset: {dataset}"
    res, n_class = DATASETS[dataset.lower()]
    model = VGGNet(arch, res, n_class)
    return model
