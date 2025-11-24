# ResNet is built on top of https://github.com/snu-ccl/approxCNN/blob/main/models/resnet_cifar10.py

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from .activation import EvoReLU, EvoReLU_Fuse
from datasets import DATASETS

cfg = {
    'resnet20_autofhe': ('basic', [3, 3, 3]),
    'resnet32_autofhe': ('basic', [5, 5, 5]),
    'resnet44_autofhe': ('basic', [7, 7, 7]),
    'resnet56_autofhe': ('basic', [9, 9, 9]),
    'resnet110_autofhe': ('basic', [18, 18, 18]),
}


def _weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")


class Downsample(nn.Module):
    def __init__(self, planes):
        super(Downsample, self).__init__()
        self.planes = planes

    def forward(self, x):
        return F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, self.planes // 4, self.planes // 4), "constant", 0)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, coeffs, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = EvoReLU(planes, coeffs.pop(0))
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = EvoReLU(planes, coeffs.pop(0))
        self.planes = planes
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                self.shortcut = Downsample(planes)
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = self.relu2(out)
        return out


class ResNet(nn.Module):
    def __init__(self, arch, coeffs, res=32, num_classes=10):
        super(ResNet, self).__init__()
        block, num_blocks = cfg[arch]
        block = BasicBlock if block == 'basic' else None
        option = "A" if num_classes == 10 else "B"
        stride = 1 if res == 32 else 2
        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = EvoReLU(16, coeffs.pop(0))
        self.layer1 = self._make_layer(block, 16, num_blocks[0], coeffs, stride=1, option=option)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], coeffs, stride=2, option=option)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], coeffs, stride=2, option=option)
        self.linear = nn.Linear(64, num_classes)
        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, coeffs, stride, option):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, coeffs, stride, option))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def enable_track_relu(self):
        for m in self.modules():
            if isinstance(m, EvoReLU):
                m.track_relu = True

    def disable_track_relu(self):
        for m in self.modules():
            if isinstance(m, EvoReLU):
                m.track_relu = False

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def fuse(model: nn.Module):
    for name, m in model.named_children():
        if isinstance(m, EvoReLU):
            setattr(model, name, EvoReLU_Fuse(m))
        else:
            fuse(m)


def resnet_autofhe(arch: str, dataset: str, coeffs):
    assert arch in cfg.keys(), f"=> Unknown architecture: {arch}"
    assert dataset in DATASETS.keys(), f"=> Unknown dataset: {dataset}"
    res, n_class = DATASETS[dataset.lower()]
    model = ResNet(arch, coeffs, res, n_class)
    return model
