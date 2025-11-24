import math
import numpy as np
import torch
import torch.nn as nn

__all__ = ["ReLU"]


class ReLU(nn.ReLU):
    def __init__(self, track_relu=False, inplace=True):
        super(ReLU, self).__init__(inplace)
        self.register_buffer('min_val', torch.Tensor([float('inf')]))
        self.register_buffer('max_val', torch.Tensor([float('-inf')]))
        self.track_relu = track_relu

    def forward(self, x):
        if self.track_relu:
            min_val, max_val = torch.aminmax(x.detach())
            if min_val < self.min_val:
                self.min_val.copy_(min_val)
            if max_val > self.max_val:
                self.max_val.copy_(max_val)
        y = super(ReLU, self).forward(x)
        return y


class HerPN(nn.Module):
    def __init__(self, planes, affine=True):
        super(HerPN, self).__init__()
        self.bn0 = nn.BatchNorm2d(planes, affine=False)
        self.bn1 = nn.BatchNorm2d(planes, affine=False)
        self.bn2 = nn.BatchNorm2d(planes, affine=False)
        self.weight = nn.Parameter(torch.ones(planes, 1, 1), requires_grad=affine)
        self.bias = nn.Parameter(torch.zeros(planes, 1, 1), requires_grad=affine)

    def forward(self, x):
        x0 = self.bn0(torch.ones_like(x))
        x1 = self.bn1(x)
        x2 = self.bn2((torch.square(x) - 1) / math.sqrt(2))
        out = torch.divide(x0, math.sqrt(2 * math.pi)) + torch.divide(x1, 2) + torch.divide(x2, np.sqrt(4 * math.pi))
        out = self.weight * out + self.bias
        return out


class HerPN_Fuse(nn.Module):
    def __init__(self, herpn: HerPN):
        super(HerPN_Fuse, self).__init__()
        with torch.no_grad():
            m0, v0 = herpn.bn0.running_mean, herpn.bn0.running_var
            m1, v1 = herpn.bn1.running_mean, herpn.bn1.running_var
            m2, v2 = herpn.bn2.running_mean, herpn.bn2.running_var
            g, b = herpn.weight.squeeze(), herpn.bias.squeeze()
            e = herpn.bn0.eps
            w2 = torch.divide(g, torch.sqrt(8 * math.pi * (v2 + e)))
            w1 = torch.divide(g, 2 * torch.sqrt(v1 + e))
            w0 = b + g * (torch.divide(1 - m0, torch.sqrt(2 * math.pi * (v0 + e))) - torch.divide(m1, 2 * torch.sqrt(v1 + e)) - torch.divide(1 + math.sqrt(2) * m2, torch.sqrt(8 * math.pi * (v2 + e))))
        self.w2 = nn.Parameter(w2.unsqueeze_(-1).unsqueeze_(-1))
        self.w1 = nn.Parameter(w1.unsqueeze_(-1).unsqueeze_(-1))
        self.w0 = nn.Parameter(w0.unsqueeze_(-1).unsqueeze_(-1))
        self.register_buffer('min_val', torch.Tensor([float('inf')]))
        self.register_buffer('max_val', torch.Tensor([float('-inf')]))
        self.min_val = self.min_val.to(self.w0)
        self.max_val = self.max_val.to(self.w0)
        self.track_herpn = False

    def forward(self, x):
        out = self.w2 * torch.square(x) + self.w1 * x + self.w0
        if self.track_herpn:
            out_ = out.detach()
            N, _, _, _ = out_.shape
            out_ = out_.reshape((N, -1))
            min_val_out, max_val_out = torch.aminmax(out_, dim=1)
            index = torch.logical_and(
                torch.logical_and(min_val_out > -1e3, max_val_out < 1e3),
                torch.logical_and(torch.isfinite(min_val_out), torch.isfinite(max_val_out)))
            min_val_out, max_val_out = min_val_out[index], max_val_out[index]
            if min_val_out.numel() > 0 and max_val_out.numel() > 0:
                min_val_out, max_val_out = min_val_out.min(), max_val_out.max()
                if min_val_out < self.min_val:
                    self.min_val.copy_(min_val_out)
                if max_val_out > self.max_val:
                    self.max_val.copy_(max_val_out)
        return out


class EvoReLU(ReLU):
    def __init__(self, planes, coeffs: list, track_relu=False):
        super(EvoReLU, self).__init__(track_relu=track_relu)
        deg = 1
        for c_ in coeffs:
            deg = deg * (len(c_) - 1)
        self.degree = deg
        self.coeffs = coeffs
        if self.degree == 1:
            self.bn = nn.BatchNorm2d(planes)
            self.coef2 = nn.Parameter(torch.ones(planes, 1, 1) * coeffs[0][1])
            self.coef1 = nn.Parameter(torch.ones(planes, 1, 1) * 0.5)
            self.coef0 = nn.Parameter(torch.zeros(planes, 1, 1))
        self.register_buffer('min_val_out', torch.Tensor([float('inf')]))
        self.register_buffer('max_val_out', torch.Tensor([float('-inf')]))

    def forward(self, x: torch.Tensor):
        if self.track_relu:
            x_ = x.detach()
            N, _, _, _ = x_.shape
            x_ = x_.reshape((N, -1))
            min_val, max_val = torch.aminmax(x_, dim=1)
            index = torch.logical_and(torch.logical_and(min_val > 1.5 * self.min_val, max_val < 1.5 * self.max_val),
                                      torch.logical_and(torch.isfinite(min_val), torch.isfinite(max_val)))
            min_val, max_val = min_val[index], max_val[index]
            if min_val.numel() > 0 and max_val.numel() > 0:
                min_val, max_val = min_val.min(), max_val.max()
                if min_val < self.min_val:
                    self.min_val.copy_(min_val)
                if max_val > self.max_val:
                    self.max_val.copy_(max_val)

        if self.degree == 0:
            # Pruning ReLU, SGD
            out = x
        elif self.degree == 1:
            # Square Activation, SGD
            out = x ** 2 * self.coef2 + x * self.coef1 + self.coef0
            out = self.bn(out)
        else:
            # High-degree EvoReLU, Polynomial Aware Training
            B = max(self.min_val.abs(), self.max_val.abs()) * 1.1
            x = x / B
            out = EvoReLUFunction.apply(x, self.coeffs)
            out = out * B

        if self.track_relu:
            if not torch.isfinite(self.min_val_out) or not torch.isfinite(self.max_val_out):
                self.min_val_out.copy_(self.min_val)
                self.max_val_out.copy_(self.max_val)

            out_ = out.detach()
            N, _, _, _ = out_.shape
            out_ = out_.reshape((N, -1))
            min_val_out, max_val_out = torch.aminmax(out_, dim=1)
            index = torch.logical_and(torch.logical_and(min_val_out > 1.5 * self.min_val_out, max_val_out < 1.5 * self.max_val_out),
                                      torch.logical_and(torch.isfinite(min_val_out), torch.isfinite(max_val_out)))
            min_val_out, max_val_out = min_val_out[index], max_val_out[index]
            if min_val_out.numel() > 0 and max_val_out.numel() > 0:
                min_val_out, max_val_out = min_val_out.min(), max_val_out.max()
                if min_val_out < self.min_val_out:
                    self.min_val_out.copy_(min_val_out)
                if max_val_out > self.max_val_out:
                    self.max_val_out.copy_(max_val_out)

        return out


class EvoReLU_Fuse(nn.Module):
    def __init__(self, evorelu: EvoReLU):
        super(EvoReLU_Fuse, self).__init__()
        self.degree = evorelu.degree
        self.coeffs = evorelu.coeffs
        self.register_buffer('Bin', max(evorelu.min_val.abs(), evorelu.max_val.abs()) * 1.1)
        self.register_buffer('Bout', max(evorelu.min_val_out.abs(), evorelu.max_val_out.abs()) * 1.1)

        if self.degree == 1:
            with torch.no_grad():
                mean, var = evorelu.bn.running_mean, evorelu.bn.running_var
                gamma, beta = evorelu.bn.weight, evorelu.bn.bias
                e = evorelu.bn.eps
                c2 = evorelu.coef2.squeeze()
                c1 = evorelu.coef1.squeeze()
                c0 = evorelu.coef0.squeeze()
                w2 = torch.divide(c2 * gamma, torch.sqrt(var + e))
                w1 = torch.divide(c1 * gamma, torch.sqrt(var + e))
                w0 = torch.divide((c0 - mean) * gamma, torch.sqrt(var + e)) + beta
            self.w2 = nn.Parameter(w2.unsqueeze_(-1).unsqueeze_(-1))
            self.w1 = nn.Parameter(w1.unsqueeze_(-1).unsqueeze_(-1))
            self.w0 = nn.Parameter(w0.unsqueeze_(-1).unsqueeze_(-1))

    def forward(self, x: torch.Tensor):
        if self.degree == 0:
            out = x
        elif self.degree == 1:
            out = x ** 2 * self.w2 + x * self.w1 + self.w0
        else:
            x = x / self.Bin
            out = EvoReLUFunction.apply(x, self.coeffs)
            out = out * self.Bin

        return out


class EvoReLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, coeffs):
        ctx.save_for_backward(x)
        out = x
        for coef in coeffs:
            out = eval_poly(out, coef)
        out = (out + 0.5) * x
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        grad = torch.ones_like(x)
        grad[x < 0] = 0
        grad = grad * grad_output
        return grad, None


def eval_poly(x: torch.Tensor, coeff: list):
    coeff = coeff[1::2]
    x2 = torch.square(x)
    y = coeff[0] * x
    for c in coeff[1:]:
        x = x * x2
        y = y + c * x
    return y
