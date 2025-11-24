from .vgg import vgg
from .vgg_aespa import vgg_aespa
from .resnet import resnet
from .resnet_aespa import resnet_aespa
from .activation import *
from .resnet_autofhe import resnet_autofhe
from .vgg_autofhe import vgg_autofhe


def create_model(arch: str, dataset: str, coeffs=None):
    arch = arch.lower()
    dataset = dataset.lower()
    if "vgg" in arch and "aespa" in arch:
        return vgg_aespa(arch, dataset)
    elif "resnet" in arch and "aespa" in arch:
        return resnet_aespa(arch, dataset)
    elif "resnet" in arch and "autofhe" in arch:
        return resnet_autofhe(arch, dataset, coeffs)
    elif "vgg" in arch and "autofhe" in arch:
        return vgg_autofhe(arch, dataset, coeffs)
    elif "vgg" in arch:
        return vgg(arch, dataset)
    elif "resnet" in arch:
        return resnet(arch, dataset)
    else:
        raise NotImplementedError(f"=> Unknown architecture: {arch}")


arch2relu = {
    'resnet20': 19,
    'resnet32': 31,
    'resnet44': 43,
    'resnet56': 55,
    'resnet110': 109,
    'vgg11': 10,
    'vgg16': 15,
    'vgg19': 18,
    'resnet20_autofhe': 19,
    'resnet32_autofhe': 31,
    'resnet44_autofhe': 43,
    'resnet56_autofhe': 55,
    'resnet110_autofhe': 109,
    'vgg11_autofhe': 10,
    'vgg16_autofhe': 15,
    'vgg19_autofhe': 18
}
