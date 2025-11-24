import os
import torch
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import argparse
import numpy as np
from tqdm import tqdm
from torchvision.datasets.cifar import *

__all__ = ['create_dataloader', 'DATASETS']


MEAN_CIFAR10 = (0.4914, 0.4822, 0.4465)
STD_CIFAR10 = (0.2023, 0.1994, 0.2010)
MEAN_CIFAR100 = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
STD_CIFAR100 = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
MEAN_TinyImageNet = (0.485, 0.456, 0.406)
STD_TinyImageNet = (0.229, 0.224, 0.225)

DATASETS = {
    "cifar10": (32, 10),
    "cifar100": (32, 100),
    "tinyimagenet": (64, 200)
}


class CIFAR10_Index(datasets.CIFAR10):
    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target, index


class CIFAR100_Index(datasets.CIFAR100):
    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target, index


def create_dataloader(dataset: str, dir: str, batch_size: int =128, workers: int = 4, minival: int = None, index: bool = False):

    if "cifar10" in dataset:
        MEAN, STD, RES = MEAN_CIFAR10, STD_CIFAR10, 32
    elif "cifar100" in dataset:
        MEAN, STD, RES = MEAN_CIFAR100, STD_CIFAR100, 32
    elif "image" in dataset or "tiny" in dataset:
        MEAN, STD, RES = MEAN_TinyImageNet, STD_TinyImageNet, 64
    else:
        raise NotImplementedError(f"=> Unknown dataset: {dataset}")
    transform_train = transforms.Compose([
        transforms.RandomCrop(RES, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    if "cifar10" == dataset:
        if index:
            train_dataset = CIFAR10_Index(root=dir, train=True, download=True, transform=transform_train)
            val_dataset = CIFAR10_Index(root=dir, train=False, download=True, transform=transform_val)
        else:
            train_dataset = datasets.CIFAR10(root=dir, train=True, download=True, transform=transform_train)
            val_dataset = datasets.CIFAR10(root=dir, train=False, download=True, transform=transform_val)
    elif "cifar100" == dataset:
        if index:
            train_dataset = CIFAR100_Index(root=dir, train=True, download=True, transform=transform_train)
            val_dataset = CIFAR100_Index(root=dir, train=False, download=True, transform=transform_val)
        else:
            train_dataset = datasets.CIFAR100(root=dir, train=True, download=True, transform=transform_train)
            val_dataset = datasets.CIFAR100(root=dir, train=False, download=True, transform=transform_val)
    elif "image" in dataset or "tiny" in dataset:
        train_dataset = datasets.ImageFolder(root=dir+"/train", transform=transform_train)
        val_dataset = datasets.ImageFolder(root=dir+"/val", transform=transform_val)

    if minival is not None:
        train_dataset, minival_dataset = torch.utils.data.random_split(train_dataset, [len(train_dataset) - minival, minival])
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=workers)
        minival_loader = torch.utils.data.DataLoader(minival_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=workers)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=workers)
        return train_loader, val_loader, minival_loader
    else:
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=workers)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=workers)
        return train_loader, val_loader


def create_subloader(dataset: str, dir: str, index: list, batch_size: int =128, workers: int = 4):

    if "cifar10" in dataset:
        MEAN, STD, RES = MEAN_CIFAR10, STD_CIFAR10, 32
    elif "cifar100" in dataset:
        MEAN, STD, RES = MEAN_CIFAR100, STD_CIFAR100, 32
    elif "image" in dataset or "tiny" in dataset:
        MEAN, STD, RES = MEAN_TinyImageNet, STD_TinyImageNet, 64
    else:
        raise NotImplementedError(f"=> Unknown dataset: {dataset}")
    transform_train = transforms.Compose([
        transforms.RandomCrop(RES, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    if "cifar10" == dataset:
        train_dataset = CIFAR10(root=dir, train=True, download=False, transform=transform_train)
    elif "cifar100" == dataset:
        train_dataset = CIFAR100(root=dir, train=True, download=False, transform=transform_train)
    elif "image" in dataset or "tiny" in dataset:
        train_dataset = datasets.ImageFolder(root=dir+"/train", transform=transform_train)

    subset = torch.utils.data.Subset(train_dataset, index)
    train_loader = torch.utils.data.DataLoader(subset, batch_size=batch_size * 4, shuffle=True, pin_memory=True, num_workers=workers)

    return train_loader


def image2txt(dataset: str, dir: str, output_dir: str):
    _, val_loader = create_dataloader(dataset, dir, batch_size=1, workers=1)
    images, labels = [], []
    for img, lab in tqdm(val_loader, desc=f"=> Extract images from {dataset}"):
        img = img.view(img.size(0), -1).numpy()
        lab = lab.flatten().numpy()
        images.append(img)
        labels.append(lab)
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    output_path = os.path.join(output_dir, dataset)
    if not os.path.isdir(output_path):
        os.mkdir(output_path)
    labels = np.vstack(labels)
    images = np.vstack(images)
    np.savetxt(os.path.join(output_path, "labels.txt"), labels, fmt="%d", delimiter=",")
    np.savetxt(os.path.join(output_path, "images.txt"), images, fmt="%.6f", delimiter=",")


if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy('file_system')
    parser = argparse.ArgumentParser(description='Write images to txt files')
    parser.add_argument('-d', '--dataset', metavar='DATASET', default='cifar10', choices=['cifar10', 'cifar100'])
    parser.add_argument('--data', metavar='DIR', help='path to dataset')
    parser.add_argument('-o', '--output', metavar='DIR', help='path to txt files')
    args = parser.parse_args()
    image2txt(args.dataset, args.data, args.output)
