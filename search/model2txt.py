import os
import math
import torch
import argparse
import numpy as np
from tqdm import tqdm
import torchmetrics
from ckks import coeffs2ckks
from models import create_model
from datasets import create_dataloader, create_subloader
from models.resnet_aespa import HerPN_Fuse
from models.resnet_aespa import fuse as fuse_aespa
from models.resnet_autofhe import EvoReLU_Fuse
from models.resnet_autofhe import fuse as fuse_autofhe
from models.activation import ReLU
from datasets import DATASETS

parser = argparse.ArgumentParser(description='Write model weights to txt')
parser.add_argument('-a', '--arch', metavar='ARCH', type=str, default=None, required=True,
                    help='models: VGG11, VGG16, VGG11_AESPA, VGG16_AESPA, VGG11_AutoFHE, VGG16_AutoFHE, '
                         'ResNet20, ResNet32, ResNet44, ResNet56, ResNet110, '
                         'ResNet20_AESPA, ResNet32_AESPA, ResNet44_AESPA, ResNet56_AESPA, ResNet110_AESPA, '
                         'ResNet20_AutoFHE, ResNet32_AutoFHE, ResNet44_AutoFHE, ResNet56_AutoFHE, ResNet110_AutoFHE, '
                         '(default: None)')
parser.add_argument('-d', '--dataset', metavar='DATASET', type=str, default='cifar10', help="datasets: cifar10, cifar100, TinyImageNet (default: cifar10)")
parser.add_argument('--data', metavar='DIR', type=str, default=None, help='path to dataset (default: None)')
parser.add_argument('-o', '--output', metavar='PATH', default=None, type=str, help='path to output (default: {arch}-{dataset})')
parser.add_argument('--ckpt', type=str, metavar='PATH', default=None, help='path to ckpt (default: None)')


def model2txt(net, folder):
    for name, param in tqdm(net.state_dict().items(), desc="=> Extract weights"):
        name = name.replace('.', '_') + '.txt'
        f = open(os.path.join(folder, name), 'w')
        if param.dim() == 0:
            f.write('{:.6e}\n'.format(0.))
        else:
            param = param.view(param.size(0), -1)
            for single in list(param):
                for num in list(single):
                    f.write('{:.6e}\n'.format(num.item()))
        f.close()


def main():
    args = parser.parse_args()
    args.arch = args.arch.lower()
    args.dataset = args.dataset.lower()
    assert os.path.isfile(args.ckpt), f"=> Not found ckpt: {args.ckpt}"

    if not os.path.isdir(args.output):
        os.mkdir(args.output)
    args.output = os.path.join(args.output, f"{args.arch}-{args.dataset}")
    if not os.path.isdir(args.output):
        os.mkdir(args.output)

    if "autofhe" in args.arch:
        ckpt = torch.load(args.ckpt, map_location="cpu")
        boot = ckpt["boot"]
        args.output = os.path.join(args.output, f"boot-{boot}")
        if not os.path.isdir(args.output):
            os.mkdir(args.output)
        coeffs = ckpt["coeffs"]
        acc = ckpt["acc"]
        model = create_model(args.arch, args.dataset, coeffs.copy())
        model.load_state_dict(ckpt["ckpt"])
        _, val_loader = create_dataloader(args.dataset, args.data)
        print("=> Accuracy before fuse is {:.4f}".format(acc))
        fuse_autofhe(model)
        _, num_classes = DATASETS[args.dataset]
        accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        if torch.cuda.is_available():
            model.cuda()
            accuracy = accuracy.cuda()
        model.eval()
        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc="=> Benchmark on validation dataset after fuse"):
                if torch.cuda.is_available():
                    images = images.cuda()
                    targets = targets.cuda()
                outputs = model(images)
                accuracy.forward(outputs, targets)
        print("=> Accuracy after fuse is {:.4f}".format(accuracy.compute()))
        model.cpu()
        evorelu_in_ranges = []
        evorelu_out_ranges = []
        for m in model.modules():
            if isinstance(m, EvoReLU_Fuse):
                evorelu_in_ranges.append(m.Bin)
                evorelu_out_ranges.append(m.Bout)
        evorelu_in_ranges = np.vstack(evorelu_in_ranges)
        evorelu_out_ranges = np.vstack(evorelu_out_ranges)
        np.savetxt(args.output + "/evorelu_in_ranges.txt", evorelu_in_ranges, fmt="%.6e")
        np.savetxt(args.output + "/evorelu_out_ranges.txt", evorelu_out_ranges, fmt="%.6e")
        model2txt(model, args.output)

        # save polynomials
        out_relu = os.path.join(args.output, "evorelus")
        all_depth = []
        if not os.path.isdir(out_relu):
            os.mkdir(out_relu)
        if 'resnet' in args.arch:
            if 'resnet20' in args.arch:
                end_num = 2
            elif 'resnet32' in args.arch:
                end_num = 4
            elif 'resnet44' in args.arch:
                end_num = 6
            elif 'resnet56' in args.arch:
                end_num = 8
            elif 'resnet110' in args.arch:
                end_num = 17
            else:
                raise NotImplementedError(f"Unknown: {args.arch}")

            coef = coeffs[0]
            depth = int(sum([math.ceil(math.log2(len(c))) for c in coef]))
            if depth > 0:
                depth += 1
            all_depth.append(depth)
            with open(out_relu + f"/0.txt", "a") as fout:
                fout.write("{:d}\n".format(depth))
                if depth == 2:
                    w0 = np.loadtxt(args.output + "/relu_w0.txt")
                    w1 = np.loadtxt(args.output + "/relu_w1.txt")
                    w2 = np.loadtxt(args.output + "/relu_w2.txt")
                    np.savetxt(fout, np.expand_dims(w0, axis=0), delimiter=",")
                    np.savetxt(fout, np.expand_dims(w1, axis=0), delimiter=",")
                    np.savetxt(fout, np.expand_dims(w2, axis=0), delimiter=",")
                if depth > 2:
                    fout.write("{:d}\n".format(len(coef)))
                    for cx in coef:
                        fout.write("{:d}\n".format(len(cx) - 1))
                    for cx in coef:
                        np.savetxt(fout, np.expand_dims(cx, axis=0), delimiter=",")
            relu_index = 0
            for j in range(0, 3):
                for k in range(0, end_num + 1):
                    for i in range(2):
                        relu_index += 1
                        coef = coeffs[relu_index]
                        depth = int(sum([math.ceil(math.log2(len(c))) for c in coef]))
                        if depth > 0:
                            depth += 1
                        all_depth.append(depth)
                        with open(out_relu + f"/{relu_index}.txt", "a") as fout:
                            fout.write("{:d}\n".format(depth))
                            if depth == 2:
                                w0 = np.loadtxt(args.output + f"/layer{j+1}_{k}_relu{i+1}_w0.txt")
                                w1 = np.loadtxt(args.output + f"/layer{j+1}_{k}_relu{i+1}_w1.txt")
                                w2 = np.loadtxt(args.output + f"/layer{j+1}_{k}_relu{i+1}_w2.txt")
                                np.savetxt(fout, np.expand_dims(w0, axis=0), delimiter=",")
                                np.savetxt(fout, np.expand_dims(w1, axis=0), delimiter=",")
                                np.savetxt(fout, np.expand_dims(w2, axis=0), delimiter=",")
                            if depth > 2:
                                fout.write("{:d}\n".format(len(coef)))
                                for cx in coef:
                                    fout.write("{:d}\n".format(len(cx) - 1))
                                for cx in coef:
                                    np.savetxt(fout, np.expand_dims(cx, axis=0), delimiter=",")
            all_depth = np.asarray(all_depth)
            all_depth = all_depth.astype(int)
            np.savetxt(args.output+"/depth.txt", all_depth, fmt="%d")
            _, _, boot_loc = coeffs2ckks(args.arch, args.dataset,  coeffs)
            np.savetxt(args.output + "/boot_loc.txt", boot_loc, fmt="%d")

        elif 'vgg' in args.arch:
            if 'vgg11' in args.arch:
                end_num = 2
            else:
                raise NotImplementedError(f"Unknown: {args.arch}")

            coef = coeffs[0]
            depth = int(sum([math.ceil(math.log2(len(c))) for c in coef]))
            if depth > 0:
                depth += 1
            all_depth.append(depth)
            with open(out_relu + f"/0.txt", "a") as fout:
                fout.write("{:d}\n".format(depth))
                if depth == 2:
                    w0 = np.loadtxt(args.output + "/features_2_w0.txt")
                    w1 = np.loadtxt(args.output + "/features_2_w1.txt")
                    w2 = np.loadtxt(args.output + "/features_2_w2.txt")
                    np.savetxt(fout, np.expand_dims(w0, axis=0), delimiter=",")
                    np.savetxt(fout, np.expand_dims(w1, axis=0), delimiter=",")
                    np.savetxt(fout, np.expand_dims(w2, axis=0), delimiter=",")
                if depth > 2:
                    fout.write("{:d}\n".format(len(coef)))
                    for cx in coef:
                        fout.write("{:d}\n".format(len(cx) - 1))
                    for cx in coef:
                        np.savetxt(fout, np.expand_dims(cx, axis=0), delimiter=",")
            relu_index = 0
            for j in range(0, 3):
                for k in range(0, end_num + 1):
                    relu_index += 1
                    coef = coeffs[relu_index]
                    depth = int(sum([math.ceil(math.log2(len(c))) for c in coef]))
                    if depth > 0:
                        depth += 1
                    all_depth.append(depth)
                    with open(out_relu + f"/{relu_index}.txt", "a") as fout:
                        fout.write("{:d}\n".format(depth))
                        if depth == 2:
                            w0 = np.loadtxt(args.output + f"/features_{j*9+k*3+5}_w0.txt")
                            w1 = np.loadtxt(args.output + f"/features_{j*9+k*3+5}_w1.txt")
                            w2 = np.loadtxt(args.output + f"/features_{j*9+k*3+5}_w2.txt")
                            np.savetxt(fout, np.expand_dims(w0, axis=0), delimiter=",")
                            np.savetxt(fout, np.expand_dims(w1, axis=0), delimiter=",")
                            np.savetxt(fout, np.expand_dims(w2, axis=0), delimiter=",")
                        if depth > 2:
                            fout.write("{:d}\n".format(len(coef)))
                            for cx in coef:
                                fout.write("{:d}\n".format(len(cx) - 1))
                            for cx in coef:
                                np.savetxt(fout, np.expand_dims(cx, axis=0), delimiter=",")
            all_depth = np.asarray(all_depth)
            all_depth = all_depth.astype(int)
            np.savetxt(args.output+"/depth.txt", all_depth, fmt="%d")
            _, _, boot_loc = coeffs2ckks(args.arch, args.dataset,  coeffs)
            np.savetxt(args.output + "/boot_loc.txt", boot_loc, fmt="%d")

    elif "aespa" in args.arch:

        model = create_model(args.arch, args.dataset)
        ckpt = torch.load(args.ckpt, map_location="cpu")
        model.load_state_dict({k.replace("model.", ""): v for k, v in ckpt["state_dict"].items() if "model" in k},
                              strict=False)
        model.eval()
        train_loader, val_loader = create_dataloader(args.dataset, args.data)
        train_loader_idx, _ = create_dataloader(args.dataset, args.data, index=True)

        if torch.cuda.is_available():
            model.cuda()
        _, num_classes = DATASETS[args.dataset]
        accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        if torch.cuda.is_available():
            accuracy = accuracy.cuda()
        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc="=> Benchmark on validation dataset before fuse"):
                if torch.cuda.is_available():
                    images = images.cuda()
                    targets = targets.cuda()
                outputs = model(images)
                accuracy.forward(outputs, targets)
        print("=> Accuracy before fuse is {:.4f}".format(accuracy.compute()))
        accuracy.reset()
        fuse_aespa(model)
        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc="=> Benchmark on validation dataset after fuse"):
                if torch.cuda.is_available():
                    images = images.cuda()
                    targets = targets.cuda()
                outputs = model(images)
                accuracy.forward(outputs, targets)
        print("=> Accuracy after fuse is {:.4f}".format(accuracy.compute()))
        correct_index = []
        with torch.no_grad():
            for images, targets, index in train_loader_idx:
                if torch.cuda.is_available():
                    images = images.cuda()
                    targets = targets.cuda()
                outputs = model(images)
                logits, outputs = torch.max(outputs, dim=1)
                correct = torch.logical_and(outputs == targets, torch.isfinite(logits))
                correct = correct.cpu()
                correct_index.extend(index[correct].tolist())
        if len(correct_index) > 0:
            subloader = create_subloader(args.dataset, args.data, correct_index)
            model.enable_track_herpn()
            with torch.no_grad():
                for images, _ in subloader:
                    if torch.cuda.is_available():
                        images = images.cuda()
                        model(images)
            model.disable_track_herpn()
        print("=> {:d} training images are used to estimate HerPN range".format(len(correct_index)))
        model.cpu()
        herpn_ranges = []
        for m in model.modules():
            if isinstance(m, HerPN_Fuse):
                herpn_ranges.append(np.asarray([m.min_val.item(), m.max_val.item()]))
        herpn_ranges = np.vstack(herpn_ranges)
        B = np.max(abs(herpn_ranges), axis=1) * 1.1  # add a small margin
        np.savetxt(args.output + "/herpn_range.txt", B, fmt="%.6e")
        model2txt(model, args.output)

    else:

        model = create_model(args.arch, args.dataset)
        ckpt = torch.load(args.ckpt, map_location="cpu")
        model.load_state_dict({k.replace("model.", ""): v for k, v in ckpt["state_dict"].items() if "model" in k},
                              strict=False)
        model.eval()
        train_loader, val_loader = create_dataloader(args.dataset, args.data)

        if torch.cuda.is_available():
            model.cuda()
        model.enable_track_relu()
        with torch.no_grad():
            for images, _ in tqdm(train_loader, desc="=> Estimate ReLU domain"):
                if torch.cuda.is_available():
                    images = images.cuda()
                model(images)
        model.disable_track_relu()
        relu_domains = []
        for m in model.modules():
            if isinstance(m, ReLU):
                relu_domains.append(np.asarray([m.min_val.item(), m.max_val.item()]))
        relu_domains = np.vstack(relu_domains)
        B = np.max(abs(relu_domains)) * 1.1  # add a small margin
        np.savetxt(args.output+"/relu_domains.txt", relu_domains, fmt="%.6f", delimiter=",")
        with open(args.output+"/B.txt", "w") as f:
            f.write("{:.6f}\n".format(B))
        model2txt(model, args.output)


if __name__ == "__main__":
    main()