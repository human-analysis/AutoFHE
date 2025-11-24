import os
import torch
import argparse
from tqdm import trange
from datasets import create_dataloader, create_subloader, DATASETS
from models import create_model
from lightning.pytorch import seed_everything
from helper import train, validate


parser = argparse.ArgumentParser('AutoFHE: Tune Polynomial CNNs')
parser.add_argument('-a', '--arch', metavar='ARCH', type=str, default=None, required=True,
                    help='models: VGG11, ResNet20, ResNet32, ResNet44 (default: None)')
parser.add_argument('-d', '--dataset', metavar='DATASET', type=str, default='cifar10',
                    help="datasets: cifar10, cifar100 (default: cifar10)")
parser.add_argument('--data', metavar='DIR', type=str, default=None, help='path to dataset (default: None)')
parser.add_argument('--ckpt', type=str, metavar='PATH', default=None, help='path to autofhe network ckpt (default: None)')
parser.add_argument('--backbone-ckpt', type=str, metavar='PATH', default=None, help='path to backbone network ckpt (default: None)')
parser.add_argument('--seed', metavar='N', type=int, default=0, help='seed for initializing training (default: 0)')
parser.add_argument('--gpu', type=int, default=None, help='GPU used to Evaluate or Finetune.')
parser.add_argument('--grad-clip', metavar='F', type=float, default=1, help='gradient clip value (default: 1.)')
parser.add_argument('--batch-size', metavar='N', default=128, type=int, help='batch size (default: 128)')
parser.add_argument('--epochs', default=90, type=int, metavar='N', help='number of total epochs to finetune (default: 90)')
parser.add_argument('-j', '--num-workers', default=16, type=int, metavar='N', help='number of workers (default: 16)')
parser.add_argument('--lr', '--learning-rate', default=0.02, type=float, metavar='LR', help='learning rate (default: 0.02)')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum (default: 0.9)')
parser.add_argument('--weight-decay', default=5e-4, type=float, help='weight decay (default: 5e-4)')
parser.add_argument('-o', '--output', metavar='PATH', default="experiments-tune", type=str, help='path to output (default: experiments-tune/{arch}-{dataset})')
parser.add_argument('--use-wandb', action='store_true', default=False)
parser.add_argument('--wandb-proj', type=str, metavar='PROJ', default='AutoFHE-Tune', help='wandb project (default: AutoFHE-Tune)')


def main():
    args = parser.parse_args()
    args.arch = args.arch.lower()
    args.dataset = args.dataset.lower()
    assert os.path.isfile(args.ckpt), f"=> Not found ckpt: {args.ckpt}"
    assert os.path.isfile(args.backbone_ckpt), f"=> Not found ckpt: {args.backbone_ckpt}"
    if not os.path.isdir(args.output):
        os.mkdir(args.output)
    dir_name = f"{args.arch}-{args.dataset}"
    args.output = os.path.join(args.output, dir_name)
    if not os.path.isdir(args.output):
        os.mkdir(args.output)
    if args.seed is not None:
        seed_everything(args.seed)
    device = torch.device("cpu")
    if torch.cuda.is_available() and args.gpu is not None:
        device = torch.device(f"cuda:{args.gpu}")
        print(f'=> GPU {args.gpu} is used to evaluate or finetune.')
    _, num_classes = DATASETS[args.dataset]

    train_loader, val_loader = create_dataloader(args.dataset, args.data, args.batch_size, args.num_workers)
    train_loader_idx, _ = create_dataloader(args.dataset, args.data, args.batch_size, args.num_workers, index=True)

    # Teacher
    teacher = create_model(args.arch, args.dataset)
    teacher_ckpt = torch.load(args.backbone_ckpt, map_location="cpu")
    teacher.load_state_dict({k.replace("model.", ""): v for k, v in teacher_ckpt["state_dict"].items() if "model" in k})
    teacher.eval()
    teacher.to(device)
    acc = validate(teacher, val_loader, num_classes, device)
    print("Teacher accuracy is: {:.2f}".format(acc))

    # Model and Dataloader
    ckpt = torch.load(args.ckpt, map_location="cpu")
    boot = ckpt["boot"]
    coeffs = ckpt["coeffs"]
    model = create_model(args.arch+"_autofhe", args.dataset, coeffs.copy())
    model.load_state_dict(ckpt["ckpt"], strict=False)
    model.to(device)

    # record
    args.boot = boot
    if args.use_wandb:
        import wandb
        wandb.init(project=args.wandb_proj, name=f"{args.arch}-{args.dataset}-boot{boot}", config=vars(args))

    # training setup
    optimizer = torch.optim.SGD(model.parameters(), args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    # fine-tuning
    epochs = trange(args.epochs, desc=f"Tuning Bootstrapping {boot}")
    for e in epochs:
        # training
        train(model, teacher, train_loader, optimizer, num_classes, device, args.grad_clip)
        # validation
        acc = validate(model, val_loader, num_classes, device)
        scheduler.step()
        epochs.set_postfix(acc=acc)
        if args.use_wandb:
            wandb.log({"lr": optimizer.param_groups[0]['lr']})
            wandb.log({"acc": acc})
            wandb.log({"epoch": e})
        # estimate range
        if e < args.epochs:
            correct_index = []
            with torch.no_grad():
                for images, targets, index in train_loader_idx:
                    images = images.to(device)
                    targets = targets.to(device)
                    outputs = model(images)
                    logits, outputs = torch.max(outputs, dim=1)
                    correct = torch.logical_and(outputs == targets, logits < 50)
                    correct = correct.cpu()
                    correct_index.extend(index[correct].tolist())
            if len(correct_index) > 0:
                subloader = create_subloader(args.dataset, args.data, correct_index, args.batch_size,
                                             args.num_workers)
                model.enable_track_relu()
                with torch.no_grad():
                    for images, _ in subloader:
                        images = images.to(device)
                        model(images)
                model.disable_track_relu()

    if args.use_wandb:
        wandb.finish()

    # save
    out_name = args.output + "/boot{:d}_acc{:.2f}.ckpt".format(boot, acc * 100)
    version = 0
    while os.path.isfile(out_name):
        version += 1
        out_name = args.output + "/boot{:d}_acc{:.2f}_v{:d}.ckpt".format(boot, acc * 100, version)

    torch.save({"ckpt": model.state_dict(), "coeffs": coeffs, "boot": boot, "acc": acc}, out_name)


if __name__ == '__main__':
    main()
