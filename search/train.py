import os
import torch
import argparse
from pl_models import Classifier
from datasets import create_dataloader
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from lightning.pytorch import seed_everything


parser = argparse.ArgumentParser(description='Train Classifier')
parser.add_argument('-a', '--arch', metavar='ARCH', type=str, default=None, required=True,
                    help='models: VGG11, VGG11_AESPA, ResNet20, ResNet32, ResNet44, ResNet20_AESPA, ResNet32_AESPA, ResNet44_AESPA (default: None)')
parser.add_argument('-d', '--dataset', metavar='DATASET', type=str, default='cifar10', help="datasets: cifar10, cifar100 (default: cifar10)")
parser.add_argument('--data', metavar='DIR', type=str, default=None, help='path to dataset (default: None)')
parser.add_argument('-o', '--output', metavar='PATH', default=None, type=str, help='path to output (default: {arch}-{dataset})')
parser.add_argument('--seed', metavar='N', type=int, default=0, help='seed for initializing training (default: 0)')
parser.add_argument('--grad-clip', metavar='F', type=float, default=None, help='gradient clip value (default: None)')
parser.add_argument('--accelerator', metavar='DEVICE', type=str, default='gpu', help='accelerator type: cpu, gpu, tpu, etc (default: gpu)')
parser.add_argument('--devices', type=int, nargs="+", default=None, help='devices', required=False)
parser.add_argument('--precision', metavar='PRE', default=32, help='precision (16-mixed, 32) (default: 32)')
parser.add_argument('--batch-size', metavar='N', default=128, type=int, help='batch size (default: 128)')
parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run (default: 200)')
parser.add_argument('-j', '--num-workers', default=16, type=int, metavar='N', help='number of workers (default: 16)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR', help='learning rate (default: 0.1)')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum (default: 0.9)')
parser.add_argument('--weight-decay', default=5e-4, type=float, help='weight decay (default: 5e-4)')
parser.add_argument('--use-wandb', action='store_true', default=False)
parser.add_argument('--wandb-proj', type=str, metavar='PROJ', default='AutoFHE', help='wandb project (default: AutoFHE)')
parser.add_argument('--resume', action='store_true', default=False)
parser.add_argument('--evaluate', action='store_true', default=False)
parser.add_argument('--ckpt', type=str, metavar='PATH', default=None, help='path to ckpt (default: None)')


class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def main():

    args = parser.parse_args()
    hparams = dotdict(vars(args))
    hparams.arch = hparams.arch.lower()
    hparams.dataset = hparams.dataset.lower()
    if hparams.seed is not None:
        seed_everything(hparams.seed)
    if hparams.output is not None:
        if not os.path.isdir(hparams.output):
            os.mkdir(hparams.output)
    else:
        hparams.output = "."
    hparams.output = hparams.output + f"/{hparams.arch}-{hparams.dataset}"

    torch.set_float32_matmul_precision('medium')
    pl_model = Classifier(**hparams)
    train_loader, val_loader = create_dataloader(args.dataset, args.data, args.batch_size, args.num_workers)

    if hparams.evaluate:
        assert os.path.isfile(hparams.ckpt_path), f"=> Not found {hparams.ckpt}"
        trainer = pl.Trainer(accelerator=hparams.accelerator, devices=hparams.devices[0])
        trainer.validate(model=pl_model, ckpt_path=hparams.ckpt, dataloaders=val_loader, verbose=True)
    else:
        ckpt_callback = ModelCheckpoint(dirpath=os.path.join(hparams.output),
                                        filename='{epoch}-{valid_acc:.2f}',
                                        monitor='valid_acc',
                                        save_last=True,
                                        save_top_k=1,
                                        mode='max')
        callback_list = [ckpt_callback]
        csv_logger = CSVLogger(save_dir=hparams.output, name='result')
        logger_list = [csv_logger]
        if hparams.use_wandb and hparams.wandb_proj is not None:
            wandb_logger = WandbLogger(name=os.path.basename(hparams.output), project=hparams.wandb_proj)
            logger_list.append(wandb_logger)

        trainer = pl.Trainer(accelerator=hparams.accelerator,
                             strategy="auto",
                             devices=hparams.devices,
                             precision=hparams.precision,
                             logger=logger_list,
                             callbacks=callback_list,
                             max_epochs=hparams.epochs,
                             accumulate_grad_batches=1,
                             num_sanity_val_steps=-1,
                             deterministic=True,
                             gradient_clip_val=hparams.grad_clip,)

        if hparams.resume and os.path.isfile(hparams.ckpt):
            trainer.fit(pl_model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=hparams.ckpt)
        else:
            trainer.fit(pl_model, train_dataloaders=train_loader, val_dataloaders=val_loader)

        trainer.validate(ckpt_path='last', dataloaders=val_loader)


if __name__ == '__main__':
    main()
