import torch
import torch.nn as nn
import torch.optim
import torchmetrics
import lightning.pytorch as pl
from models import create_model
from datasets import DATASETS


class Classifier(pl.LightningModule):
    def __init__(self, **kwargs):
        super(Classifier, self).__init__()
        self.save_hyperparameters()
        arch = kwargs.get("arch")
        dataset = kwargs.get("dataset")
        _, num_classes = DATASETS[dataset]
        self.model = create_model(arch, dataset)
        self.loss = nn.CrossEntropyLoss()
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.valid_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, images):
        output = self.model(images)
        return output

    def training_step(self, batch, batch_idx):
        images, labels = batch
        output = self(images)
        train_loss = self.loss(output, labels)
        self.train_acc(output, labels)
        self.log("train_loss", train_loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return train_loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        output = self(images)
        self.valid_acc(output, labels)
        self.log("valid_acc", self.valid_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.model.parameters(), self.hparams.lr, weight_decay=self.hparams.weight_decay,
                                    momentum=self.hparams.momentum)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.epochs)
        return [optimizer], [scheduler]
