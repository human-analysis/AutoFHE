import torch
import torch.nn as nn
import torchmetrics
import torch.nn.functional as F


def train(model, teacher, train_loader, optimizer, num_classes, device=torch.device('cpu'), grad_clip=None, alpha=0.9):
    model.train()
    teacher.eval()
    accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
    accuracy = accuracy.to(device)
    criterion = nn.CrossEntropyLoss()
    kd_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)
    for images, targets in train_loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        outputs = model(images)
        accuracy.forward(outputs, targets)
        with torch.no_grad():
            teacher_outputs = teacher(images)
        loss = (1 - alpha) * criterion(outputs, targets) + alpha * kd_loss(F.log_softmax(outputs, dim=1), F.log_softmax(teacher_outputs, dim=1))
        if not torch.isfinite(loss):
            return float('-inf')
        optimizer.zero_grad()
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
    return accuracy.compute().cpu().item()


def validate(model, val_loader, num_classes, device=torch.device('cpu')):
    model.eval()
    accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
    accuracy = accuracy.to(device)
    with torch.no_grad():
        for i, (images, targets) in enumerate(val_loader):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = model(images)
            accuracy.forward(outputs, targets)
    return accuracy.compute().cpu().item()