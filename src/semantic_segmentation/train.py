import torch
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torchmetrics import MetricCollection
from torchmetrics.segmentation import DiceScore
from torchmetrics.classification import MulticlassPrecision, MulticlassRecall, MulticlassAccuracy, JaccardIndex, MulticlassConfusionMatrix
import torch.nn as nn
from src.dataset.dataset import CycloneDatasetSS
from src.config.loss import WeightedCrossEntropyLoss, FocalLoss, DiceLoss
import argparse
from typing import Union
from typing import Tuple
import numpy as np
from tqdm import tqdm
# import os
from models import UNet
import optuna

NUM_CLASSES = 2
# PATH_SAVE_MODEL = r'/home/al5098/Github/ML_for_Medicane_Wind_Rings/models/semantic_segmentation'
METRICS = MetricCollection({
    "precision": MulticlassPrecision(num_classes=NUM_CLASSES),
    "recall": MulticlassRecall(num_classes=NUM_CLASSES),
    # "pixel_wise_acc": MulticlassAccuracy(num_classes=NUM_CLASSES),
    "dice_score": DiceScore(num_classes=NUM_CLASSES, average='macro'),
    "mIoU": JaccardIndex(task='multiclass', num_classes=NUM_CLASSES, average='macro')
    # "confusion_matrix": MulticlassConfusionMatrix(num_classes=NUM_CLASSES)
})

TRAIN_LOSSES = []
VAL_LOSSES = []

torch.manual_seed(52205)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def z_score_norm(data: torch.Tensor, target: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    data_copy = data.numpy()
    mean = np.nanmean(data_copy, axis=(1, 2), keepdims=True)
    std = np.nanstd(data_copy, axis=(1, 2), keepdims=True)
    mean = torch.from_numpy(mean).type_as(data)
    std = torch.from_numpy(std).type_as(data)
    data_norm = (data - mean) / std

    return data_norm, target


def min_max(data: torch.Tensor, target: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    eplison = 1e-6
    data_copy = data.numpy()
    min = np.nanmin(data_copy, axis=(1, 2), keepdims=True)
    max = np.nanmax(data_copy, axis=(1, 2), keepdims=True)
    min = torch.from_numpy(min).type_as(data)
    max = torch.from_numpy(max).type_as(data)
    data_norm = (data - min) / (max - min + eplison)
    return data_norm, target


def load_data(batch_size: int, val_split: float, test_split: float, transform: str) -> Union[tuple[DataLoader, DataLoader], tuple[DataLoader, DataLoader, DataLoader]]:
    if transform == 'min_max':
        dataset = CycloneDatasetSS(
            r'/scratch/network/al5098/Medicanes/data/processed/annotations_SS.txt', r'/scratch/network/al5098/Medicanes/data/processed/dataset', transform=min_max)
    else:
        dataset = CycloneDatasetSS(
            r'/scratch/network/al5098/Medicanes/data/processed/annotations_SS.txt', r'/scratch/network/al5098/Medicanes/data/processed/dataset', transform=z_score_norm)

    num_workers = 2
    if test_split == 0:
        training_samples = int(len(dataset) * (1 - val_split))

        train_set, validation_set = torch.utils.data.random_split(
            dataset, [training_samples, len(dataset) - training_samples])
    else:
        test_samples = int((len(dataset)) * test_split)
        validation_samples = int(len(dataset) * val_split)
        training_samples = len(dataset) - test_samples - validation_samples

        train_set, validation_set, test_set = torch.utils.data.random_split(
            dataset, [training_samples, validation_samples, test_samples])

        test_loader = DataLoader(
            dataset=test_set, batch_size=batch_size, shuffle=False, persistent_workers=True, num_workers=num_workers, pin_memory=True)

    train_loader = DataLoader(
        dataset=train_set, batch_size=batch_size, shuffle=True, persistent_workers=True, num_workers=num_workers, pin_memory=True)
    validation_loader = DataLoader(
        dataset=validation_set, batch_size=batch_size, shuffle=False, persistent_workers=True, num_workers=num_workers, pin_memory=True)

    if test_split == 0:
        return train_loader, validation_loader
    else:
        return train_loader, validation_loader, test_loader


def criterion(outputs: torch.Tensor, masks: torch.Tensor, loss_weights: Tuple[float, float, float]) -> torch.Tensor:
    w1, w2, w3 = loss_weights
    return w1 * WeightedCrossEntropyLoss(outputs, masks, NUM_CLASSES) + w2 * DiceLoss(outputs, masks) + w3 * FocalLoss(outputs, masks)


def train(model: nn.Module, optimizer: Optimizer, train_loader: DataLoader, loss_weights: Tuple[float, float, float]):
    model.train()
    train_loss = 0.0

    # datas is shaped (B, C, H, W) and masks is shaped (B, H, W)
    for datas, masks in train_loader:
        datas, masks = datas.to(device), masks.to(device)
        # binary_masks = (datas >= 0)
        # binary_masks = binary_masks[:, 0:1, :, :]
        optimizer.zero_grad()
        outputs = model(datas)
        loss = criterion(outputs, masks, loss_weights)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    return model, train_loss / len(train_loader)


def objective(trial) -> tuple[float, float, float, float, float, float]:
    lr = trial.suggest_float('lr', 1e-5, 1, log=True)
    batch_size = trial.suggest_categorical(
        'batch_size', [1, 2, 4, 8, 16, 32, 64, 128, 256])
    transform = trial.suggest_categorical(
        'transform', ['min_max', 'z_score_norm'])
    optimizer = trial.suggest_categorical('optimizer', ['Adam', 'SGD'])
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-1, log=True)
    w1 = trial.suggest_float('weight_cross_entropy_loss', 0, 1)
    w2 = trial.suggest_float('weight_dice_loss', 0, 1 - w1)
    w3 = 1 - w1 - w2
    loss_weights = (w1, w2, w3)

    if args.test_split == 0:
        train_loader, validation_loader = load_data(  # type: ignore
            batch_size, args.validation_split, args.test_split, transform)
    else:
        train_loader, validation_loader, test_loader = load_data(  # type: ignore
            batch_size, args.validation_split, args.test_split, transform)

    model = UNet(channels_in=2, channels_out=NUM_CLASSES)
    model.to(device)

    if optimizer == 'SGD':
        momentum = trial.suggest_float('momentum', 0.8, 0.99, step=0.01)
        optimizer = torch.optim.SGD(
            model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        beta1 = trial.suggest_float('beta1', 0.8, 0.99, step=0.01)
        beta2 = trial.suggest_float('beta2', 0.9, 0.99, step=0.01)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(
            beta1, beta2), weight_decay=weight_decay)

    # type: ignore
    for epoch in tqdm(range(epochs), desc=f'Trial: {trial.number}', unit='epoch'):
        model, train_loss = train(model, optimizer, train_loader, loss_weights)
        # save_model(model, optimizer, train_loader, args.epochs, epoch)
        # TRAIN_LOSSES.append(train_loss)
        val_loss = validate(model, validation_loader, loss_weights)
        # VAL_LOSSES.append(val_loss)
        results = METRICS.compute()

    if training:
        return results['mIoU'].item()
    else:
        return results['precision'].item(), results['recall'].item(), results['dice_score'].item(), results['mIoU'].item() # type: ignore


# def save_model(model: nn.Module, optimizer: Optimizer, train_loader: DataLoader, epochs: int, epoch: int) -> None:
#     batch_size = train_loader.batch_size
#     lr = optimizer.param_groups[0]['lr']
#     title = f"{epochs}_{batch_size}_{lr}_{args.validation_split}"
#     path = os.path.join(PATH_SAVE_MODEL, title)
#     os.makedirs(path, exist_ok=True)
#     model_path = os.path.join(path, f'checkpoint_{epoch + 1}.pth')
#     torch.save(model.state_dict(), model_path)


def validate(model: nn.Module, validation_loader: DataLoader, loss_weights: Tuple[float, float, float]) -> float:
    total_loss = 0.0
    model.eval()
    METRICS.to(device)
    METRICS.reset()
    with torch.no_grad():
        for datas, masks in validation_loader:
            datas, masks = datas.to(device), masks.to(device)
            # binary_masks = (datas >= 0)
            # binary_masks = binary_masks[:, 0:1, :, :]
            outputs = model(datas)
            preds = torch.argmax(outputs, dim=1)
            loss = criterion(outputs, masks, loss_weights)
            total_loss += loss.item()
            METRICS.update(preds=preds, target=masks)

    return total_loss / len(validation_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--validation_split", type=float, default=0.25)
    parser.add_argument("--test_split", type=float, default=0)
    args = parser.parse_args()

    # study = optuna.create_study(directions=['minimize', 'maximize', 'maximize', 'maximize', 'maximize', 'maximize'], pruner=optuna.pruners.MedianPruner())
    study = optuna.create_study(
        directions=['maximize'], pruner=optuna.pruners.MedianPruner())

    global epochs
    global training
    epochs = 1
    training = True
    study.optimize(objective, n_trials=args.trials)

    print("Best trial: ")
    print(f"\tnumber: {study.best_trial.number}")
    print(f"\tparams: {study.best_trial.params}")
    print(f"\tvalues: {study.best_trial.values}")

    epochs = args.epochs
    training = False
    results = objective(study.best_trial)
    print("Performance of best trial: ")
    print(f"\tPrecision: {results[0]}")
    print(f"\tResults: {results[1]}")
    print(f"\tDice Score: {results[2]}")
    print(f"\tmIoU: {results[3]}")
