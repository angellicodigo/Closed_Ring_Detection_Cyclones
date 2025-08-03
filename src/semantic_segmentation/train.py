import torch
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torchmetrics import MetricCollection
from torchmetrics.segmentation import DiceScore
from torchmetrics.classification import MulticlassPrecision, MulticlassRecall, MulticlassAccuracy, JaccardIndex, MulticlassConfusionMatrix
import torch.nn as nn
from dataset.dataset import CycloneDatasetSS
import argparse
from typing import Union
import numpy as np
from tqdm import tqdm
import os
from config.models import UNet
import optuna

NUM_CLASSES = 3
# PATH_SAVE_MODEL = r'/home/al5098/Github/ML_for_Medicane_Wind_Rings/models/semantic_segmentation'
METRICS = MetricCollection({
    "precision": MulticlassPrecision(num_classes=NUM_CLASSES),
    "recall": MulticlassRecall(num_classes=NUM_CLASSES),
    "pixel_wise_acc": MulticlassAccuracy(num_classes=NUM_CLASSES),
    "dice_score": DiceScore(num_classes=NUM_CLASSES, average='macro'),
    "mIoU": JaccardIndex(task='multiclass', num_classes=NUM_CLASSES, average='macro'),
    "confusion_matrix": MulticlassConfusionMatrix(num_classes=NUM_CLASSES)
})
TRAIN_LOSSES = []
VAL_LOSSES = []


def z_score_norm(data: torch.Tensor, target: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    wind_data = data[:2]
    land_sea_mask = data[2:]
    wind_data_copy = wind_data.numpy()
    mean = np.nanmean(wind_data_copy, axis=(1, 2), keepdims=True)
    std = np.nanstd(wind_data_copy, axis=(1, 2), keepdims=True)
    mean = torch.from_numpy(mean).type_as(wind_data)
    std = torch.from_numpy(std).type_as(wind_data)
    wind_data_norm = (wind_data - mean) / std
    data_norm = torch.cat([wind_data_norm, land_sea_mask], dim=0)
    return data_norm, target


def load_data(batch_size: int, val_split: float, test_split: float) -> Union[tuple[DataLoader, DataLoader], tuple[DataLoader, DataLoader, DataLoader]]:
    dataset = CycloneDatasetSS(
        r'/scratch/network/al5098/Medicanes/data/processed/annotations_SS.txt', r'/scratch/network/al5098/Medicanes/data/processed/dataset', transform=z_score_norm)
    if test_split == 0:
        training_samples = int(len(dataset) * (1 - val_split))

        train_set, validation_set = torch.utils.data.random_split(
            dataset, [training_samples, len(dataset) - training_samples])

        train_loader = DataLoader(
            dataset=train_set, batch_size=batch_size, shuffle=True, persistent_workers=True, num_workers=4, pin_memory=True)
        validation_loader = DataLoader(
            dataset=validation_set, batch_size=batch_size, shuffle=False, persistent_workers=True, num_workers=4, pin_memory=True)
        return train_loader, validation_loader

    else:
        test_samples = int((len(dataset)) * test_split)
        validation_samples = int(len(dataset) * val_split)
        training_samples = len(dataset) - test_samples - validation_samples

        train_set, validation_set, test_set = torch.utils.data.random_split(
            dataset, [training_samples, validation_samples, test_samples])

        train_loader = DataLoader(
            dataset=train_set, batch_size=batch_size, shuffle=True, persistent_workers=True, num_workers=4, pin_memory=True)
        validation_loader = DataLoader(
            dataset=validation_set, batch_size=batch_size, shuffle=False, persistent_workers=True, num_workers=4, pin_memory=True)
        test_loader = DataLoader(
            dataset=test_set, batch_size=batch_size, shuffle=False, persistent_workers=True, num_workers=4, pin_memory=True)

        return train_loader, validation_loader, test_loader


def init_model() -> nn.Module:
    return UNet(channels_in=3, channels_out=3)


def train(model: nn.Module, optimizer: Optimizer, train_loader: DataLoader, device: torch.device):
    model.train()
    train_loss = 0.0
    for datas, masks in train_loader:
        datas = datas.to(device)
        masks = masks.to(device)
        optimizer.zero_grad()
        outputs = model(datas)
        loss = loss_fn(outputs, masks)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    return model, train_loss / len(train_loader)


def objective(trial: optuna.Trial) -> tuple[float, float, float, float, float, float]:
    lr = trial.suggest_categorical('lr', [1e-2, 1e-3, 1e-4])
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32, 64, 128])
    momentum = trial.suggest_float('momentum', 0.8, 0.99, step=0.01)
    weight_decay = trial.suggest_categorical(
        'weight_decay', [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6])

    if args.test_split == 0:
        train_loader, validation_loader = load_data(  # type: ignore
            batch_size, args.validation_split, args.test_split)
    else:
        train_loader, validation_loader, test_loader = load_data(  # type: ignore
            batch_size, args.validation_split, args.test_split)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = init_model().to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    for epoch in tqdm(range(args.num_epochs), desc=f'Trial: {trial.number}', unit='epoch'):
        model, train_loss = train(model, optimizer, train_loader, device)
        # save_model(model, optimizer, train_loader, args.num_epochs, epoch)
        TRAIN_LOSSES.append(train_loss)
        val_loss = validate(model, validation_loader, device)
        VAL_LOSSES.append(val_loss)
        results = METRICS.compute()

    
    return val_loss, results['precision'].item(), results['recall'].item(), results['pixel_wise_acc'].item(), results['dice_score'].item(), results['mIoU'].item() # type: ignore


def loss_fn(outputs: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    function = nn.CrossEntropyLoss()
    return function(outputs, masks)


# def save_model(model: nn.Module, optimizer: Optimizer, train_loader: DataLoader, num_epochs: int, epoch: int) -> None:
#     batch_size = train_loader.batch_size
#     lr = optimizer.param_groups[0]['lr']
#     title = f"{num_epochs}_{batch_size}_{lr}_{args.validation_split}"
#     path = os.path.join(PATH_SAVE_MODEL, title)
#     os.makedirs(path, exist_ok=True)
#     model_path = os.path.join(path, f'checkpoint_{epoch + 1}.pth')
#     torch.save(model.state_dict(), model_path)


def validate(model: nn.Module, validation_loader: DataLoader, device: torch.device) -> float:
    total_loss = 0.0
    model.eval()
    METRICS.to(device)
    METRICS.reset()
    with torch.no_grad():
        for datas, masks in validation_loader:
            datas = datas.to(device)
            masks = masks.to(device)
            outputs = model(datas)
            preds = torch.argmax(outputs, dim=1)
            loss = loss_fn(outputs, masks)
            total_loss += loss.item()
            METRICS.update(preds=preds, target=masks)

    return total_loss / len(validation_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--validation_split", type=float, default=0.25)
    parser.add_argument("--test_split", type=float, default=0)
    args = parser.parse_args()

    study = optuna.create_study(directions=['minimize', 'maximize', 'maximize',
                                'maximize', 'maximize', 'maximize'], pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=args.trials)

    trial_with_highest_accuracy = max(
        study.best_trials, key=lambda t: t.values[1])
    print("Trial with highest accuracy: ")
    print(f"\tnumber: {trial_with_highest_accuracy.number}")
    print(f"\tparams: {trial_with_highest_accuracy.params}")
    print(f"\tvalues: {trial_with_highest_accuracy.values}")
