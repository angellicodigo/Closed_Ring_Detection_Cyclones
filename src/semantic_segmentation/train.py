import torch
from torch.utils.data import WeightedRandomSampler, DataLoader
from torch.optim import Optimizer
from torchmetrics import MetricCollection
from torchmetrics.segmentation import DiceScore
from torchmetrics.classification import MulticlassPrecision, MulticlassRecall, MulticlassAccuracy, JaccardIndex, MulticlassConfusionMatrix
import torch.nn as nn
from src.dataset.dataset import CycloneDataset
from src.config.loss import FocalLoss
import argparse
import numpy as np
from tqdm import tqdm
# import os
from models import PUNet
import optuna

NUM_CLASSES = 2
# PATH_SAVE_MODEL = r'/home/al5098/Github/ML_for_Medicane_Wind_Rings/models/semantic_segmentation'
METRICS = MetricCollection({
    "precision": MulticlassPrecision(num_classes=NUM_CLASSES),
    "recall": MulticlassRecall(num_classes=NUM_CLASSES),
    # "pixel_wise_acc": MulticlassAccuracy(num_classes=NUM_CLASSES),
    "dice_score_per_class": DiceScore(num_classes=NUM_CLASSES, average=None, input_format='index'),
    "dice_score_micro": DiceScore(num_classes=NUM_CLASSES, average='micro', input_format='index'),
    "dice_score_macro": DiceScore(num_classes=NUM_CLASSES, average='macro', input_format='index'),
    "mIoU_macro": JaccardIndex(task='multiclass', num_classes=NUM_CLASSES, average='macro'),
    "mIoU_micro": JaccardIndex(task='multiclass', num_classes=NUM_CLASSES, average='micro'),
    "mIoU_per_class": JaccardIndex(task='multiclass', num_classes=NUM_CLASSES, average=None)
    # "confusion_matrix": MulticlassConfusionMatrix(num_classes=NUM_CLASSES)
})


torch.manual_seed(52205)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def min_max(data: torch.Tensor, target: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    eplison = 1e-7
    data_copy = data.numpy()
    min = np.nanmin(data_copy, axis=(1, 2), keepdims=True)
    max = np.nanmax(data_copy, axis=(1, 2), keepdims=True)
    min = torch.from_numpy(min).type_as(data)
    max = torch.from_numpy(max).type_as(data)
    data_norm = (data - min) / (max - min + eplison)
    return data_norm, target

# def z_score_norm(data: torch.Tensor, target: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
#     data_copy = data.numpy()
#     mean = np.nanmean(data_copy, axis=(1, 2), keepdims=True)
#     std = np.nanstd(data_copy, axis=(1, 2), keepdims=True)
#     mean = torch.from_numpy(mean).type_as(data)
#     std = torch.from_numpy(std).type_as(data)
#     data_norm = (data - mean) / std

#     return data_norm, target


DATASET = CycloneDataset(r'/scratch/network/al5098/Medicanes/data/processed/annotations_SS.txt',
                         r'/scratch/network/al5098/Medicanes/data/processed/dataset', transform=min_max, augment=True, reduction_ratio=0.5)
PIXEL_WEIGHTS = DATASET.get_weights_pixels(NUM_CLASSES)

def load_data(batch_size: int, val_split: float, test_split: float):
    num_workers = 2
    
    if test_split == 0:
        training_samples = int(len(DATASET) * (1 - val_split))

        train_set, validation_set = torch.utils.data.random_split(
            DATASET, [training_samples, len(DATASET) - training_samples])
    else:
        test_samples = int((len(DATASET)) * test_split)
        validation_samples = int(len(DATASET) * val_split)
        training_samples = len(DATASET) - test_samples - validation_samples

        train_set, validation_set, test_set = torch.utils.data.random_split(DATASET, [training_samples, validation_samples, test_samples])

        test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False, persistent_workers=True, num_workers=num_workers, pin_memory=True)

    sample_weights = DATASET.get_weights_class(NUM_CLASSES, train_set.indices) # type: ignore
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(train_set), replacement=True) 
    train_loader = DataLoader(dataset=train_set, sampler=sampler, batch_size=batch_size, shuffle=False, persistent_workers=True, num_workers=num_workers, pin_memory=True)
    validation_loader = DataLoader(dataset=validation_set, batch_size=batch_size, shuffle=False, persistent_workers=True, num_workers=num_workers, pin_memory=True)

    if test_split == 0:
        return train_loader, validation_loader
    else:
        return train_loader, validation_loader, test_loader


def train(model: nn.Module, optimizer: Optimizer, train_loader: DataLoader, loss_fn):
    model.train()
    train_loss = 0.0

    # datas is shaped (B, C, H, W) and masks is shaped (B, H, W)
    for datas, masks, binary_masks in train_loader:
        datas, masks, binary_masks = datas.to(
            device), masks.to(device), binary_masks.to(device)
        optimizer.zero_grad()
        outputs = model(datas, binary_masks)
        loss = loss_fn(outputs, masks)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    return model, train_loss / len(train_loader)


def objective(trial):
    lr = trial.suggest_float('lr', 1e-5, 1, log=True)
    batch_size = trial.suggest_categorical(
        'batch_size', [8, 16, 32, 64, 128, 256])
    # optimizer = trial.suggest_categorical('optimizer', ['Adam', 'SGD'])
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-1, log=True)
    # smooth = trial.suggest_float('smooth', 0, 1)
    alpha = trial.suggest_float('alpha', 0, 1)
    gamma = trial.suggest_float('gamma', 0, 5)
    # handle_loss = trial.suggest_categorical('Handle_Loss', ['Normalize', 'Individual'])
    # if handle_loss == 'Normalize':
    # w1 = trial.suggest_float('weight_cross_entropy_loss', 0, 1)
    # w2 = trial.suggest_float('weight_dice_loss', 0, 1 - w1)
    # w3 = 1 - w1 - w2
    # else:
    #     w1 = trial.suggest_float('weight_cross_entropy_loss', 0, 1)
    #     w2 = trial.suggest_float('weight_dice_loss', 0, 1)

    if args.test_split == 0:
        train_loader, validation_loader = load_data(  # type: ignore
            batch_size, args.validation_split, args.test_split)
    else:
        train_loader, validation_loader, test_loader = load_data(  # type: ignore
            batch_size, args.validation_split, args.test_split)

    model = PUNet(channels_in=2, channels_out=NUM_CLASSES)
    model.to(device)

    # if optimizer == 'SGD':
    #     momentum = trial.suggest_float('momentum', 0.8, 0.99, step=0.01)
    #     optimizer = torch.optim.SGD(
    #         model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    # else:
    beta1 = trial.suggest_float('beta1', 0.8, 0.99, step=0.01)
    beta2 = trial.suggest_float('beta2', 0.9, 0.99, step=0.01)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(
        beta1, beta2), weight_decay=weight_decay)

    criterion = FocalLoss(alpha=alpha, gamma=gamma)

    for epoch in tqdm(range(epochs), desc=f'Trial: {trial.number}', unit='epoch'):
        model, train_loss = train(model, optimizer, train_loader, criterion)
        # save_model(model, optimizer, train_loader, args.epochs, epoch)
        val_loss = validate(model, validation_loader, criterion)
        results = METRICS.compute()
        trial.report(results['dice_score_macro'].item(), epoch)

        if trial.should_prune():
            raise optuna.TrialPruned()

    if training:
        return results['dice_score_macro'].item()
    else:
        return results['precision'].item(), results['recall'].item(), results['dice_score_per_class'], results['dice_score_macro'].item(), results['dice_score_micro'].item(), results['mIoU_per_class'], results['mIoU_macro'].item(), results['mIoU_micro'].item()  # type: ignore


# def save_model(model: nn.Module, optimizer: Optimizer, train_loader: DataLoader, epochs: int, epoch: int) -> None:
#     batch_size = train_loader.batch_size
#     lr = optimizer.param_groups[0]['lr']
#     title = f"{epochs}_{batch_size}_{lr}_{args.validation_split}"
#     path = os.path.join(PATH_SAVE_MODEL, title)
#     os.makedirs(path, exist_ok=True)
#     model_path = os.path.join(path, f'checkpoint_{epoch + 1}.pth')
#     torch.save(model.state_dict(), model_path)


def validate(model: nn.Module, validation_loader: DataLoader, loss_fn) -> float:
    total_loss = 0.0
    model.eval()
    METRICS.to(device)
    METRICS.reset()
    with torch.no_grad():
        for datas, masks, binary_masks in validation_loader:
            datas, masks, binary_masks = datas.to(
                device), masks.to(device), binary_masks.to(device)
            outputs = model(datas, binary_masks)
            preds = torch.argmax(outputs, dim=1)
            loss = loss_fn(outputs, masks)
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
    epochs = 25
    training = True
    study.optimize(objective, n_trials=args.trials)

    print("Best trial: ")
    print(f"\tnumber: {study.best_trial.number}")
    print(f"\tparams: {study.best_trial.params}")
    print(f"\tvalues: {study.best_trial.values}")

    epochs = args.epochs
    training = False
    results = objective(study.best_trial)
    print(f"Performance of best trial {study.best_trial.number}: ")
    print(f"\tPrecision: {results[0]}")
    print(f"\tRecall: {results[1]}")
    print(f"\tDice Score per class: {results[2]}")
    print(f"\tDice Score macro: {results[3]}")
    print(f"\tDice Score micro: {results[4]}")
    print(f"\tmIoU per class: {results[5]}")
    print(f"\tmIoU macro: {results[6]}")
    print(f"\tmIoU micro: {results[7]}")
