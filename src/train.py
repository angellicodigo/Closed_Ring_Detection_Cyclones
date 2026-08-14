import os
import argparse
from functools import partial
import numpy as np
import dotenv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torchmetrics import MetricCollection
from torchmetrics.segmentation import DiceScore
import optuna
from data.dataset import CycloneDataset
from models.models import PUNet
from models.loss import DiceLoss
from models.save_model import (
    create_dir,
    save_config,
    log_metrics,
    save_checkpoint,
)

project_dir = os.path.join(os.path.dirname(__file__), os.pardir)
dotenv_path = os.path.join(project_dir, ".env")
dotenv.load_dotenv(dotenv_path)

NUM_CLASSES = 2
METRICS = MetricCollection(
    {
        "dice_score": DiceScore(
            num_classes=NUM_CLASSES, average="macro", input_format="index"
        )
    }
)

torch.manual_seed(52205)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# def min_max(
#     data: torch.Tensor, target: torch.Tensor
# ) -> tuple[torch.Tensor, torch.Tensor]:
#     epsilon = 1e-7
#     data_copy = data.numpy()

#     min_val = np.nanmin(data_copy, axis=(1, 2), keepdims=True)
#     max_val = np.nanmax(data_copy, axis=(1, 2), keepdims=True)

#     min_val = torch.from_numpy(min_val).type_as(data)
#     max_val = torch.from_numpy(max_val).type_as(data)

#     data_norm = (data - min_val) / (max_val - min_val + epsilon)
#     return data_norm, target

# def z_score_norm(data: torch.Tensor, target: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
#     data_copy = data.numpy()
#     mean = np.nanmean(data_copy, axis=(1, 2), keepdims=True)
#     std = np.nanstd(data_copy, axis=(1, 2), keepdims=True)
#     mean = torch.from_numpy(mean).type_as(data)
#     std = torch.from_numpy(std).type_as(data)
#     data_norm = (data - mean) / std

#     return data_norm, target

def load_data(batch_size: int, train_val_split: float):
    """Dynamically instantiates the dataset and creates DataLoaders."""
    dataset = CycloneDataset(
        os.getenv("ANNOTATIONS_FILE_PATH"),
        os.getenv("TRACKS_PATH"),
        augment=True
    )

    num_workers = 2

    training_samples = int(len(dataset) * (1 - train_val_split))
    train_set, validation_set = torch.utils.data.random_split(
        dataset, [training_samples, len(dataset) - training_samples]
    )

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        shuffle=True,
        persistent_workers=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    validation_loader = DataLoader(
        dataset=validation_set,
        batch_size=batch_size,
        shuffle=False,
        persistent_workers=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, validation_loader

def train(model: nn.Module, optimizer: Optimizer, train_loader: DataLoader, loss_fn) -> tuple[nn.Module, float]:
    model.train()
    train_loss = 0.0

    for datas, masks, binary_masks in train_loader:
        datas = datas.to(device)
        masks = masks.to(device)
        binary_masks = binary_masks.to(device)

        optimizer.zero_grad()
        outputs = model(datas, binary_masks)
        loss = loss_fn(outputs, masks)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    return model, train_loss / len(train_loader)


def validate(model: nn.Module, validation_loader: DataLoader, loss_fn) -> tuple[float, float]:
    total_loss = 0.0
    model.eval()

    METRICS.to(device)
    METRICS.reset()

    with torch.no_grad():
        for datas, masks, binary_masks in validation_loader:
            datas = datas.to(device)
            masks = masks.to(device)
            binary_masks = binary_masks.to(device)

            outputs = model(datas, binary_masks)
            preds = torch.argmax(outputs, dim=1)
            loss = loss_fn(outputs, masks)

            total_loss += loss.item()

            valid = binary_masks.squeeze(1).bool()
            preds_valid = torch.where(valid, preds, torch.zeros_like(preds))
            masks_valid = torch.where(valid, masks, torch.zeros_like(masks))

            METRICS.update(preds=preds_valid, target=masks_valid)

            # -------------------------------------------------------------------------
            # Alternative approach:
            # Since background & invalid regions in `masks` are already labeled as class 0,
            # you could pass the raw tensors directly without explicit masking:
            #
            # METRICS.update(preds=preds, target=masks)
            # -------------------------------------------------------------------------

    results = METRICS.compute()
    val_loss = total_loss / len(validation_loader)
    dice_score = results["dice_score"].item()

    return val_loss, dice_score


def objective(trial: optuna.Trial, args: argparse.Namespace) -> float:
    """Optuna objective function for hyperparameter search."""
    lr = trial.suggest_float("lr", 1e-5, 1, log=True)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128, 256])
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-1, log=True)
    beta1 = trial.suggest_float("beta1", 0.8, 0.99, step=0.01)
    beta2 = trial.suggest_float("beta2", 0.9, 0.99, step=0.01)

    save_base_path = os.getenv("SAVE_MODELS_PATH")
    run_dir = create_dir(save_base_path)

    config_data = {
        "trial_number": trial.number,
        "lr": lr,
        "batch_size": batch_size,
        "weight_decay": weight_decay,
        "betas": (beta1, beta2),
        "epochs": args.epochs,
        "num_classes": NUM_CLASSES,
        "mode": "optuna_trial",
    }
    save_config(run_dir, config_data)

    data_loaders = load_data(batch_size, args.train_val_split)
    train_loader, validation_loader = data_loaders[0], data_loaders[1]

    model = PUNet(channels_in=2, channels_out=NUM_CLASSES).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, betas=(beta1, beta2), weight_decay=weight_decay
    )
    criterion = DiceLoss(smooth=0)

    for epoch in range(args.epochs):
        model, train_loss = train(
            model, optimizer, train_loader, criterion
        )
        val_loss, dice_score = validate(model, validation_loader, criterion)

        log_metrics(
            run_dir,
            epoch,
            {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "dice_score": dice_score,
            },
        )
        save_checkpoint(model, optimizer, run_dir, epoch)

        trial.report(dice_score, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return dice_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--train_val_split", type=float, default=0.25)
    args = parser.parse_args()
    import time
    start_time = time.perf_counter()
    study = optuna.create_study(
        directions=["maximize"], pruner=optuna.pruners.MedianPruner()
    )
    study.optimize(partial(objective, args=args), n_trials=args.trials)
    elapsed = time.perf_counter() - start_time
    print(f"The time it took to train the model is {elapsed:.4f} seconds.")
    print("\nSearch complete!")
    print(f"Best Trial Score: {study.best_value}")
    print("Best Parameters:", study.best_params)