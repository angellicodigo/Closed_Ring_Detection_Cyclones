import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from dataset.dataset import CycloneDatasetSS
import argparse
from typing import Tuple
from typing import Dict
from typing import Union
from typing import List
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from config.models import UNet

PATH_SAVE_MODEL = r'C:\Users\angel\VSCode\ML-Detect-Closed-Ring-Medicanes\Semantic Segmentation\models'


def z_score_norm(data: torch.Tensor, target: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    data_copy = data.numpy()
    mean = np.nanmean(data_copy, axis=(1, 2), keepdims=True)
    std = np.nanstd(data_copy, axis=(1, 2), keepdims=True)
    mean = torch.from_numpy(mean).type_as(data)
    std = torch.from_numpy(std).type_as(data)
    data_norm = (data - mean) / std
    return data_norm, target


def load_data(batch_size: int, val_split: float, test_split: float) -> Union[Tuple[DataLoader, DataLoader], Tuple[DataLoader, DataLoader, DataLoader]]:
    dataset = CycloneDatasetSS(
        r'annotations.txt', r'dataset', transform=z_score_norm)
    if test_split == 0:
        training_samples = int(len(dataset) * (1 - val_split))

        train_set, validation_set = torch.utils.data.random_split(
            dataset, [training_samples, len(dataset) - training_samples])

        train_loader = DataLoader(
            dataset=train_set, batch_size=batch_size, shuffle=True)
        validation_loader = DataLoader(
            dataset=validation_set, batch_size=batch_size, shuffle=False)
        return train_loader, validation_loader
    else:
        test_samples = int((len(dataset)) * test_split)
        validation_samples = int(len(dataset) * val_split)
        training_samples = len(dataset) - test_samples - validation_samples

        train_set, validation_set, test_set = torch.utils.data.random_split(
            dataset, [training_samples, validation_samples, test_samples])

        train_loader = DataLoader(
            dataset=train_set, batch_size=batch_size, shuffle=True)
        validation_loader = DataLoader(
            dataset=validation_set, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(
            dataset=test_set, batch_size=batch_size, shuffle=False)

        return train_loader, validation_loader, test_loader


def init_model() -> nn.Module:
    return UNet()


def train(model: nn.Module, optimizer, train_loader: DataLoader, validation_loader: DataLoader, num_epochs: int) -> Tuple[nn.Module, List[float], List[float]]:
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    train_losses = []
    val_losses = []
    criterion = nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        with tqdm(train_loader, unit="batch") as tepoch:
            for batch_index, (datas, masks) in enumerate(tepoch, start=1):
                tepoch.set_description(f"Epoch {epoch + 1}")
                datas = datas.to(device)
                masks = masks.to(device)
                optimizer.zero_grad()
                outputs = model(datas)
                print(outputs.shape)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

                if batch_index == len(train_loader):
                    val_loss = validate(model, validation_loader, device)
                    tepoch.set_postfix(
                        train_loss=f"{train_loss / len(train_loader):.3f}", validation_loss=f"{val_loss:.3f}")
                    val_losses.append(val_loss)
                    train_losses.append(train_loss / len(train_loader))

        save_model(model, epoch)
    return model, train_losses, val_losses


def save_model(model: nn.Module, epoch: int) -> None:
    model_path = os.path.join(
        PATH_SAVE_MODEL, f'checkpoint_epoch_{epoch + 1}.pth')
    torch.save(model.state_dict(), model_path)
    temp = init_model()
    temp.load_state_dict(torch.load(model_path, weights_only=True))


def validate(model: nn.Module, validation_loader: DataLoader, device):
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for datas, masks in validation_loader:
            datas = datas.to(device)
            masks = masks.to(device)
            outputs = model(datas)
            loss = criterion(outputs, masks)
            total_loss += loss.item()

    return total_loss / len(validation_loader)


def plot(train_loss: List[float], validation_loss: List[float]) -> None:
    fig = plt.figure(figsize=(12, 6))
    epochs = np.arange(1, len(train_loss) + 1)
    plt.plot(epochs, train_loss, label='Training Loss', color='#1f77b4')
    plt.plot(epochs, validation_loss, label='Validation Loss', color='#ff7f0e')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('result.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--validation_split", type=float, default=0.20)
    parser.add_argument("--test_split", type=float, default=0)
    args = parser.parse_args()

    if args.test_split == 0:
        train_loader, validation_loader = load_data(  # type: ignore
            args.batch_size, args.validation_split, args.test_split)
    else:
        train_loader, validation_loader, test_loader = load_data(  # type: ignore
            args.batch_size, args.validation_split, args.test_split)

    model = init_model()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0005)
    model, train_loss, validation_loss = train(
        model, optimizer, train_loader, validation_loader, args.epochs)
    plot(train_loss, validation_loss)
