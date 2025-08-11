import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from src.dataset.dataset import CycloneDatasetSS
from torch.utils.data import DataLoader


class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, num_classes):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.weights = self.get_weights(num_classes)

    def get_weights(self, num_classes: int) -> torch.Tensor:
        dataset = CycloneDatasetSS(r'/scratch/network/al5098/Medicanes/data/processed/annotations_SS.txt',
                                   r'/scratch/network/al5098/Medicanes/data/processed/dataset')
        training_samples = len(dataset)
        train_set, _ = torch.utils.data.random_split(
            dataset, [training_samples, len(dataset) - training_samples])
        train_loader = DataLoader(
            dataset=train_set, batch_size=1, shuffle=False)
        counts = torch.zeros(num_classes)
        for batch in train_loader:
            counts += torch.bincount(batch[1].flatten(),
                                     minlength=num_classes).float()

        epsilon = 1e-07
        weights = 1.0 / (torch.sqrt(counts) + epsilon)
        return weights

    def forward(self, outputs, targets):
        self.weights = self.weights.to(outputs.device)
        return nn.CrossEntropyLoss(weight=self.weights)(outputs, targets)


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, outputs, targets):
        focal = smp.losses.FocalLoss(
            mode='multiclass', alpha=self.alpha, gamma=self.gamma)
        return focal(outputs, targets)


class DiceLoss(nn.Module):
    def __init__(self, smooth=0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, outputs, targets):
        dice = smp.losses.DiceLoss(
            mode='multiclass', smooth=self.smooth, from_logits=True)
        return dice(outputs, targets)


class Criterion(nn.Module):
    def __init__(self, w1, w2, w3, num_classes, smooth=0, alpha=None, gamma=2):
        super(Criterion, self).__init__()
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.wce = WeightedCrossEntropyLoss(num_classes)
        self.dice = DiceLoss(smooth=smooth)
        self.focal = FocalLoss(alpha=alpha, gamma=gamma)

    def forward(self, outputs, targets):
        return self.w1 * self.wce(outputs, targets) + self.w2 * self.dice(outputs, targets) + self.w3 * self.focal(outputs, targets)
