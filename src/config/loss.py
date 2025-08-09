import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp # type: ignore

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def WeightedCrossEntropyLoss(outputs: torch.Tensor, targets: torch.Tensor, num_classes: int) -> torch.Tensor:
    epsilon = 1e-07
    weights = torch.bincount(targets.flatten(), minlength=num_classes).float()
    weights = torch.tensor([1 / (weights[0] + epsilon), 1])
    weights = weights.to(device)
    loss = nn.CrossEntropyLoss(weight=weights)(outputs, targets)
    return loss


def FocalLoss(outputs: torch.Tensor, targets: torch.Tensor, alpha=None, gamma=2) -> torch.Tensor:
    focal = smp.losses.FocalLoss(mode='multiclass', alpha=alpha, gamma=gamma)
    return focal(outputs, targets)


def DiceLoss(outputs: torch.Tensor, targets: torch.Tensor, smooth=0) -> torch.Tensor:
    dice = smp.losses.DiceLoss(
        mode='multiclass', smooth=smooth, from_logits=True)
    return dice(outputs, targets)
