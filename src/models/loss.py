import torch.nn as nn
import segmentation_models_pytorch as smp

class WeightedCrossEntropyLoss(nn.Module):
    # Cross-entropy loss adjusted with class weights to handle class imbalance 
    def __init__(self, weights):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.weights = weights

    def forward(self, outputs, targets):
        # Ensure weights tensor resides on the same compute device (CPU/GPU) as model outputs
        self.weights = self.weights.to(outputs.device)
        return nn.CrossEntropyLoss(weight=self.weights)(outputs, targets)


class FocalLoss(nn.Module):
    # Focal loss wrapper for multiclass tasks to down-weight easy examples and focus on hard ones
    def __init__(self, alpha=None, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, outputs, targets):
        # Instantiate SMP's multiclass focal loss implementation
        focal = smp.losses.FocalLoss(
            mode='multiclass', alpha=self.alpha, gamma=self.gamma)
        return focal(outputs, targets)


class DiceLoss(nn.Module):
    # Dice loss wrapper to optimize spatial overlap between predictions and ground-truth targets
    def __init__(self, smooth=0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, outputs, targets):
        # Instantiate SMP's multiclass dice loss, converting raw model outputs via from_logits=True
        dice = smp.losses.DiceLoss(
            mode='multiclass', smooth=self.smooth, from_logits=True)
        return dice(outputs, targets)


class Criterion(nn.Module):
    # Combined composite loss function balancing Weighted CE, Dice Loss, and Focal Loss
    def __init__(self, w1, w2, w3, weights, smooth=0, alpha=None, gamma=2):
        super(Criterion, self).__init__()
        # Hyperparameters controlling the relative weight of each loss component
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        
        # Initialize the individual loss modules
        self.wce = WeightedCrossEntropyLoss(weights)
        self.dice = DiceLoss(smooth=smooth)
        self.focal = FocalLoss(alpha=alpha, gamma=gamma)

    def forward(self, outputs, targets):
        # Compute and return the weighted sum of all three loss functions
        return self.w1 * self.wce(outputs, targets) + self.w2 * self.dice(outputs, targets) + self.w3 * self.focal(outputs, targets)

if __name__ == "__main__":
    import torch

    torch.manual_seed(0)

    # Example multiclass segmentation data
    batch_size = 2
    num_classes = 3
    height = 32
    width = 32

    outputs = torch.randn(
        batch_size, num_classes, height, width
    )

    targets = torch.randint(
        0, num_classes, (batch_size, height, width)
    )

    weights = torch.tensor([1.0, 2.0, 0.5])

    # ---------------------------------------------------------
    # Test WeightedCrossEntropyLoss
    # ---------------------------------------------------------
    wce = WeightedCrossEntropyLoss(weights)
    wce_value = wce(outputs, targets)

    print(f"Weighted CE Loss: {wce_value.item():.6f}")

    assert torch.isfinite(wce_value), "Weighted CE returned NaN/Inf"
    assert wce_value.item() >= 0, "Weighted CE should be non-negative"


    # ---------------------------------------------------------
    # Test FocalLoss
    # ---------------------------------------------------------
    focal = FocalLoss(gamma=2)
    focal_value = focal(outputs, targets)

    print(f"Focal Loss:        {focal_value.item():.6f}")

    assert torch.isfinite(focal_value), "Focal loss returned NaN/Inf"
    assert focal_value.item() >= 0, "Focal loss should be non-negative"


    # ---------------------------------------------------------
    # Test DiceLoss
    # ---------------------------------------------------------
    dice = DiceLoss(smooth=1)
    dice_value = dice(outputs, targets)

    print(f"Dice Loss:         {dice_value.item():.6f}")

    assert torch.isfinite(dice_value), "Dice loss returned NaN/Inf"
    assert dice_value.item() >= 0, "Dice loss should be non-negative"


    # ---------------------------------------------------------
    # Test Criterion
    # ---------------------------------------------------------
    criterion = Criterion(
        w1=1.0,
        w2=1.0,
        w3=1.0,
        weights=weights,
        smooth=1,
        gamma=2
    )

    criterion_value = criterion(outputs, targets)

    print(f"Combined Loss:     {criterion_value.item():.6f}")

    assert torch.isfinite(criterion_value), "Criterion returned NaN/Inf"
    assert criterion_value.item() >= 0, "Criterion should be non-negative"


    # ---------------------------------------------------------
    # Verify Criterion actually combines the three losses
    # ---------------------------------------------------------
    expected_value = (
        criterion.w1 * wce_value
        + criterion.w2 * dice_value
        + criterion.w3 * focal_value
    )

    assert torch.allclose(
        criterion_value,
        expected_value,
        atol=1e-6
    ), "Criterion does not match the weighted sum of its components"


    print("\nAll loss function tests passed.")