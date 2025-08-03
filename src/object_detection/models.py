import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn


class FasterRCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(FasterRCNN, self).__init__()
        self.model = fasterrcnn_resnet50_fpn(
            weights=None,
            weights_backbone=None,
            num_classes=num_classes,
            image_mean=[0, 0],
            image_std=[1, 1]
        )

        backbone = self.model.backbone.body
        backbone.conv1 = nn.Conv2d(  # type: ignore
            2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )

    def forward(self, images, targets=None):
        if self.training:
            return self.model(images, targets)
        else:
            return self.model(images)