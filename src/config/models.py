import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn


class UNet(nn.Module):
    def __init__(self, in_channels=2, out_channels=3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(64, out_channels, 1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


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
