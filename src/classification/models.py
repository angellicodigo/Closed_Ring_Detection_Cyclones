import torch
import torch.nn as nn
import torch.nn.functional as F
from partialconv2d import PartialConv2d # type: ignore


class UNetDown(nn.Module):
    def __init__(self, input_size, output_size):
        super(UNetDown, self).__init__()

        model = [nn.BatchNorm2d(input_size),
                 nn.ELU(),
                 nn.Conv2d(input_size, output_size,
                           kernel_size=3, stride=1, padding=1),
                 nn.BatchNorm2d(output_size),
                 nn.ELU(),
                 nn.MaxPool2d(2),
                 nn.Conv2d(output_size, output_size, kernel_size=3, stride=1, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, input_size, output_size):
        super(UNetUp, self).__init__()

        model = [nn.BatchNorm2d(input_size),
                 nn.ELU(),
                 nn.Conv2d(input_size, output_size,
                           kernel_size=3, stride=1, padding=1),
                 nn.BatchNorm2d(output_size),
                 nn.ELU(),
                 nn.Upsample(scale_factor=2, mode="nearest"),
                 nn.Conv2d(output_size, output_size, kernel_size=3, stride=1, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class UNet(nn.Module):
    def __init__(self, channels_in, channels_out, width_multiplier=1):
        super(UNet, self).__init__()
        # self.mha = nn.MultiheadAttention(
        #     channels_in, num_heads=1, batch_first=True)
        self.conv_in = nn.Conv2d(channels_in, out_channels=64,
                                 kernel_size=3, stride=1, padding=1)   # H X W --> 64, H X W

        self.down1 = UNetDown(64, 64)  # H   X W   --> H/2 X W/2
        self.down2 = UNetDown(64, 128)  # H/2 X W/2 --> H/4 X W/4
        self.down3 = UNetDown(128, 128)  # H/4 X W/4 --> H/8 X W/8
        self.down4 = UNetDown(128, 256)  # H/8 X W/8 --> H/16 X W/16

        # Note that the first parameter is the amount of channels
        self.up4 = UNetUp(256, 128)  # H/16 X W/16 --> H/8 X W/8
        self.up5 = UNetUp(128 * 2, 128)  # H/8 X W/8 --> H/4 X W/4
        self.up6 = UNetUp(128 * 2, 64)  # H/4 X W/4 --> H/2 X W/2
        self.up7 = UNetUp(64 * 2, 64)  # H/2 X W/2 --> H   X W

        self.conv_out = nn.Conv2d(64 * 2, channels_out,
                                  kernel_size=3, stride=1, padding=1)  # H X W --> H X W

    def forward(self, x):
        # B, C, H, W = x.shape
        # x = x.reshape(B, C, H * W).transpose(1, 2)
        # binary_mask = binary_mask.reshape(B, H * W)
        # binary_mask = ~binary_mask
        # x, _ = self.mha(x, x, x, key_padding_mask=binary_mask)

        # x = x.transpose(1, 2).reshape(B, C, H, W)

        x0 = self.conv_in(x)  # 16 x H x W

        x1 = self.down1(x0)  # 32 x H/2 x W/2
        x2 = self.down2(x1)  # 64 x H/4 x W/4
        x3 = self.down3(x2)  # 64 x H/8 x W/8
        x4 = self.down4(x3)  # 128 x H/16 x W/16

        # Bottle-neck --> 128 x H/16 x W/16

        x5 = self.up4(x4)  # 64 x H/8 x W/8

        # Torch.cat is doing skip-connections
        x5_ = torch.cat((x5, x3), 1)  # 128 x H/8 x W/8
        x6 = self.up5(x5_)  # 32 x H/4 x W/4

        x6_ = torch.cat((x6, x2), 1)  # 64 x H/4 x W/4
        x7 = self.up6(x6_)  # 16 x H/2 x W/2

        x7_ = torch.cat((x7, x1), 1)  # 64 x H/2 x W/2
        x8 = self.up7(x7_)  # 16 x H x W

        x8_ = F.elu(torch.cat((x8, x0), 1))  # 32 x H x W
        return self.conv_out(x8_)  # Co x H x W


class PUNetDown(nn.Module):
    def __init__(self, input_size, output_size, kernel_size, padding, batch_norm=True):
        super(PUNetDown, self).__init__()
        self.pconv = PartialConv2d(input_size, output_size, kernel_size=kernel_size,
                                   stride=2, padding=padding, multi_channel=False, return_mask=True)
        self.batch_norm = batch_norm
        if batch_norm:
            self.bn = nn.BatchNorm2d(output_size)
        self.relu = nn.ReLU()

    def forward(self, x, mask):
        x, mask = self.pconv(x, mask_in=mask)
        if self.batch_norm:
            x = self.bn(x)
        x = self.relu(x)
        return x, mask


class PUNetUp(nn.Module):
    def __init__(self, input_size, output_size, kernel_size, padding, batch_norm=True):
        super(PUNetUp, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.pconv = PartialConv2d(input_size, output_size, kernel_size=kernel_size,
                                   stride=1, padding=padding, multi_channel=False, return_mask=True)
        self.batch_norm = batch_norm
        if batch_norm:
            self.bn = nn.BatchNorm2d(output_size)
        self.leakyrelu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x, mask):
        x = self.upsample(x)
        mask = self.upsample(mask)
        x, mask = self.pconv(x, mask_in=mask)
        if self.batch_norm:
            x = self.bn(x)
        x = self.leakyrelu(x)
        return x, mask


class PUNet(nn.Module):
    def __init__(self, channels_in, channels_out):
        super(PUNet, self).__init__()

        self.down1 = PUNetDown(channels_in, 64, 7, 3, False)
        self.down2 = PUNetDown(64, 128, 5, 2)
        self.down3 = PUNetDown(128, 256, 5, 2)
        self.down4 = PUNetDown(256, 512, 3, 1)

        # Note that the first parameter is the amount of channels
        self.up1 = PUNetUp(512, 256, 3, 1)
        self.up2 = PUNetUp(256 * 2, 128, 3, 1)
        self.up3 = PUNetUp(128 * 2, 64, 3, 1)
        self.up4 = PUNetUp(64 * 2, channels_in, 3, 1)

        self.conv_out = nn.Conv2d(channels_in * 2, channels_out, kernel_size=1)

    def forward(self, x, mask):
        x1, mask1 = self.down1(x, mask)
        x2, mask2 = self.down2(x1, mask1)
        x3, mask3 = self.down3(x2, mask2)
        x4, mask4 = self.down4(x3, mask3)

        # Bottle-neck
        x5, mask5 = self.up1(x4, mask4)

        # Torch.cat is doing skip-connections
        x5_ = torch.cat((x5, x3), 1)
        mask5_ = ((mask5 + mask3) > 0).float()
        x6, mask6 = self.up2(x5_, mask5_)

        x6_ = torch.cat((x6, x2), 1)
        mask6_ = ((mask6 + mask2) > 0).float()
        x7, mask7 = self.up3(x6_, mask6_)

        x7_ = torch.cat((x7, x1), 1)
        mask7_ = ((mask7 + mask1) > 0).float()
        x8, _ = self.up4(x7_, mask7_)

        x8_ = torch.cat((x8, x), 1)
        return F.elu(self.conv_out(x8_))


# class PUNet_Attention(nn.Module):
#     def __init__(self, channels_in, channels_out):
#         super(PUNet_Attention, self).__init__()

#         self.down1 = PUNetDown(channels_in, 64, 7, 3, False)
#         self.down2 = PUNetDown(64, 128, 5, 2)
#         self.down3 = PUNetDown(128, 128, 5, 2)
#         self.down4 = PUNetDown(128, 256, 3, 1)
#         self.down5 = PUNetDown(256, 512, 3, 1)

#         # Note that the first parameter is the amount of channels
#         self.up4 = PUNetUp(256, 128, 3, 1)
#         self.up5 = PUNetUp(128 * 2, 128, 3, 1)
#         self.up6 = PUNetUp(128 * 2, 64, 3, 1)
#         self.up7 = PUNetUp(64 * 2, channels_in, 3, 1)

#         self.conv_out = nn.Conv2d(channels_in, channels_out, kernel_size=1)

#     def forward(self, x, mask):
#         x1, mask1 = self.down1(x, mask)
#         x2, mask2 = self.down2(x1, mask1)
#         x3, mask3 = self.down3(x2, mask2)
#         x4, mask4 = self.down4(x3, mask3)

#         # Bottle-neck
#         x5, mask5 = self.up4(x4, mask4)

#         # Torch.cat is doing skip-connections
#         x5_ = torch.cat((x5, x3), 1)
#         mask5_ = ((mask5 + mask3) > 0).float()
#         x6, mask6 = self.up5(x5_, mask5_)

#         x6_ = torch.cat((x6, x2), 1)
#         mask6_ = ((mask6 + mask2) > 0).float()
#         x7, mask7 = self.up6(x6_, mask6_)

#         x7_ = torch.cat((x7, x1), 1)
#         mask7_ = ((mask7 + mask1) > 0).float()
#         x8, _ = self.up7(x7_, mask7_)

#         x8_ = torch.cat((x8, x), 1)
#         return F.elu(self.conv_out(x8_))

class PartialClassifier(nn.Module):
    def __init__(self, channels_in):
        super(PartialClassifier, self).__init__()

        # Downsampling blocks
        self.down1 = PUNetDown(channels_in, 64, kernel_size=7, padding=3, batch_norm=False)
        self.down2 = PUNetDown(64, 128, kernel_size=5, padding=2)
        self.down3 = PUNetDown(128, 256, kernel_size=5, padding=2)
        self.down4 = PUNetDown(256, 512, kernel_size=3, padding=1)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, 1) 

    def forward(self, x, mask):
        x, mask = self.down1(x, mask)
        x, mask = self.down2(x, mask)
        x, mask = self.down3(x, mask)
        x, mask = self.down4(x, mask)

        pooled = self.global_pool(x).view(x.size(0), -1)  
        logit = self.fc(pooled)
        return logit
