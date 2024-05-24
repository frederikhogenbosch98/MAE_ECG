import torch
import torch.nn as nn
import torch.nn.functional as F
import tltorch

class UNet_TD(nn.Module):
    def __init__(self, in_channels=1, channels=[32, 64, 128, 256], depths=[1, 1, 1], R=None, factorization='cp'):
        super(UNet_TD, self).__init__()
        # Encoder
        self.enc1 = nn.Sequential(
            tltorch.FactorizedConv.from_conv(nn.Conv2d(in_channels, channels[0], kernel_size=7, stride=1, padding=3), rank=R, decompose_weights=True, factorization=factorization, implementation='factorized'),
            nn.BatchNorm2d(channels[0]),
            nn.GELU(),
            tltorch.FactorizedConv.from_conv(nn.Conv2d(channels[0], channels[0], kernel_size=3, stride=1, padding=1), rank=R, decompose_weights=True, factorization=factorization, implementation='factorized'),
            nn.BatchNorm2d(channels[0]),
            nn.GELU(),
            tltorch.FactorizedConv.from_conv(nn.Conv2d(channels[0], channels[0], kernel_size=3, stride=1, padding=1), rank=R, decompose_weights=True, factorization=factorization, implementation='factorized'),
            nn.BatchNorm2d(channels[0]),
            nn.GELU(),
        )
        self.pool1 = nn.MaxPool2d(2, stride=2)

        self.enc2 = nn.Sequential(
            tltorch.FactorizedConv.from_conv(nn.Conv2d(channels[0], channels[1], kernel_size=3, stride=1, padding=1), rank=R, decompose_weights=True, factorization=factorization, implementation='factorized'),
            nn.BatchNorm2d(channels[1]),
            nn.GELU(),
            tltorch.FactorizedConv.from_conv(nn.Conv2d(channels[1], channels[1], kernel_size=3, stride=1, padding=1), rank=R, decompose_weights=True, factorization=factorization, implementation='factorized'),
            nn.BatchNorm2d(channels[1]),
            nn.GELU(),
            tltorch.FactorizedConv.from_conv(nn.Conv2d(channels[1], channels[1], kernel_size=3, stride=1, padding=1), rank=R, decompose_weights=True, factorization=factorization, implementation='factorized'),
            nn.BatchNorm2d(channels[1]),
            nn.GELU(),
        )
        self.pool2 = nn.MaxPool2d(2, stride=2)

        self.enc3 = nn.Sequential(
            tltorch.FactorizedConv.from_conv(nn.Conv2d(channels[1], channels[2], kernel_size=3, stride=1, padding=1), rank=R, decompose_weights=True, factorization=factorization, implementation='factorized'),
            nn.BatchNorm2d(channels[2]),
            nn.GELU(),
            tltorch.FactorizedConv.from_conv(nn.Conv2d(channels[2], channels[2], kernel_size=3, stride=1, padding=1), rank=R, decompose_weights=True, factorization=factorization, implementation='factorized'),
            nn.BatchNorm2d(channels[2]),
            nn.GELU(),
            tltorch.FactorizedConv.from_conv(nn.Conv2d(channels[2], channels[2], kernel_size=3, stride=1, padding=1), rank=R, decompose_weights=True, factorization=factorization, implementation='factorized'),
            nn.BatchNorm2d(channels[2]),
            nn.GELU(),
        )
        self.pool3 = nn.MaxPool2d(2, stride=2)


        self.enc4 = nn.Sequential(
            tltorch.FactorizedConv.from_conv(nn.Conv2d(channels[2], channels[3], kernel_size=3, stride=1, padding=1), rank=R, decompose_weights=True, factorization=factorization, implementation='factorized'),
            nn.BatchNorm2d(channels[2]),
            nn.GELU(),
            tltorch.FactorizedConv.from_conv(nn.Conv2d(channels[3], channels[3], kernel_size=3, stride=1, padding=1), rank=R, decompose_weights=True, factorization=factorization, implementation='factorized'),
            nn.BatchNorm2d(channels[2]),
            nn.GELU(),
            tltorch.FactorizedConv.from_conv(nn.Conv2d(channels[3], channels[3], kernel_size=3, stride=1, padding=1), rank=R, decompose_weights=True, factorization=factorization, implementation='factorized'),
            nn.BatchNorm2d(channels[3]),
            nn.GELU(),
        )
        self.pool4 = nn.MaxPool2d(2, stride=2)


        # Decoder
        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec4 = nn.Sequential(
            tltorch.FactorizedConv.from_conv(nn.Conv2d(channels[3] + channels[3], channels[2], kernel_size=3, stride=1, padding=1), rank=R, decompose_weights=True, factorization=factorization, implementation='factorized'),
            nn.BatchNorm2d(channels[2]),
            nn.GELU(),
            tltorch.FactorizedConv.from_conv(nn.Conv2d(channels[2], channels[2], kernel_size=3, stride=1, padding=1), rank=R, decompose_weights=True, factorization=factorization, implementation='factorized'),
            nn.BatchNorm2d(channels[2]),
            nn.GELU(),
            tltorch.FactorizedConv.from_conv(nn.Conv2d(channels[2], channels[2], kernel_size=3, stride=1, padding=1), rank=R, decompose_weights=True, factorization=factorization, implementation='factorized'),
            nn.BatchNorm2d(channels[2]),
            nn.GELU(), 
            
        )

        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec3 = nn.Sequential(
            tltorch.FactorizedConv.from_conv(nn.Conv2d(channels[2] + channels[2], channels[1], kernel_size=3, stride=1, padding=1), rank=R, decompose_weights=True, factorization=factorization, implementation='factorized'),
            nn.BatchNorm2d(channels[1]),
            nn.GELU(),
            tltorch.FactorizedConv.from_conv(nn.Conv2d(channels[1], channels[1], kernel_size=3, stride=1, padding=1), rank=R, decompose_weights=True, factorization=factorization, implementation='factorized'),
            nn.BatchNorm2d(channels[1]),
            nn.GELU(),
            tltorch.FactorizedConv.from_conv(nn.Conv2d(channels[1], channels[1], kernel_size=3, stride=1, padding=1), rank=R, decompose_weights=True, factorization=factorization, implementation='factorized'),
            nn.BatchNorm2d(channels[1]),
            nn.GELU(),
        )

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec2 = nn.Sequential(
            tltorch.FactorizedConv.from_conv(nn.Conv2d(channels[1] + channels[1], channels[0], kernel_size=3, stride=1, padding=1), rank=R, decompose_weights=True, factorization=factorization, implementation='factorized'),
            nn.BatchNorm2d(channels[0]),
            nn.GELU(),
            tltorch.FactorizedConv.from_conv(nn.Conv2d(channels[0], channels[0], kernel_size=3, stride=1, padding=1), rank=R, decompose_weights=True, factorization=factorization, implementation='factorized'),
            nn.BatchNorm2d(channels[0]),
            nn.GELU(),
            tltorch.FactorizedConv.from_conv(nn.Conv2d(channels[0], channels[0], kernel_size=3, stride=1, padding=1), rank=R, decompose_weights=True, factorization=factorization, implementation='factorized'),
            nn.BatchNorm2d(channels[0]),
            nn.GELU(),
        )

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec1 = nn.Sequential(
            tltorch.FactorizedConv.from_conv(nn.Conv2d(channels[0] + channels[0], in_channels, kernel_size=3, stride=1, padding=1), rank=R, decompose_weights=True, factorization=factorization, implementation='factorized'),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        x = self.pool1(enc1)
        enc2 = self.enc2(x)
        x = self.pool2(enc2)
        enc3 = self.enc3(x)
        x = self.pool3(enc3)
        print(x.shape)
        # Decoder
        x = self.up3(x)
        x = torch.cat([x, enc3], dim=1)
        x = self.dec3(x)

        x = self.up2(x)
        x = torch.cat([x, enc2], dim=1)
        x = self.dec2(x)

        x = self.up1(x)
        x = torch.cat([x, enc1], dim=1)
        x = self.dec1(x)

        return x


