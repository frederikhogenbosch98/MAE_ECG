import torch
import torch.nn as nn
import torch.nn.functional as F
import tltorch

### TEST RUN 11 AM
class AutoEncoder11(nn.Module):
    def __init__(self, R=20, factorization='cp', in_channels=1, channels=[64, 128, 256, 512], depths=[1, 1, 1]):
        super(AutoEncoder11, self).__init__()
        print(channels)
        self.encoder = nn.Sequential(
            # LAYER 1
            tltorch.FactorizedConv.from_conv(nn.Conv2d(in_channels, channels[0], kernel_size=3, stride=1, padding=1), rank=R, decompose_weights=False, factorization=factorization, implementation='factorized'),
            nn.BatchNorm2d(channels[0]),
            nn.GELU(),
            # LAYER 2
            tltorch.FactorizedConv.from_conv(nn.Conv2d(channels[0], channels[0], kernel_size=3, stride=1, padding=1), rank=R, decompose_weights=False, factorization=factorization, implementation='factorized'),
            nn.BatchNorm2d(channels[0]),
            nn.GELU(),
            # LAYER 3
            nn.MaxPool2d(2, stride=2),


            # LAYER 4
            tltorch.FactorizedConv.from_conv(nn.Conv2d(channels[0], channels[1], kernel_size=3, stride=1, padding=1), rank=R, decompose_weights=False, factorization=factorization, implementation='factorized'),
            nn.BatchNorm2d(channels[1]),
            nn.GELU(),
            # LAYER 5
            tltorch.FactorizedConv.from_conv(nn.Conv2d(channels[1], channels[1], kernel_size=3, stride=1, padding=1), rank=R, decompose_weights=False, factorization=factorization, implementation='factorized'),
            nn.BatchNorm2d(channels[1]),
            nn.GELU(),
            # LAYER 6
            nn.MaxPool2d(2, stride=2),


            # LAYER 4
            tltorch.FactorizedConv.from_conv(nn.Conv2d(channels[1], channels[2], kernel_size=3, stride=1, padding=1), rank=R, decompose_weights=False, factorization=factorization, implementation='factorized'),
            nn.BatchNorm2d(channels[2]),
            nn.GELU(),
            # LAYER 5
            tltorch.FactorizedConv.from_conv(nn.Conv2d(channels[2], channels[2], kernel_size=3, stride=1, padding=1), rank=R, decompose_weights=False, factorization=factorization, implementation='factorized'),
            nn.BatchNorm2d(channels[2]),
            nn.GELU(),
            # LAYER 6
            nn.MaxPool2d(2, stride=2),


            tltorch.FactorizedConv.from_conv(nn.Conv2d(channels[2], channels[3], kernel_size=3, stride=1, padding=1), rank=R, decompose_weights=False, factorization=factorization, implementation='factorized'),
            nn.BatchNorm2d(channels[3]),
            nn.GELU(),
            # LAYER 5
            tltorch.FactorizedConv.from_conv(nn.Conv2d(channels[3], channels[3], kernel_size=3, stride=1, padding=1), rank=R, decompose_weights=False, factorization=factorization, implementation='factorized'),
            nn.BatchNorm2d(channels[3]),
            nn.GELU(),
            # LAYER 6
            nn.MaxPool2d(2, stride=2)
            
        )

        self.decoder = nn.Sequential(
            # Corresponds to LAYER 6 in Encoder
            nn.Upsample(scale_factor=2, mode='bilinear'),
            tltorch.FactorizedConv.from_conv(nn.Conv2d(channels[3], channels[3], kernel_size=3, stride=1, padding=1), rank=R, decompose_weights=False, factorization=factorization, implementation='factorized'),
            nn.BatchNorm2d(channels[3]),
            nn.GELU(),
            # Corresponds to LAYER 5 in Encoder
            tltorch.FactorizedConv.from_conv(nn.Conv2d(channels[3], channels[2], kernel_size=3, stride=1, padding=1), rank=R, decompose_weights=False, factorization=factorization, implementation='factorized'),
            nn.BatchNorm2d(channels[2]),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),


            tltorch.FactorizedConv.from_conv(nn.Conv2d(channels[2], channels[2], kernel_size=3, stride=1, padding=1), rank=R, decompose_weights=False, factorization=factorization, implementation='factorized'),
            nn.BatchNorm2d(channels[2]),
            nn.GELU(),
            # Corresponds to LAYER 5 in Encoder
            tltorch.FactorizedConv.from_conv(nn.Conv2d(channels[2], channels[1], kernel_size=3, stride=1, padding=1), rank=R, decompose_weights=False, factorization=factorization, implementation='factorized'),
            nn.BatchNorm2d(channels[1]),
            nn.GELU(),
            # Corresponds to LAYER 4 in Encoder
            nn.Upsample(scale_factor=2, mode='bilinear'),


            tltorch.FactorizedConv.from_conv(nn.Conv2d(channels[1], channels[1], kernel_size=3, stride=1, padding=1), rank=R, decompose_weights=False, factorization=factorization, implementation='factorized'),
            nn.BatchNorm2d(channels[1]),
            nn.GELU(),
            # Corresponds to LAYER 5 in Encoder
            tltorch.FactorizedConv.from_conv(nn.Conv2d(channels[1], channels[0], kernel_size=3, stride=1, padding=1), rank=R, decompose_weights=False, factorization=factorization, implementation='factorized'),
            nn.BatchNorm2d(channels[0]),
            nn.GELU(),
            # Corresponds to LAYER 4 in Encoder
            nn.Upsample(scale_factor=2, mode='bilinear'),


            tltorch.FactorizedConv.from_conv(nn.Conv2d(channels[0], channels[0], kernel_size=3, stride=1, padding=1), rank=R, decompose_weights=False, factorization=factorization, implementation='factorized'),
            nn.BatchNorm2d(channels[0]),
            nn.GELU(),
            # Corresponds to LAYER 2 in Encoder
            tltorch.FactorizedConv.from_conv(nn.Conv2d(channels[0], channels[0], kernel_size=3, stride=1, padding=1), rank=R, decompose_weights=False, factorization=factorization, implementation='factorized'),
            nn.BatchNorm2d(channels[0]),
            nn.GELU(),

            
            # Corresponds to LAYER 1 in Encoder
            tltorch.FactorizedConv.from_conv(nn.Conv2d(channels[0], in_channels, kernel_size=3, stride=1, padding=1), rank=R, decompose_weights=False, factorization=factorization, implementation='factorized'),
            nn.Sigmoid(),
        )


    def forward(self, x):
        x = self.encoder(x)
        # print(x.shape)
        x = self.decoder(x)
        return x