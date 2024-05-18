import torch
import torch.nn as nn
import torch.nn.functional as F
import tltorch

### TEST RUN 11 AM
class AutoEncoder11_UN(nn.Module):
    def __init__(self, R=20, factorization='cp', in_channels=1, channels=[64, 128, 256, 512], depths=[1, 1, 1]):
        super(AutoEncoder11_UN, self).__init__()
        print(channels)
        self.encoder = nn.Sequential(
            # LAYER 1
            nn.Conv2d(channels[0], channels[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels[0]),
            nn.GELU(),
            # LAYER 2
            nn.Conv2d(channels[0], channels[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels[0]),
            nn.GELU(),
            # LAYER 3
            nn.MaxPool2d(2, stride=2),
            # LAYER 4
            nn.Conv2d(channels[1], channels[1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels[1]),
            nn.GELU(),
            # LAYER 5
            nn.Conv2d(channels[1], channels[1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels[1]),
            nn.GELU(),
            # LAYER 6
            nn.MaxPool2d(2, stride=2),
            # LAYER 4
            nn.Conv2d(channels[1], channels[2], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels[2]),
            nn.GELU(),
            # LAYER 5
            nn.Conv2d(channels[2], channels[2], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels[2]),
            nn.GELU(),
            # LAYER 6
            nn.MaxPool2d(2, stride=2)
            # nn.Conv2d(channels[2], channels[3], kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(channels[3]),
            # nn.GELU(),
            # # LAYER 5
            # nn.Conv2d(channels[3], channels[3], kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(channels[3]),
            # nn.GELU(),
            # # LAYER 6
            # nn.MaxPool2d(2, stride=2)
            
        )

        self.decoder = nn.Sequential(
            # Corresponds to LAYER 6 in Encoder
            # nn.Upsample(scale_factor=2, mode='nearest'),
            # nn.Conv2d(channels[3], channels[3], kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(channels[3]),
            # nn.GELU(),
            # # Corresponds to LAYER 5 in Encoder
            # nn.Conv2d(channels[3], channels[2], kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(channels[2]),
            # nn.GELU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(channels[2], channels[2], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels[2]),
            nn.GELU(),
            # Corresponds to LAYER 5 in Encoder
            nn.Conv2d(channels[2], channels[2], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels[2]),
            nn.GELU(),
            # Corresponds to LAYER 4 in Encoder
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(channels[2], channels[1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels[1]),
            nn.GELU(),
            # Corresponds to LAYER 5 in Encoder
            nn.Conv2d(channels[1], channels[1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels[1]),
            nn.GELU(),
            # Corresponds to LAYER 4 in Encoder
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(channels[1], channels[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels[0]),
            nn.GELU(),
            # Corresponds to LAYER 2 in Encoder
            nn.Conv2d(channels[0], channels[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels[0]),
            nn.GELU(),
            # Corresponds to LAYER 1 in Encoder
            nn.Conv2d(channels[0], in_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )


    def forward(self, x):
        x = self.encoder(x)
        # print(x.shape)
        x = self.decoder(x)
        return x