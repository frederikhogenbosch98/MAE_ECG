import torch
import torch.nn as nn
import torch.nn.functional as F
import tltorch


class AutoEncoder56_TD(nn.Module):
    def __init__(self, in_channels=1, channels=[16, 32, 64, 128], depths=[1, 1, 1], R=20, factorization='cp'):
    # def __init__(self, in_channels=1, channels=[32, 64, 128], depths=[1, 1, 1]):
        super(AutoEncoder56_TD, self).__init__()
        self.encoder = nn.Sequential(
            # LAYER 1
            tltorch.FactorizedConv.from_conv(nn.Conv2d(in_channels, channels[0], kernel_size=7, stride=1, padding=3), rank=R, decompose_weights=True, factorization=factorization),
            nn.BatchNorm2d(channels[0]),
            nn.GELU(),
            tltorch.FactorizedConv.from_conv(nn.Conv2d(channels[0], channels[0], kernel_size=3, stride=1, padding=1), rank=R, decompose_weights=True, factorization=factorization),
            nn.BatchNorm2d(channels[0]),
            nn.GELU(),
            tltorch.FactorizedConv.from_conv(nn.Conv2d(channels[0], channels[0], kernel_size=3, stride=1, padding=1), rank=R, decompose_weights=True, factorization=factorization),
            nn.BatchNorm2d(channels[0]),
            nn.GELU(),
            # LAYER 3
            nn.MaxPool2d(2, stride=2),
            # LAYER 4
            tltorch.FactorizedConv.from_conv(nn.Conv2d(channels[0], channels[1], kernel_size=3, stride=1, padding=1), rank=R, decompose_weights=True, factorization=factorization),
            nn.BatchNorm2d(channels[1]),
            nn.GELU(),
            tltorch.FactorizedConv.from_conv(nn.Conv2d(channels[1], channels[1], kernel_size=3, stride=1, padding=1), rank=R, decompose_weights=True, factorization=factorization),
            nn.BatchNorm2d(channels[1]),
            nn.GELU(),
            tltorch.FactorizedConv.from_conv(nn.Conv2d(channels[1], channels[1], kernel_size=3, stride=1, padding=1), rank=R, decompose_weights=True, factorization=factorization),
            nn.BatchNorm2d(channels[1]),
            nn.GELU(),
            # LAYER 6
            nn.MaxPool2d(2, stride=2),
            # LAYER 6
            tltorch.FactorizedConv.from_conv(nn.Conv2d(channels[1], channels[2], kernel_size=3, stride=1, padding=1), rank=R, decompose_weights=True, factorization=factorization),
            nn.BatchNorm2d(channels[2]),
            nn.GELU(),
            tltorch.FactorizedConv.from_conv(nn.Conv2d(channels[2], channels[2], kernel_size=3, stride=1, padding=1), rank=R, decompose_weights=True, factorization=factorization),
            nn.BatchNorm2d(channels[2]),
            nn.GELU(),
            tltorch.FactorizedConv.from_conv(nn.Conv2d(channels[2], channels[2], kernel_size=3, stride=1, padding=1), rank=R, decompose_weights=True, factorization=factorization),
            nn.BatchNorm2d(channels[2]),
            nn.GELU(),
            # LAYER 8
            nn.MaxPool2d(2, stride=2),

            tltorch.FactorizedConv.from_conv(nn.Conv2d(channels[2], channels[3], kernel_size=3, stride=1, padding=1), rank=R, decompose_weights=True, factorization=factorization),
            nn.BatchNorm2d(channels[3]),
            nn.GELU(),
            tltorch.FactorizedConv.from_conv(nn.Conv2d(channels[3], channels[3], kernel_size=3, stride=1, padding=1), rank=R, decompose_weights=True, factorization=factorization),
            nn.BatchNorm2d(channels[3]),
            nn.GELU(),
            tltorch.FactorizedConv.from_conv(nn.Conv2d(channels[3], channels[3], kernel_size=3, stride=1, padding=1), rank=R, decompose_weights=True, factorization=factorization),
            nn.BatchNorm2d(channels[3]),
            nn.GELU(),
            nn.MaxPool2d(2, stride=2)
        )
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            tltorch.FactorizedConv.from_conv(nn.Conv2d(channels[3], channels[2], kernel_size=3, stride=1, padding=1), rank=R, decompose_weights=True, factorization=factorization),
            nn.BatchNorm2d(channels[2]),
            nn.GELU(),
            tltorch.FactorizedConv.from_conv(nn.Conv2d(channels[2], channels[2], kernel_size=3, stride=1, padding=1), rank=R, decompose_weights=True, factorization=factorization),
            nn.BatchNorm2d(channels[2]),
            nn.GELU(),
            tltorch.FactorizedConv.from_conv(nn.Conv2d(channels[2], channels[2], kernel_size=3, stride=1, padding=1), rank=R, decompose_weights=True, factorization=factorization),
            nn.BatchNorm2d(channels[2]),
            nn.GELU(), 
            # Corresponds to LAYER 4 in Encoder
            nn.Upsample(scale_factor=2, mode='bilinear'),
            tltorch.FactorizedConv.from_conv(nn.Conv2d(channels[2], channels[1], kernel_size=3, stride=1, padding=1), rank=R, decompose_weights=True, factorization=factorization),
            nn.BatchNorm2d(channels[1]),
            nn.GELU(),
            tltorch.FactorizedConv.from_conv(nn.Conv2d(channels[1], channels[1], kernel_size=3, stride=1, padding=1), rank=R, decompose_weights=True, factorization=factorization),
            nn.BatchNorm2d(channels[1]),
            nn.GELU(),
            tltorch.FactorizedConv.from_conv(nn.Conv2d(channels[1], channels[1], kernel_size=3, stride=1, padding=1), rank=R, decompose_weights=True, factorization=factorization),
            nn.BatchNorm2d(channels[1]),
            nn.GELU(),
            # Corresponds to LAYER 4 in Encoder
            nn.Upsample(scale_factor=2, mode='bilinear'),
            tltorch.FactorizedConv.from_conv(nn.Conv2d(channels[1], channels[0], kernel_size=3, stride=1, padding=1), rank=R, decompose_weights=True, factorization=factorization),
            nn.BatchNorm2d(channels[0]),
            nn.GELU(),
            tltorch.FactorizedConv.from_conv(nn.Conv2d(channels[0], channels[0], kernel_size=3, stride=1, padding=1), rank=R, decompose_weights=True, factorization=factorization),
            nn.BatchNorm2d(channels[0]),
            nn.GELU(),
            tltorch.FactorizedConv.from_conv(nn.Conv2d(channels[0], channels[0], kernel_size=3, stride=1, padding=1), rank=R, decompose_weights=True, factorization=factorization),
            nn.BatchNorm2d(channels[0]),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            # Corresponds to LAYER 1 in Encoder
            tltorch.FactorizedConv.from_conv(nn.Conv2d(channels[0], in_channels, kernel_size=3, stride=1, padding=1), rank=R, decompose_weights=True, factorization=factorization),
            nn.Sigmoid(),


        )

    def forward(self, x):
        x = self.encoder(x)
        # print(x.shape)
        x = self.decoder(x)
        return x



class Classifier56_TD(nn.Module):
    def __init__(self, autoencoder, in_features, out_features):
        super(Classifier56_TD, self).__init__()
        self.encoder = autoencoder.encoder
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
                # nn.Linear(50176+1, 256),
                nn.Linear(6272, 256), #16
                # nn.Linear(12544, 256), #32
                nn.GELU(),
                nn.BatchNorm1d(num_features=256),
                nn.Dropout(0.5),
                nn.Linear(256, out_features)
        )

    def forward(self, images, features):
        x = self.encoder(images)
        x = self.flatten(x)
        # combined_features = torch.cat((x, features), dim=1)
        # x = self.classifier(combined_features)
        x = self.classifier(x)
        return x

