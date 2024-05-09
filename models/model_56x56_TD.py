import torch
import torch.nn as nn
import torch.nn.functional as F
import tltorch


class AutoEncoder56_TD(nn.Module):
    def __init__(self, R, factorization='cp', in_channels=1, channels=[16, 32, 64, 128], depths=[1, 1, 1]):
    # def __init__(self, in_channels=1, channels=[64, 128, 256], depths=[1, 1, 1]):
        super(AutoEncoder56_TD, self).__init__()
        self.encoder = nn.Sequential(
            # LAYER 1
            tltorch.FactorizedConv.from_conv(nn.Conv2d(in_channels, channels[0], kernel_size=3, stride=1, padding=1), rank=R, decompose_weights=True, factorization=factorization),
            nn.BatchNorm2d(channels[0]),
            nn.GELU(),
            # LAYER 2
            tltorch.FactorizedConv.from_conv(nn.Conv2d(channels[0], channels[0], kernel_size=3, stride=1, padding=1), rank=R, decompose_weights=True, factorization=factorization),
            nn.BatchNorm2d(channels[0]),
            nn.GELU(),
            # LAYER 3
            nn.MaxPool2d(2, stride=2),
            # LAYER 4
            tltorch.FactorizedConv.from_conv(nn.Conv2d(channels[0], channels[1], kernel_size=3, stride=1, padding=1), rank=R, decompose_weights=True, factorization=factorization),
            nn.BatchNorm2d(channels[1]),
            nn.GELU(),
            # LAYER 5
            tltorch.FactorizedConv.from_conv(nn.Conv2d(channels[1], channels[1], kernel_size=3, stride=1, padding=1), rank=R, decompose_weights=True, factorization=factorization),
            nn.BatchNorm2d(channels[1]),
            nn.GELU(),
            # LAYER 6
            nn.MaxPool2d(2, stride=2),
            # LAYER 6
            tltorch.FactorizedConv.from_conv(nn.Conv2d(channels[1], channels[2], kernel_size=3, stride=1, padding=1), rank=R, decompose_weights=True, factorization=factorization),
            nn.BatchNorm2d(channels[2]),
            nn.GELU(),
            # LAYER 7
            tltorch.FactorizedConv.from_conv(nn.Conv2d(channels[2], channels[2], kernel_size=3, stride=1, padding=1), rank=R, decompose_weights=True, factorization=factorization),
            nn.BatchNorm2d(channels[2]),
            nn.GELU(),
            # LAYER 8
            nn.MaxPool2d(2, stride=2)
        )
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            tltorch.FactorizedConv.from_conv(nn.Conv2d(channels[2], channels[2], kernel_size=3, stride=1, padding=1), rank=R, decompose_weights=True, factorization=factorization),
            nn.BatchNorm2d(channels[2]),
            nn.GELU(),
            # Corresponds to LAYER 5 in Encoder
            tltorch.FactorizedConv.from_conv(nn.Conv2d(channels[2], channels[2], kernel_size=3, stride=1, padding=1), rank=R, decompose_weights=True, factorization=factorization),
            nn.BatchNorm2d(channels[2]),
            nn.GELU(),
            # Corresponds to LAYER 4 in Encoder
            nn.Upsample(scale_factor=2, mode='bilinear'),
            tltorch.FactorizedConv.from_conv(nn.Conv2d(channels[2], channels[1], kernel_size=3, stride=1, padding=1), rank=R, decompose_weights=True, factorization=factorization),
            nn.BatchNorm2d(channels[1]),
            nn.GELU(),
            # Corresponds to LAYER 5 in Encoder
            tltorch.FactorizedConv.from_conv(nn.Conv2d(channels[1], channels[1], kernel_size=3, stride=1, padding=1), rank=R, decompose_weights=True, factorization=factorization),
            nn.BatchNorm2d(channels[1]),
            nn.GELU(),
            # Corresponds to LAYER 4 in Encoder
            nn.Upsample(scale_factor=2, mode='bilinear'),
            tltorch.FactorizedConv.from_conv(nn.Conv2d(channels[1], channels[0], kernel_size=3, stride=1, padding=1), rank=R, decompose_weights=True, factorization=factorization),
            nn.BatchNorm2d(channels[0]),
            nn.GELU(),
            # Corresponds to LAYER 2 in Encoder
            tltorch.FactorizedConv.from_conv(nn.Conv2d(channels[0], channels[0], kernel_size=3, stride=1, padding=1), rank=R, decompose_weights=True, factorization=factorization),
            nn.BatchNorm2d(channels[0]),
            nn.GELU(),
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

        self.norm = nn.LayerNorm(in_features, eps=1e-6) 
        # self.classifier = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(in_features, in_features),
        #     nn.GELU(),
        #     nn.Linear(in_features, out_features)
        # )
        self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(12544, 2048),
                nn.GELU(),
                nn.BatchNorm1d(num_features=2048),
                nn.Dropout(0.5),
                nn.Linear(2048, out_features)
        )

    def forward(self, x):
        x = self.encoder(x)
        # print(x.shape)
        # x = self.norm(x.mean([-2,-1]))
        x = self.classifier(x)
        return x


    

### TEST RUN 11 AM
# class AutoEncoder56_CPD(nn.Module):
#     def __init__(self, R, factorization='cp', in_channels=1, channels=[16, 32, 64, 128], depths=[1, 1, 1]):
#         super(AutoEncoder56_CPD, self).__init__()
#         self.encoder = nn.Sequential(
#             # LAYER 1
#             tltorch.FactorizedConv.from_conv(nn.Conv2d(in_channels, channels[0], kernel_size=3, stride=1, padding=1), rank=R, decompose_weights=True, factorization=factorization),
#             nn.GELU(),
#             nn.BatchNorm2d(channels[0]),
#             # LAYER 2
#             tltorch.FactorizedConv.from_conv(nn.Conv2d(channels[0], channels[0], kernel_size=3, stride=1, padding=1), rank=R, decompose_weights=True, factorization=factorization),
#             nn.GELU(),
#             nn.BatchNorm2d(channels[0]),
#             # LAYER 3
#             nn.MaxPool2d(2, stride=2),
#             # LAYER 4
#             tltorch.FactorizedConv.from_conv(nn.Conv2d(channels[0], channels[1], kernel_size=3, stride=1, padding=1), rank=R, decompose_weights=True, factorization=factorization),
#             nn.GELU(),
#             nn.BatchNorm2d(channels[1]),
#             # LAYER 5
#             tltorch.FactorizedConv.from_conv(nn.Conv2d(channels[1], channels[1], kernel_size=3, stride=1, padding=1), rank=R, decompose_weights=True, factorization=factorization),
#             nn.GELU(),
#             nn.BatchNorm2d(channels[1]),
#             # LAYER 6
#             nn.MaxPool2d(2, stride=2),
#             # LAYER 4
#             tltorch.FactorizedConv.from_conv(nn.Conv2d(channels[1], channels[2], kernel_size=3, stride=1, padding=1), rank=R, decompose_weights=True, factorization=factorization),
#             nn.GELU(),
#             nn.BatchNorm2d(channels[2]),
#             # LAYER 5
#             tltorch.FactorizedConv.from_conv(nn.Conv2d(channels[2], channels[2], kernel_size=3, stride=1, padding=1), rank=R, decompose_weights=True, factorization=factorization),
#             nn.GELU(),
#             nn.BatchNorm2d(channels[2]),
#             # LAYER 6
#             nn.MaxPool2d(2, stride=2)
            

            
#         )
#         self.decoder = nn.Sequential(
#             # Corresponds to LAYER 6 in Encoder
#             nn.Upsample(scale_factor=2, mode='nearest'),
#             tltorch.FactorizedConv.from_conv(nn.Conv2d(channels[2], channels[2], kernel_size=3, stride=1, padding=1), rank=R, decompose_weights=True, factorization=factorization),
#             nn.GELU(),
#             nn.BatchNorm2d(channels[2]),
#             # Corresponds to LAYER 5 in Encoder
#             tltorch.FactorizedConv.from_conv(nn.Conv2d(channels[2], channels[2], kernel_size=3, stride=1, padding=1), rank=R, decompose_weights=True, factorization=factorization),
#             nn.GELU(),
#             nn.BatchNorm2d(channels[2]),
#             # Corresponds to LAYER 4 in Encoder
#             nn.Upsample(scale_factor=2, mode='nearest'),
#             tltorch.FactorizedConv.from_conv(nn.Conv2d(channels[2], channels[1], kernel_size=3, stride=1, padding=1), rank=R, decompose_weights=True, factorization=factorization),
#             nn.GELU(),
#             nn.BatchNorm2d(channels[1]),
#             # Corresponds to LAYER 5 in Encoder
#             tltorch.FactorizedConv.from_conv(nn.Conv2d(channels[1], channels[1], kernel_size=3, stride=1, padding=1), rank=R, decompose_weights=True, factorization=factorization),
#             nn.GELU(),
#             nn.BatchNorm2d(channels[1]),
#             # Corresponds to LAYER 4 in Encoder
#             nn.Upsample(scale_factor=2, mode='nearest'),
#             tltorch.FactorizedConv.from_conv(nn.Conv2d(channels[1], channels[0], kernel_size=3, stride=1, padding=1), rank=R, decompose_weights=True, factorization=factorization),
#             nn.GELU(),
#             nn.BatchNorm2d(channels[0]),
#             # Corresponds to LAYER 2 in Encoder
#             tltorch.FactorizedConv.from_conv(nn.Conv2d(channels[0], channels[0], kernel_size=3, stride=1, padding=1), rank=R, decompose_weights=True, factorization=factorization),
#             nn.GELU(),
#             nn.BatchNorm2d(channels[0]),
#             # Corresponds to LAYER 1 in Encoder
#             tltorch.FactorizedConv.from_conv(nn.Conv2d(channels[0], in_channels, kernel_size=3, stride=1, padding=1), rank=R, decompose_weights=True, factorization=factorization),
#             nn.GELU(),
#         )


#     def forward(self, x):
#         x = self.encoder(x)
#         # print(x.shape)
#         x = self.decoder(x)
#         return x