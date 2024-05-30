import torch
import torch.nn as nn
import torch.nn.functional as F
import tltorch

### TEST RUN 11 AM
class AutoEncoder11_UN(nn.Module):
    def __init__(self, R=20, factorization='cp', in_channels=1, channels=[32, 64, 128, 256], depths=[1, 1, 1]):
        super(AutoEncoder11_UN, self).__init__()
        print(channels)
        self.encoder = nn.Sequential(

            # LAYER 1
            nn.Conv2d(in_channels, channels[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels[0]),
            nn.GELU(),
            nn.Conv2d(channels[0], channels[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels[0]),
            nn.GELU(),
            nn.MaxPool2d(4, stride=4),

            # LAYER 2
            nn.Conv2d(channels[0], channels[1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels[1]),
            nn.GELU(),
            nn.Conv2d(channels[1], channels[1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels[1]),
            nn.GELU(),
            nn.MaxPool2d(2, stride=2),

            # LAYER 3
            nn.Conv2d(channels[1], channels[2], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels[2]),
            nn.GELU(),
            nn.Conv2d(channels[2], channels[2], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels[2]),
            nn.GELU(),
            nn.MaxPool2d(2, stride=2),

            # LAYER 4
            nn.Conv2d(channels[2], channels[3], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels[3]),
            nn.GELU(),
            # LAYER 5
            nn.Conv2d(channels[3], channels[3], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels[3]),
            nn.GELU(),
            # LAYER 6
            nn.MaxPool2d(2, stride=2)
            
        )

        self.decoder = nn.Sequential(
            # Corresponds to LAYER 6 in Encoder
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(channels[3], channels[3], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels[3]),
            nn.GELU(),
            # Corresponds to LAYER 5 in Encoder
            nn.Conv2d(channels[3], channels[2], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels[2]),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(channels[2], channels[2], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels[2]),
            nn.GELU(),
            # Corresponds to LAYER 5 in Encoder
            nn.Conv2d(channels[2], channels[1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels[1]),
            nn.GELU(),
            # Corresponds to LAYER 4 in Encoder
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(channels[1], channels[1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels[1]),
            nn.GELU(),
            # Corresponds to LAYER 5 in Encoder
            nn.Conv2d(channels[1], channels[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels[0]),
            nn.GELU(),
            # Corresponds to LAYER 4 in Encoder
            nn.Upsample(scale_factor=4, mode='bilinear'),
            nn.Conv2d(channels[0], channels[0], kernel_size=3, stride=1, padding=1),
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

class Classifier_UN(nn.Module):
    def __init__(self, autoencoder, in_features, out_features):
        super(Classifier_UN, self).__init__()
        self.encoder = autoencoder.encoder
        self.flatten = nn.Flatten(start_dim=1)
        self.classifier = nn.Sequential(
                # nn.Linear(8*8*512, 256),
                nn.Linear(4*4*256, 512),
                nn.GELU(),
                nn.BatchNorm1d(num_features=512),
                nn.Dropout(0.5),
                nn.Linear(512, out_features)
        )
        
        self.lastlin = nn.Linear(256, out_features)
        self.reallylastlin = nn.Linear(64+1, out_features)


    def forward(self, images, features):
        x = self.encoder(images)
        # x = self.avgpool(x)
        x = self.flatten(x) 
        # combined_features = torch.cat((x, features), dim=1)
        x = self.classifier(x)
        # x = self.lastlin(x)
        
        return x


# class Classifier_UN(nn.Module):
#     def __init__(self, autoencoder, in_features, out_features):
#         super(Classifier_UN, self).__init__()
#         self.encoder = autoencoder.encoder
#         self.flatten = nn.Flatten()
#         self.classifier = nn.Sequential(
#                 # nn.Linear(50176+1, 256),
#                 # nn.Linear(6272, 256), #16
#                 # nn.Linear(32768+1, 256), #32
#                 nn.Linear(16384+1, 512), #32
#                 # nn.Linear(16*16*256, 256), #32
#                 nn.GELU(),
#                 nn.BatchNorm1d(num_features=512),
#                 nn.Dropout(0.5),
#                 nn.Linear(512, out_features)
#         )

#     def forward(self, images, features):
#         x = self.encoder(images)
#         x = self.flatten(x)
#         combined_features = torch.cat((x, features), dim=1)
#         x = self.classifier(combined_features)
#         # x = self.classifier(x)
#         return x


# class Classifier_UN(nn.Module):
#     def __init__(self, autoencoder, in_features, out_features):
#         super(Classifier_UN, self).__init__()
#         self.conv1 = autoencoder.conv1
#         self.downlayer0 = autoencoder.downlayer0
#         self.downlayer1 = autoencoder.downlayer1
#         self.maxpool = autoencoder.maxpool
#         self.downlayer2 = autoencoder.downlayer2
#         self.downlayer3 = autoencoder.downlayer3
#         self.flatten = nn.Flatten()
#         self.classifier = nn.Sequential(
#                 # nn.Linear(50176+1, 256),
#                 # nn.Linear(6272, 256), #16
#                 nn.Linear(65536, 256), #32
#                 # nn.Linear(16384, 256), #32
#                 # nn.Linear(25088, 256), #32
#                 nn.GELU(),
#                 nn.BatchNorm1d(num_features=256),
#                 nn.Dropout(0.5),
#                 nn.Linear(256, out_features)
#         )

#     def forward(self, images):
#         x = self.conv1(images)
#         x = self.maxpool(x)
#         x = self.downlayer0(x)
#         x = self.downlayer1(x)
#         x = self.maxpool(x)
#         x = self.downlayer2(x)
#         x = self.downlayer3(x)
#         x = self.flatten(x)
#         x = self.classifier(x)
#         return x