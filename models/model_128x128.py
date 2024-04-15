import torch.nn as nn
import torch.nn.functional as F


class ConvNeXtBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvNeXtBlock, self).__init__()
        self.norm1 = nn.LayerNorm(normalized_shape=(in_channels, 1, 1))
        self.pointwise_conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.depthwise_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels)
        self.gelu = nn.GELU()
        self.pointwise_conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.norm2 = nn.LayerNorm(normalized_shape=(out_channels, 1, 1))

    def forward(self, x):
        identity = x
        x = self.norm1(x)
        x = self.pointwise_conv1(x)
        x = self.depthwise_conv(x)
        x = self.gelu(x)
        x = self.pointwise_conv2(x)
        x = self.norm2(x)
        x += identity  
        return x
    
class SelfBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SelfBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.gelu = nn.GELU()

    def forward(self, x):
        identity = x
        x = self.conv(x)
        x = self.bn(x)
        x = self.gelu(x)
        x += identity
        return x



class AutoEncoder128(nn.Module):
    def __init__(self, channels=[96, 192, 384, 768], dims=[3, 3, 9, 3]):
        super(AutoEncoder128, self).__init__()
        self.blocksize = 2
        self.encoder = nn.Sequential( 
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.GELU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.GELU(),
            *[SelfBlock(32,32) for i in range(self.blocksize)],
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GELU(),
            *[SelfBlock(64,64) for i in range(self.blocksize)],
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.GELU(),
            *[SelfBlock(128,128) for i in range(self.blocksize)],
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.GELU(),
            *[SelfBlock(256,256) for i in range(self.blocksize)],
            # nn.Conv2d(256, 512, 3, stride=2, padding=1),
            # # nn.AvgPool2d(2, strde=2),
            # nn.GELU()

            
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 256, 3, stride=2, padding=1, output_padding=1),
            nn.GELU(),
            # nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1),
            # nn.GELU(),
            *[SelfBlock(256,256) for i in range(self.blocksize)],
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.GELU(),
            *[SelfBlock(128,128) for i in range(self.blocksize)],
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.GELU(),
            *[SelfBlock(64,64) for i in range(self.blocksize)],
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.GELU(),
            *[SelfBlock(32,32) for i in range(self.blocksize)],
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class Classifier128(nn.Module):
    def __init__(self, autoencoder, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.encoder = autoencoder.encoder
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.encoder(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# class Classifier128(nn.Module):
#     def __init__(self, autoencoder, num_classes):
#         super(Classifier128, self).__init__()
#         # input_dim = 8192
#         input_dim = 1024
#         # self.classifier = nn.Sequential(
#         #     nn.Flatten(),
#         #     nn.Linear(input_dim, input_dim//2),
#         #     nn.GELU(),
#         #     nn.Linear(input_dim//2, input_dim//4),
#         #     nn.GELU(),
#         #     nn.Linear(input_dim//4, input_dim//8),
#         #     nn.GELU(),
#         #     nn.Linear(input_dim//8, num_classes)
#         # )
#         self.classifier = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(input_dim, num_classes),
#             nn.Dropout(p=0.3)
#         )

#     def forward(self, x):
#         x = self.encoder(x)
#         x = self.classifier(x)
#         # return F.softmax(x, dim=1)
#         return x