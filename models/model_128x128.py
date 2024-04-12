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



class AutoEncoder128(nn.Module):
    def __init__(self):
        super(AutoEncoder128, self).__init__()
        self.encoder = nn.Sequential( 
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, 8),
            # nn.GELU(),
            # nn.Conv2d(64, 128, 8)
            
        )
        self.decoder = nn.Sequential(
            # nn.ConvTranspose2d(128, 64, 7),
            # nn.GELU(),
            nn.ConvTranspose2d(64, 32, 8),
            nn.GELU(),
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
    def __init__(self, autoencoder, num_classes):
        super(Classifier128, self).__init__()
        self.encoder = autoencoder.encoder
        input_dim = 64
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, input_dim//2),
            nn.GELU(),
            nn.Linear(input_dim//2, num_classes)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.classifier(x)
        return F.softmax(x, dim=1)
        # return x