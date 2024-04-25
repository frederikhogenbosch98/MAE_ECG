import torch.nn as nn
import torch.nn.functional as F

class AutoEncoder28(nn.Module):
    def __init__(self, in_channels=1):
        super(AutoEncoder28, self).__init__()
        self.encoder = nn.Sequential( 
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, 7),
            # nn.GELU(),
            # nn.Conv2d(64, 128, 7)
            
        )

        # self.decoder = nn.Sequential(
        #     nn.Upsample(size=(7, 7), mode='bilinear', align_corners=False),
        #     nn.Conv2d(64,32,7, padding=3),
        #     nn.GELU(),
        #     nn.Conv2d(32, 16, 3, stride=1, padding=1),
        #     nn.GELU(),
        #     nn.Upsample(size=(28, 28), mode='bilinear', align_corners=False),
        #     nn.Conv2d(16, 1, 3, stride=1, padding=1),
        #     nn.Sigmoid()
        # )

        # self.decoder = nn.Sequential(
        #     # Start: [128, 64, 1, 1]
        #     # Upsample to 7x7
        #     nn.Upsample(size=(7, 7), mode='nearest'),  # Explicit size setting to ensure matching
        #     nn.Conv2d(64, 32, kernel_size=7, padding=3),  # No padding, the kernel fits exactly
        #     nn.GELU(),

        #     # Next layer, upsample from 7x7 to 14x14
        #     nn.Upsample(scale_factor=2, mode='nearest'),  # Upscaling to 14x14
        #     nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
        #     nn.GELU(),

        #     # Finally, upsample from 14x14 to 28x28
        #     nn.Upsample(scale_factor=2, mode='nearest'),  # Upscaling to 28x28
        #     nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1),
        #     nn.Sigmoid()
        # )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7),
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


class Classifier28(nn.Module):
    def __init__(self, autoencoder, num_classes):
        super(Classifier28, self).__init__()
        self.encoder = autoencoder.encoder
        input_dim = 64
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # nn.Linear(input_dim, num_classes),
            nn.Linear(input_dim, input_dim//2),
            nn.GELU(),
            nn.Linear(input_dim//2, num_classes)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.classifier(x)
        # return F.softmax(x, dim=1)
        return x