import torch.nn as nn
import torch.nn.functional as F
import tensorly as tl
import tltorch
import torch
from tensorly.decomposition import parafac



class Encoder28_CPD(nn.Module):
    def __init__(self, R, factorization='cp'):
        super(Encoder28_CPD, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 7)
        self.fact_conv1 = tltorch.FactorizedConv.from_conv(self.conv1, rank=R, decompose_weights=True, factorization=factorization)
        self.fact_conv2 = tltorch.FactorizedConv.from_conv(self.conv2, rank=R, decompose_weights=True, factorization=factorization)
        self.fact_conv3 = tltorch.FactorizedConv.from_conv(self.conv3, rank=R, decompose_weights=True, factorization=factorization)

        self.gelu = nn.GELU() 
         
    def forward(self, x):
        x = self.fact_conv1(x) 
        x = self.gelu(x)
        x = self.fact_conv2(x)
        x = self.gelu(x)
        x = self.fact_conv3(x)
        # print(x.shape)
        return x


class Decoder28_CPD(nn.Module):
    def __init__(self, R, factorization='cp'):
        super(Decoder28_CPD, self).__init__()
        
        self.conv1 = nn.Conv2d(64, 32, 7, padding=3)
        self.conv2 = nn.Conv2d(32, 16, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(16, 1, 3, stride=1, padding=1)
        self.fact_conv1 = tltorch.FactorizedConv.from_conv(self.conv1, rank=R, decompose_weights=True, factorization=factorization)
        self.fact_conv2 = tltorch.FactorizedConv.from_conv(self.conv2, rank=R, decompose_weights=True, factorization=factorization)
        self.fact_conv3 = tltorch.FactorizedConv.from_conv(self.conv3, rank=R, decompose_weights=True, factorization=factorization)

        self.up1 = nn.Upsample(size=(7, 7), mode='bilinear', align_corners=False)
        self.up2 = nn.Upsample(size=(28, 28), mode='bilinear', align_corners=False)

    def forward(self, x):
        # print(x.shape)
        x = self.up1(x)
        # print(x.shape)
        x = self.fact_conv1(x)
        # print(x.shape)
        x = self.fact_conv2(x)
        # print(x.shape)
        x = self.up2(x)
        # print(x.shape)
        x = self.fact_conv3(x)
        # print(x.shape)
        return x


class AutoEncoder28_CPD(nn.Module):
    def __init__(self, encoder, decoder):
        super(AutoEncoder28_CPD, self).__init__()

        self.encoder = encoder
        self.decoder = decoder


    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x


class Classifier28_CPD(nn.Module):
    def __init__(self, autoencoder, num_classes):
        super(Classifier28_CPD, self).__init__()
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