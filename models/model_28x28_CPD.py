import torch
import torch.nn as nn
import torch.nn.functional as F
import tltorch
import tensorly as tl
from tensorly.decomposition import parafac


tl.set_backend('pytorch')

class ParafacConvolution2D():
    def __init__(self, conv, R) -> None:
        layer = conv.weight
        print(f'layer shape: {conv.weight.shape}')
        weights, factors = parafac(layer, rank=R)
        print(f'weights: {len(weights)}')
        print(f'factors: {len(factors)}')
        for i in range(len(factors)):
            print(f'factor {i}: {factors[i].shape}')

        
        self.s_to_r = nn.Conv2d(conv.in_channels, R, (1, 1), bias=False)
        self.s_to_r.weight.data = factors[1].unsqueeze(-1).unsqueeze(-1)
        self.depth_vert = self.depth_vert = nn.Conv2d(R, R, (factors[2].shape[0], 1), groups=R, bias=False)
        self.depth_vert.weight.data = factors[2].unsqueeze(1).unsqueeze(-1)
        self.depth_hor = self.depth_hor = nn.Conv2d(R, R, (1, factors[3].shape[0]), groups=R, bias=False)
        self.depth_hor.weight.data = factors[3].unsqueeze(1).unsqueeze(1)
        self.r_to_t = self.r_to_t = nn.Conv2d(R, conv.out_channels, (1, 1), bias=False)
        self.r_to_t.weight.data = factors[0].unsqueeze(-1).unsqueeze(-1)
    
    def forward(self, x):
        x = self.s_to_r(x)
        x = self.depth_vert(x)
        x = self.depth_hor(x)
        x = self.r_to_t(x)
        return x


class Encoder28_CPD(nn.Module):
    def __init__(self, R, factorization='cp'):
        super(Encoder28_CPD, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 7)
        self.fact_conv1 = tltorch.FactorizedConv.from_conv(self.conv1, rank=R, decompose_weights=True, factorization=factorization)
        self.fact_conv2 = tltorch.FactorizedConv.from_conv(self.conv2, rank=R, decompose_weights=True, factorization=factorization)
        self.fact_conv3 = tltorch.FactorizedConv.from_conv(self.conv3, rank=R, decompose_weights=True, factorization=factorization, implementation='factorized')

        self.test_fact = ParafacConvolution2D(self.conv3, R=5)

        self.gelu = nn.GELU() 
         
    def forward(self, x):
        print(self.conv3.weight.size())
        print(self.fact_conv3.weight.size())
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