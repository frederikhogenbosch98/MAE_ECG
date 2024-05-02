import torch, torchvision
import torch.nn as nn
import tltorch


class ResidualBlock_TD(nn.Module):
    def __init__(self, in_channels, out_channels, R, factorization='cp', stride = 1, downsample = None):
        super(ResidualBlock_TD, self).__init__()
        self.conv1 = nn.Sequential(
                        tltorch.FactorizedConv.from_conv(nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1), rank=R, decompose_weights=True, factorization=factorization),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU())
        self.conv2 = nn.Sequential(
                        tltorch.FactorizedConv.from_conv(nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1), rank=R, decompose_weights=True, factorization=factorization),
                        nn.BatchNorm2d(out_channels))
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet_TD(nn.Module):
    def __init__(self, layers, R, factorization='cp', num_classes = 7):
        super(ResNet_TD, self).__init__()
        self.R = R
        self.factorization = factorization
        self.inplanes = 64
        self.conv1 = nn.Sequential(
                        tltorch.FactorizedConv.from_conv(nn.Conv2d(1, 64, kernel_size = 7, stride = 2, padding = 3), rank=R, decompose_weights=True, factorization=factorization),
                        nn.BatchNorm2d(64),
                        nn.ReLU())
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.layer0 = self._make_layer(ResidualBlock_TD, 64, layers[0], stride = 1)
        self.layer1 = self._make_layer(ResidualBlock_TD, 128, layers[1], stride = 2)
        self.layer2 = self._make_layer(ResidualBlock_TD, 256, layers[2], stride = 2)
        self.layer3 = self._make_layer(ResidualBlock_TD, 512, layers[3], stride = 2)
        self.avgpool = nn.AvgPool2d(4, stride=1)
        self.fc = nn.Linear(512, num_classes)
        
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(block(in_channels = self.inplanes, out_channels=planes, R=self.R, factorization=self.factorization, stride=stride, downsample=downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(in_channels=self.inplanes, out_channels=planes, R=self.R, factorization=self.factorization))

        return nn.Sequential(*layers)
    
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x