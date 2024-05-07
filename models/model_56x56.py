import torch
import torch.nn as nn
import torch.nn.functional as F


class ResEncoderBlock(nn.Module):

    def __init__(self, in_channels, out_channels, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm1 = nn.BatchNorm2d(in_channels)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()


    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act(x)
        return x


class ResDecoderBlock(nn.Module):

    def __init__(self, in_channels, out_channels, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()


    def forward(self, x):
        x = self.conv1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.norm(x)
        x = self.act(x)
        return x



class AutoEncoder56(nn.Module):
    # def __init__(self, in_channels=1, channels=[16, 32, 64, 128], depths=[1, 1, 1]):
    def __init__(self, in_channels=1, channels=[32, 64, 128, 256], depths=[1, 1, 1, 1]):
        super(AutoEncoder56, self).__init__()
        self.encoder = nn.Sequential(
            *[ResEncoderBlock(in_channels, channels[0]) for i in range(depths[0])],
            # ResBlock(dim=channels[0]),
            nn.MaxPool2d(2, stride=2), 
            *[ResEncoderBlock(channels[0], channels[1]) for i in range(depths[1])],
            # ResBlock(dim=channels[1]),
            nn.MaxPool2d(2, stride=2),  
            *[ResEncoderBlock(channels[1], channels[2]) for i in range(depths[2])],
            # ResBlock(dim=channels[2]),
            nn.MaxPool2d(2, stride=2),
            *[ResEncoderBlock(channels[2], channels[3]) for i in range(depths[3])],
            # ResBlock(dim=channels[3]),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(channels[3], channels[3], kernel_size=7, stride=1, padding=0),
            nn.BatchNorm2d(channels[3]),
            nn.GELU(),
        )
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=14, mode='bilinear'),
            *[ResDecoderBlock(in_channels=channels[3], out_channels=channels[2]) for i in range(depths[3])],
            nn.Upsample(scale_factor=2, mode='bilinear'),
            *[ResDecoderBlock(in_channels=channels[2], out_channels=channels[1]) for i in range(depths[2])],
            nn.Upsample(scale_factor=2, mode='bilinear'),
            *[ResDecoderBlock(in_channels=channels[1], out_channels=channels[0]) for i in range(depths[1])],
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(channels[0], channels[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels[0]),
            nn.GELU(),
            nn.Conv2d(channels[0], in_channels, kernel_size=3, stride=1, padding=1),
            nn.GELU()
            )


    def forward(self, x):
        x = self.encoder(x)
        # print(x.shape)
        x = self.decoder(x)
        return x

class Classifier56(nn.Module):
    def __init__(self, autoencoder, in_features, out_features):
        super(Classifier56, self).__init__()
        self.encoder = autoencoder.encoder

        self.norm = nn.LayerNorm(in_features, eps=1e-6) 
        # self.classifier = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(in_features, in_features),
        #     nn.GELU(),
        #     nn.Linear(in_features, out_features)
        # )
        # self.classifier = nn.Sequential(
        #         nn.Flatten(),
        #         nn.Linear(12544, 2048),
        #         nn.GELU(),
        #         nn.BatchNorm1d(num_features=2048),
        #         nn.Dropout(0.5),
        #         nn.Linear(2048, out_features)
        # )
        self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(256, 256),
                nn.GELU(),
                nn.BatchNorm1d(num_features=256),
                nn.Dropout(0.5),
                nn.Linear(256, out_features)
        )

    def forward(self, x):
        x = self.encoder(x)
        # print(x.shape)
        # x = self.norm(x.mean([-2,-1]))
        x = self.classifier(x)
        return x


class ResBlock(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        # self.norm = LayerNorm(dim, eps=1e-6)
        self.norm = nn.BatchNorm2d(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + x#+ self.drop_path(x)
        return x
    

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
