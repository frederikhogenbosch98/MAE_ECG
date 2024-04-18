import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
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
        self.norm = LayerNorm(dim, eps=1e-6)
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

    
class SelfBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SelfBlock, self).__init__()

        self.conv1x1_1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)#, stride=1, padding=1)
        self.conv1x1_2 = nn.Conv2d(in_channels, out_channels, kernel_size=1)#, stride=1, padding=1)
        self.conv3x3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, )
        self.bn = nn.BatchNorm2d(out_channels)
        self.gelu = nn.GELU()

    def forward(self, x):
        residual = x
        x = self.conv(x)
        x = self.bn(x)
        x = self.gelu(x)
        x += residual
        return x



class AutoEncoder128(nn.Module):
    def __init__(self, in_channels=1, channels=[96, 192, 384, 768], depths=[3, 3, 9, 3]):
        super(AutoEncoder128, self).__init__()
        self.blocksize = 2
        self.encoder = nn.Sequential(
            # DOWNSAMPLE 1
            nn.Conv2d(in_channels, channels[0], 3, stride=2, padding=1),
            LayerNorm(channels[0], eps=1e-6, data_format="channels_first"),
            # BLOCK 1
            *[Block(channels[0], channels[0]) for i in range(depths[0])],
            # DOWNSAMPLE 2
            LayerNorm(channels[0], eps=1e-6, data_format="channels_first"),
            nn.Conv2d(channels[0], channels[1], kernel_size=2, stride=2),
            # BLOCK 2
            *[Block(channels[1], channels[1]) for i in range(depths[1])], 
            # DOWNSAMPLE 3
            LayerNorm(channels[1], eps=1e-6, data_format="channels_first"),
            nn.Conv2d(channels[1], channels[2], kernel_size=2, stride=2),
            # BLOCK 3
            *[Block(channels[2], channels[2]) for i in range(depths[2])], 
            # DOWNSAMPLE 4
            LayerNorm(channels[2], eps=1e-6, data_format="channels_first"),
            nn.Conv2d(channels[2], channels[3], kernel_size=2, stride=2),
            # BLOCK 4
            *[Block(channels[3], channels[3]) for i in range(depths[3])] 
            
        )
        self.decoder = nn.Sequential(
            # BLOCK 4
            *[Block(channels[3], channels[3]) for i in range(depths[3])], 
            # UPSAMPLE 4
            LayerNorm(channels[3], eps=1e-6, data_format="channels_first"),
            nn.ConvTranspose2d(channels[3], channels[2], kernel_size=2, stride=2),
            # BLOCK 3
            *[Block(channels[2], channels[2]) for i in range(depths[2])], 
            # UPSAMPLE 3
            LayerNorm(channels[2], eps=1e-6, data_format="channels_first"),
            nn.ConvTranspose2d(channels[2], channels[1], kernel_size=2, stride=2),
            # BLOCK 2
            *[Block(channels[1], channels[1]) for i in range(depths[1])], 
            # UPSAMPLE 2
            LayerNorm(channels[1], eps=1e-6, data_format="channels_first"),
            nn.ConvTranspose2d(channels[1], channels[0], kernel_size=2, stride=2),
            # BLOCK 1
            *[Block(channels[0], channels[0]) for i in range(depths[0])],
            # UPSAMPLE 1
            nn.ConvTranspose2d(channels[0],in_channels, 3, stride=2, padding=1, output_padding=1),
            LayerNorm(in_channels, eps=1e-6, data_format="channels_first")



            # nn.ConvTranspose2d(256, 256, 3, stride=2, padding=1, output_padding=1),
            # nn.GELU(),
            # # nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1),
            # # nn.GELU(),
            # *[SelfBlock(256,256) for i in range(self.blocksize)],
            # nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            # nn.GELU(),
            # *[SelfBlock(128,128) for i in range(self.blocksize)],
            # nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            # nn.GELU(),
            # *[SelfBlock(64,64) for i in range(self.blocksize)],
            # nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            # nn.GELU(),
            # *[SelfBlock(32,32) for i in range(self.blocksize)],
            # nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            # nn.GELU(),
            # nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
            # nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        # print(x.shape)
        x = self.decoder(x)
        return x


# class Classifier128(nn.Module):
#     def __init__(self, autoencoder, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
#         super().__init__()
#         out_features = out_features or in_features
#         hidden_features = hidden_features or in_features
#         self.in_features = in_features
#         self.flatten = nn.Flatten()
#         self.encoder = autoencoder.encoder
#         self.fc1 = nn.Linear(in_features, hidden_features)
#         self.act = act_layer()
#         self.fc2 = nn.Linear(hidden_features, out_features)
#         self.drop = nn.Dropout(drop)

#     def forward(self, x):
#         x = self.encoder(x)
#         x = self.flatten(x)
#         x = self.fc1(x)
#         x = self.act(x)
#         # x = self.drop(x)
#         x = self.fc2(x)
#         # x = self.drop(x)
#         return x

class Classifier128(nn.Module):
    def __init__(self, autoencoder, in_features, out_features):
        super(Classifier128, self).__init__()
        self.encoder = autoencoder.encoder
        # self.decoder = autoencoder.decoder
        # self.classifier = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(input_dim, input_dim//2),
        #     nn.GELU(),
        #     nn.Linear(input_dim//2, input_dim//4),
        #     nn.GELU(),
        #     nn.Linear(input_dim//4, input_dim//8),
        #     nn.GELU(),
        #     nn.Linear(input_dim//8, num_classes)
        # )
        self.norm = nn.LayerNorm(in_features, eps=1e-6) 
        self.classifier = nn.Sequential(
            # nn.Flatten()
            nn.Linear(in_features, in_features),
            nn.GELU(),
            nn.Linear(in_features, out_features)
            # nn.GELU(),
            # nn.Linear(1024, out_features)
            # nn.Dropout(p=0.3)
        )

    def forward(self, x):
        x = self.encoder(x)
        # x = self.decoder(x)
        x = self.norm(x.mean([-2,-1]))
        x = self.classifier(x)
        # return F.softmax(x, dim=1)
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
