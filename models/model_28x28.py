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


        

class AutoEncoder28(nn.Module):
    def __init__(self, in_channels=1):
        super(AutoEncoder28, self).__init__()
        self.encoder = nn.Sequential( 
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            # nn.BatchNorm2d(16),
            nn.GELU(),
            # LayerNorm(16, eps=1e-6, data_format="channels_first"),
            # *[Block(16, 16) for i in range(1)],
            # LayerNorm(16, eps=1e-6, data_format="channels_first"),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            # nn.BatchNorm2d(32),
            nn.GELU(),
            # *[Block(32, 32) for i in range(3)],
            # LayerNorm(32, eps=1e-6, data_format="channels_first"),
            nn.Conv2d(32, 64, 7),
            # nn.BatchNorm2d(64),
            nn.GELU()
            # *[Block(64, 64) for i in range(1)]
            # nn.Conv2d(16, 32, 3, stride=2, padding=1),
            # nn.GELU(),
            # nn.Conv2d(32, 64, 7),
            # nn.GELU(),
            # nn.Conv2d(64, 128, 7)
            
        )

        # self.decoder = nn.Sequential(
        #     nn.Conv2d(
        #     in_channels=64, 
        #     out_channels=32, 
        #     kernel_size=1),
        #     Block(32, 32),
        #     nn.Upsample(scale_factor=7, mode='bilinear'),
        #     Block(32, 32),
        #     nn.Upsample(scale_factor=2, mode='bilinear'),
        #     Block(32, 32),
        #     nn.Upsample(scale_factor=2, mode='bilinear'),
        #     nn.Conv2d(
        #     in_channels=32,
        #     out_channels=1,
        #     kernel_size=1)

        # )



        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7),
            # nn.BatchNorm2d(32),
            nn.GELU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            # nn.BatchNorm2d(16),
            nn.GELU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
            # nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        # self.decoder = nn.Sequential(
        #     nn.Conv2d(64, 56)
        # )

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



# class AutoEncoder28(nn.Module):
#     def __init__(self, in_channels=1):
#         super(AutoEncoder28, self).__init__()
#         self.encoder = nn.Sequential( 
#             nn.Conv2d(1, 16, 3, stride=2, padding=1),
#             nn.GELU(),
#             nn.Conv2d(16, 32, 3, stride=2, padding=1),
#             nn.GELU(),
#             nn.Conv2d(32, 64, 7),
#             # nn.GELU(),
#             # nn.Conv2d(64, 128, 7)
            
#         )

#         self.decoder = nn.Sequential(
#             nn.ConvTranspose2d(64, 32, 7),
#             nn.GELU(),
#             nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
#             nn.GELU(),
#             nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         x = self.encoder(x)
#         x = self.decoder(x)
#         return x
