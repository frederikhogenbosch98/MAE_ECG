import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=7, stride=1, padding=3)
        # self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)
        self.norm1 = nn.BatchNorm2d(dim)
        # self.norm2 = nn.BatchNorm2d(dim)
        self.act = nn.GELU()


    def forward(self, x):
        # input = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)
        # x = input + x
        return x



# class AutoEncoder56(nn.Module):
#     # def __init__(self, in_channels=1, channels=[16, 32, 64, 128], depths=[1, 1, 1, 1]):
#     def __init__(self, in_channels=1, channels=[32, 64, 128, 256], depths=[1, 1, 3, 1]):
#         super(AutoEncoder56, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Conv2d(in_channels, channels[0], 3, stride=1, padding=1),
#             *[ResBlock(channels[0]) for i in range(depths[0])],
#             # ResBlock(dim=channels[0]),
#             nn.MaxPool2d(2, stride=2), 
#             nn.Conv2d(channels[0], channels[1], 3, stride=1, padding=1),
#             *[ResBlock(channels[1]) for i in range(depths[1])],
#             # ResBlock(dim=channels[1]),
#             nn.MaxPool2d(2, stride=2),  
#             nn.Conv2d(channels[1], channels[2], 3, stride=1, padding=1),
#             *[ResBlock(channels[2]) for i in range(depths[2])],
#             # ResBlock(dim=channels[2]),
#             nn.MaxPool2d(2, stride=2),
#             nn.Conv2d(channels[2], channels[3], 3, stride=1, padding=1),
#             *[ResBlock(channels[3]) for i in range(depths[3])],
#             # ResBlock(dim=channels[3]),
#             nn.MaxPool2d(2, stride=2),
#             # nn.Conv2d(channels[3], channels[3], kernel_size=7, stride=1, padding=0),
#             # nn.BatchNorm2d(channels[3]),
#             # nn.GELU(),
#         )

#         self.decoder = nn.Sequential(
#             nn.Upsample(scale_factor=2, mode='bilinear'),
#             nn.Conv2d(channels[3], channels[2], 3, stride=1, padding=1),
#             *[ResBlock(channels[2]) for i in range(depths[3])],
#             nn.Upsample(scale_factor=2, mode='bilinear'),
#             nn.Conv2d(channels[2], channels[1], 3, stride=1, padding=1),
#             *[ResBlock(channels[1]) for i in range(depths[2])],
#             nn.Upsample(scale_factor=2, mode='bilinear'),
#             nn.Conv2d(channels[1], channels[0], 3, stride=1, padding=1),
#             *[ResBlock(channels[0]) for i in range(depths[1])],
#             nn.Upsample(scale_factor=2, mode='bilinear'),
#             nn.Conv2d(channels[0], channels[0], kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(channels[0]),
#             nn.GELU(),
#             nn.Conv2d(channels[0], in_channels, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(in_channels),
#             nn.GELU()
#             )


#     def forward(self, x):
#         x = self.encoder(x)
#         # print(x.shape)
#         x = self.decoder(x)
#         return x

class Classifier56(nn.Module):
    def __init__(self, autoencoder, in_features, out_features):
        super(Classifier56, self).__init__()
        self.encoder = autoencoder.encoder
        self.enc1 = autoencoder.enc1
        self.pool1 = autoencoder.pool1
        self.enc2 = autoencoder.enc2
        self.pool2 = autoencoder.pool2
        self.enc3 = autoencoder.enc3
        self.pool3 = autoencoder.pool3


        self.norm = nn.LayerNorm(in_features, eps=1e-6) 

        # self.classifier = nn.Sequential(
        #         nn.Flatten(),
        #         nn.Linear(12544, 2048),
        #         nn.GELU(),
        #         nn.BatchNorm1d(num_features=2048),
        #         nn.Dropout(0.5),
        #         nn.Linear(2048, out_features)
        # )
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(12544, 256),
                # nn.Linear(64, 64),
                nn.GELU(),
                nn.BatchNorm1d(num_features=256),
                nn.Dropout(0.5),
                nn.Linear(256, out_features)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.pool1(x)
        x = self.enc2(x)
        x = self.pool2(x)
        x = self.enc3(x)
        x = self.pool3(x)
        # x = self.encoder(x)
        # print(x.shape)
        # x = self.avg_pool(x)
        x = self.classifier(x)
        return x


# class ResBlock(nn.Module):
#     r""" ConvNeXt Block. There are two equivalent implementations:
#     (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
#     (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
#     We use (2) as we find it slightly faster in PyTorch
    
#     Args:
#         dim (int): Number of input channels.
#         drop_path (float): Stochastic depth rate. Default: 0.0
#         layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
#     """
#     def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
#         super().__init__()
#         self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
#         # self.norm = LayerNorm(dim, eps=1e-6)
#         self.norm = nn.BatchNorm2d(dim, eps=1e-6)
#         self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
#         self.act = nn.GELU()
#         self.pwconv2 = nn.Linear(4 * dim, dim)
#         self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
#                                     requires_grad=True) if layer_scale_init_value > 0 else None
#         # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

#     def forward(self, x):
#         input = x
#         x = self.dwconv(x)
#         x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
#         x = self.norm(x)
#         x = self.pwconv1(x)
#         x = self.act(x)
#         x = self.pwconv2(x)
#         if self.gamma is not None:
#             x = self.gamma * x
#         x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

#         x = input + x#+ self.drop_path(x)
#         return x
    

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

# class Encoder56(nn.Module):
#     def __init__(self, in_channels=1, channels=[16, 32, 64, 128], depths=[1, 1, 1]):
#     # def __init__(self, in_channels=1, channels=[64, 128, 256], depths=[1, 1, 1]):
#         super(Encoder56, self).__init__()

#         self.conv1 = nn.Conv2d(in_channels, channels[0], kernel_size=3, stride=1, padding=1)
#         self.norm1 =  nn.BatchNorm2d(channels[0])
#         self.conv2 = nn.Conv2d(channels[0], channels[0], kernel_size=3, stride=1, padding=1)
#         self.norm2 = nn.BatchNorm2d(channels[0])
#         self.conv3 = nn.Conv2d(channels[0], channels[1], kernel_size=3, stride=1, padding=1)
#         self.norm3 = nn.BatchNorm2d(channels[1]) 
#         self.conv4 = nn.Conv2d(channels[1], channels[1], kernel_size=3, stride=1, padding=1)
#         self.norm4 = nn.BatchNorm2d(channels[1]) 
#         self.conv5 = nn.Conv2d(channels[1], channels[2], kernel_size=3, stride=1, padding=1)
#         self.norm5 = nn.BatchNorm2d(channels[2]) 
#         self.conv6 = nn.Conv2d(channels[2], channels[2], kernel_size=3, stride=1, padding=1)
#         self.norm6 = nn.BatchNorm2d(channels[2]) 
#         self.conv7 = nn.Conv2d(channels[2], channels[2], kernel_size=3, stride=1, padding=1)
#         self.norm7 = nn.BatchNorm2d(channels[2]) 

#         self.maxpool2d = nn.MaxPool2d(2, stride=2)
#         self.gelu = nn.GELU()

#     def forwward(self, x):
#         input = x
#         x = self.conv1(x)
#         x = self.norm1(x)
#         x = self.gelu(x)
#         x = self.conv2(x)
#         x = self.norm2(x)
#         x = self.gelu(x) 
#         x = self.conv3(x)
#         x = self.norm3(x)
#         x = self.gelu(x) 
#         x = self.conv4(x)
#         x = self.norm4(x)
#         x = self.gelu(x) 
#         x = self.conv5(x)
#         x = self.norm5(x)
#         x = self.gelu(x) 
#         x = self.conv6(x)
#         x = self.norm6(x)
#         x = self.gelu(x) 
#         x = self.conv7(x)
#         x = self.norm7(x)
#         x = self.gelu(x)
#         x = x + input

#         self.encoder = nn.Sequential(
#             # LAYER 1
#             nn.Conv2d(in_channels, channels[0], kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(channels[0]),
#             nn.GELU(),
#             # LAYER 2
#             nn.Conv2d(channels[0], channels[0], kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(channels[0]),
#             nn.GELU(),
#             # LAYER 3
#             nn.MaxPool2d(2, stride=2),
#             # LAYER 4
#             nn.Conv2d(channels[0], channels[1], kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(channels[1]),
#             nn.GELU(),
#             # LAYER 5
#             nn.Conv2d(channels[1], channels[1], kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(channels[1]),
#             nn.GELU(),
#             # LAYER 6
#             nn.MaxPool2d(2, stride=2),
#             # LAYER 6
#             nn.Conv2d(channels[1], channels[2], kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(channels[2]),
#             nn.GELU(),
#             # LAYER 7
#             nn.Conv2d(channels[2], channels[2], kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(channels[2]),
#             nn.GELU(),
#             # LAYER 8
#             nn.MaxPool2d(2, stride=2)
#         )

# class AutoEncoder56(nn.Module):
#     def __init__(self, in_channels=1, channels=[16, 32, 64, 128], depths=[1, 1, 1]):
#     # def __init__(self, in_channels=1, channels=[64, 128, 256], depths=[1, 1, 1]):
#         super(AutoEncoder56, self).__init__()
#         self.encoder = nn.Sequential(
#             # LAYER 1
#             nn.Conv2d(in_channels, channels[0], kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(channels[0]),
#             nn.GELU(),
#             # nn.Dropout(p=0.5),
#             # LAYER 2
#             nn.Conv2d(channels[0], channels[0], kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(channels[0]),
#             nn.GELU(),
#             # nn.Dropout(p=0.5),
#             # LAYER 3
#             nn.MaxPool2d(2, stride=2),
#             # LAYER 4
#             nn.Conv2d(channels[0], channels[1], kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(channels[1]),
#             nn.GELU(),
#             # nn.Dropout(p=0.5),
#             # LAYER 5
#             nn.Conv2d(channels[1], channels[1], kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(channels[1]),
#             nn.GELU(),
#             # nn.Dropout(p=0.5),
#             # LAYER 6
#             nn.MaxPool2d(2, stride=2),
#             # LAYER 6
#             nn.Conv2d(channels[1], channels[2], kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(channels[2]),
#             nn.GELU(),
#             # nn.Dropout(p=0.5),
#             # LAYER 7
#             nn.Conv2d(channels[2], channels[2], kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(channels[2]),
#             nn.GELU(),
#             # nn.Dropout(p=0.5),
#             # LAYER 8
#             nn.MaxPool2d(2, stride=2)
#         )
#         self.decoder = nn.Sequential(
#             nn.Upsample(scale_factor=2, mode='bilinear'),
#             nn.Conv2d(channels[2], channels[2], kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(channels[2]),
#             nn.GELU(),
#             # nn.Dropout(p=0.5),
#             # Corresponds to LAYER 5 in Encoder
#             nn.Conv2d(channels[2], channels[2], kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(channels[2]),
#             nn.GELU(),
#             # nn.Dropout(p=0.5),
#             # Corresponds to LAYER 4 in Encoder
#             nn.Upsample(scale_factor=2, mode='bilinear'),
#             nn.Conv2d(channels[2], channels[1], kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(channels[1]),
#             nn.GELU(),
#             # nn.Dropout(p=0.5),
#             # Corresponds to LAYER 5 in Encoder
#             nn.Conv2d(channels[1], channels[1], kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(channels[1]),
#             nn.GELU(),
#             # nn.Dropout(p=0.5),
#             # Corresponds to LAYER 4 in Encoder
#             nn.Upsample(scale_factor=2, mode='bilinear'),
#             nn.Conv2d(channels[1], channels[0], kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(channels[0]),
#             nn.GELU(),
#             # nn.Dropout(p=0.5),
#             # Corresponds to LAYER 2 in Encoder
#             nn.Conv2d(channels[0], channels[0], kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(channels[0]),
#             nn.GELU(),
#             # nn.Dropout(p=0.5),
#             # Corresponds to LAYER 1 in Encoder
#             nn.Conv2d(channels[0], in_channels, kernel_size=3, stride=1, padding=1),
#             nn.Sigmoid(),


#         )

#     def forward(self, x):
#         x = self.encoder(x)
#         # print(x.shape)
#         x = self.decoder(x)
#         return x


class AutoEncoder56(nn.Module):
    def __init__(self, in_channels=1, channels=[16, 32, 64, 128], depths=[1, 1, 1]):
        super(AutoEncoder56, self).__init__()
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, channels[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels[0]),
            nn.GELU(),
            nn.Conv2d(channels[0], channels[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels[0]),
            nn.GELU(),
        )
        self.pool1 = nn.MaxPool2d(2, stride=2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(channels[0], channels[1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels[1]),
            nn.GELU(),
            nn.Conv2d(channels[1], channels[1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels[1]),
            nn.GELU(),
        )
        self.pool2 = nn.MaxPool2d(2, stride=2)

        self.enc3 = nn.Sequential(
            nn.Conv2d(channels[1], channels[2], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels[2]),
            nn.GELU(),
            nn.Conv2d(channels[2], channels[2], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels[2]),
            nn.GELU(),
        )
        self.pool3 = nn.MaxPool2d(2, stride=2)

        # Decoder
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec3 = nn.Sequential(
            nn.Conv2d(channels[2] + channels[2], channels[1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels[1]),
            nn.GELU(),
        )

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec2 = nn.Sequential(
            nn.Conv2d(channels[1] + channels[1], channels[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels[0]),
            nn.GELU(),
        )

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec1 = nn.Sequential(
            nn.Conv2d(channels[0] + channels[0], in_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        x = self.pool1(enc1)
        enc2 = self.enc2(x)
        x = self.pool2(enc2)
        enc3 = self.enc3(x)
        x = self.pool3(enc3)

        # Decoder
        x = self.up3(x)
        x = torch.cat([x, enc3], dim=1)
        x = self.dec3(x)

        x = self.up2(x)
        x = torch.cat([x, enc2], dim=1)
        x = self.dec2(x)

        x = self.up1(x)
        x = torch.cat([x, enc1], dim=1)
        x = self.dec1(x)

        return x


