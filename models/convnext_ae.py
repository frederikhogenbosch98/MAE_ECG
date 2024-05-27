
import torch
import torch.nn as nn

from timm.models.layers import trunc_normal_
from .convnextv2 import ConvNeXtV2
from .convnextv2 import Block
from .utils import LayerNorm, GRN
class FCMAE(nn.Module):
    """ Fully Convolutional Masked Autoencoder with ConvNeXtV2 backbone
    """
    def __init__(
                self,
                img_size=128,
                in_chans=1,
                depths=[1, 1, 3, 1],
                dims=[32, 64, 128, 256],
                decoder_depth=4,
                decoder_embed_dim=512,
                norm_pix_loss=False):
        super().__init__()

        # configs
        self.img_size = img_size
        self.depths = depths
        self.imds = dims
        self.decoder_embed_dim = decoder_embed_dim
        self.decoder_depth = decoder_depth
        self.norm_pix_loss = norm_pix_loss

        # encoder
        self.encoder = ConvNeXtV2(
            in_chans=in_chans, depths=depths, dims=dims)
        # decoder
        # self.proj = nn.Conv2d(
        #     in_channels=dims[-1], 
        #     out_channels=decoder_embed_dim, 
        #     kernel_size=1)


        self.rev_dims = list(reversed(dims))
        self.rev_depths = list(reversed(depths))
        
        self.decoder = nn.ModuleList()
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=self.rev_dims[i], drop_path=0.) for j in range(self.rev_depths[i])]
            )
            self.decoder.append(stage)

        
        self.upsample_layers = nn.ModuleList()
        for i in range(3):
            upsample_layer = nn.Sequential(
                    LayerNorm(self.rev_dims[i], eps=1e-6, data_format="channels_first"),
                    nn.ConvTranspose2d(self.rev_dims[i], self.rev_dims[i+1], kernel_size=2, stride=2),
            )
            self.upsample_layers.append(upsample_layer)
        stem = nn.Sequential(
            nn.ConvTranspose2d(self.rev_dims[-1], in_chans, kernel_size=4, stride=4),
            LayerNorm(in_chans, eps=1e-6, data_format="channels_first")
        )
        self.upsample_layers.append(stem)
        self.pred = nn.Conv2d(
            in_channels=decoder_embed_dim,
            out_channels=1,
            kernel_size=1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            w = m.weight.data
            trunc_normal_(w.view([w.shape[0], -1]))
            nn.init.constant_(m.bias, 0)
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    
    def forward_encoder(self, imgs):
        # generate random masks
        # encoding
        x = self.encoder(imgs)
        return x

    def forward_decoder(self, x):
        # x = self.proj(x)
        # x = self.decoder(x)
        # print(x.shape)
        for i in range(4):
            x = self.decoder[i](x)
            # print(x.shape)
            x = self.upsample_layers[i](x) 
            # print(x.shape)
        return x



    def forward(self, imgs):
        x = self.forward_encoder(imgs)
        # print(x.shape)
        x = self.forward_decoder(x)
        # print(x.shape)
        return x
