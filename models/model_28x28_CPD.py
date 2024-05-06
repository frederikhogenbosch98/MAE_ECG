import torch
import torch.nn as nn
import torch.nn.functional as F
import tltorch
import tensorly as tl
from tensorly.decomposition import parafac, tucker, parafac2, partial_tucker
import numpy as np
from old_files.ConvTranpose_CP import FactorizedConvTranspose

tl.set_backend('pytorch')


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
        # self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.dwconv = tltorch.FactorizedConv(dim, dim, kernel_size=7, padding=3, order=2, rank=25)
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



class ParafacConvolution2D(nn.Module):
    def __init__(self, conv, R):
        super(ParafacConvolution2D,self).__init__()
        weights, factors = parafac(conv.weight.detach(), init='random', rank=R)
        self.s_to_r = nn.Conv2d(conv.in_channels, R, (1, 1), stride=1, padding=0, bias=False)
        self.s_to_r.weight.data = factors[1].permute(1,0).unsqueeze(-1).unsqueeze(-1)

        self.depth_vert = nn.Conv2d(R, R, (factors[2].shape[0], 1), groups=R, stride=(conv.stride[0],1), padding=(conv.padding[0],0), bias=False)
        self.depth_vert.weight.data = factors[2].permute(1,0).unsqueeze(1).unsqueeze(-1)

        self.depth_hor = nn.Conv2d(R, R, (1, factors[3].shape[0]), groups=R, stride=(1, conv.stride[1]), padding=(0, conv.padding[1]), bias=False)
        self.depth_hor.weight.data = factors[3].permute(1,0).unsqueeze(1).unsqueeze(1)

        self.r_to_t = nn.Conv2d(R, conv.out_channels, (1, 1), stride=1, padding=0, bias=True)
        self.r_to_t.weight.data = factors[0].unsqueeze(-1).unsqueeze(-1)

        if conv.bias is not None:
            self.r_to_t.bias.data = conv.bias.data
    
    def forward(self, x):
        x = self.s_to_r(x)
        x = self.depth_vert(x)
        x = self.depth_hor(x)
        x = self.r_to_t(x)
        return x



class ParafacConvolutionTranspose2D(nn.Module):
    def __init__(self, conv, R):
        super(ParafacConvolutionTranspose2D, self).__init__()
        weights, factors = parafac(conv.weight.detach(), init='svd', rank=R)
        # weights, factors = tucker(conv.weight.detach(), init='svd', rank=R)

        self.s_to_r = nn.Conv2d(conv.in_channels, R, (1, 1), stride=1, padding=0, bias=False)
        self.s_to_r.weight = nn.Parameter(factors[0].permute(1,0).unsqueeze(-1).unsqueeze(-1))
        # print(self.s_to_r.weight.data.shape)

        self.depth_vert = nn.ConvTranspose2d(R, R, (factors[2].shape[0], 1), groups=R, stride=(conv.stride[0], 1), 
                                             padding=(conv.padding[0], 0), output_padding=(conv.output_padding[0], 0), bias=False)
        self.depth_vert.weight = nn.Parameter(factors[2].permute(1, 0).unsqueeze(1).unsqueeze(-1))

        # print(self.depth_vert.weight.data.shape)
        self.depth_hor = nn.ConvTranspose2d(R, R, (1, factors[3].shape[0]), groups=R, stride=(1, conv.stride[1]), 
                                            padding=(0, conv.padding[1]), output_padding=(0, conv.output_padding[1]), bias=False)
        self.depth_hor.weight = nn.Parameter(factors[3].permute(1, 0).unsqueeze(1).unsqueeze(1))
        # print(self.depth_hor.weight.data.shape)
        self.r_to_t = nn.Conv2d(R, conv.out_channels, (1, 1), stride=1, padding=0, bias=True)
        self.r_to_t.weight = nn.Parameter(factors[1].unsqueeze(-1).unsqueeze(-1))
        # print(self.r_to_t.weight.data.shape)
        if conv.bias is not None:
            self.r_to_t.bias.data = conv.bias.data
    
    def forward(self, x):
        x = self.s_to_r(x)
        # print(x.shape)
        x = self.depth_vert(x)
        # print(x.shape)
        x = self.depth_hor(x)
        # print(x.shape)
        x = self.r_to_t(x)
        # print(x.shape)
        return x


class TuckerConvolutionTranspose2D(nn.Module):
    def __init__(self, conv, rank_s, rank_t):
        super(TuckerConvolutionTranspose2D, self).__init__()
        # Decompose the original weight tensor using Tucker decomposition
        core, factors = tucker(conv.weight.detach(), rank=[rank_t, rank_s, None, None])
        
        # factors[0] corresponds to the input channel factor matrix U_s
        # factors[1] corresponds to the output channel factor matrix U_t
        # core is the compressed core tensor
        print(len(factors))
        
        self.s_to_rs = nn.ConvTranspose2d(conv.in_channels, rank_s, (1, 1), stride=1, padding=0, bias=False)
        self.s_to_rs.weight = nn.Parameter(factors[0].unsqueeze(-1).unsqueeze(-1))
        
        self.core_conv = nn.ConvTranspose2d(rank_s, rank_t, conv.kernel_size, stride=conv.stride, 
                                            padding=conv.padding, output_padding=conv.output_padding, 
                                            groups=1, bias=False)
        self.core_conv.weight = nn.Parameter(core)#.permute(3, 2, 0, 1))
        
        self.rt_to_t = nn.ConvTranspose2d(rank_t, conv.out_channels, (1, 1), stride=1, padding=0, bias=True)
        self.rt_to_t.weight = nn.Parameter(factors[1].permute(1, 0).unsqueeze(-1).unsqueeze(-1))
        if conv.bias is not None:
            self.rt_to_t.bias.data = conv.bias.data

    def forward(self, x):
        x = self.s_to_rs(x)
        x = self.core_conv(x)
        x = self.rt_to_t(x)
        return x



class Encoder28_CPD(nn.Module):
    def __init__(self, R, factorization='cp'):
        super(Encoder28_CPD, self).__init__()
        # self.TuckerRank = [15, 15, 15, R-10
        # R = R*np.ones(4)
        # print(R)
        self.fact_conv1 = tltorch.FactorizedConv.from_conv(nn.Conv2d(1, 16, 3, stride=2, padding=1), rank=R, decompose_weights=True, factorization=factorization)
        self.fact_conv2 = tltorch.FactorizedConv.from_conv(nn.Conv2d(16, 32, 3, stride=2, padding=1), rank=R, decompose_weights=True, factorization=factorization)
        self.fact_conv3 = tltorch.FactorizedConv.from_conv(nn.Conv2d(32, 64, 7), rank=R, decompose_weights=True, factorization=factorization)

        # self.encoder = nn.Sequential( 
        #     tltorch.FactorizedConv.from_conv(nn.Conv2d(1, 16, 3, stride=2, padding=1), rank=R, decompose_weights=True, factorization=factorization),
        #     LayerNorm(16, eps=1e-6, data_format="channels_first"),
        #     *[Block(dim=16) for i in range(1)],
        #     LayerNorm(16, eps=1e-6, data_format="channels_first"),
        #     tltorch.FactorizedConv.from_conv(nn.Conv2d(16, 32, 3, stride=2, padding=1), rank=R, decompose_weights=True, factorization=factorization),
        #     *[Block(32, 32) for i in range(3)],
        #     LayerNorm(32, eps=1e-6, data_format="channels_first"),
        #     tltorch.FactorizedConv.from_conv(nn.Conv2d(32, 64, 7), rank=R, decompose_weights=True, factorization=factorization),
        #     *[Block(64, 64) for i in range(1)]
        #     # nn.Conv2d(16, 32, 3, stride=2, padding=1),
        #     # nn.GELU(),
        #     # nn.Conv2d(32, 64, 7),
        #     # nn.GELU(),
        #     # nn.Conv2d(64, 128, 7)
            
        # )

        # self.fact_conv1 = ParafacConvolution2D(nn.Conv2d(1, 16, 3, stride=2, padding=1), R=R)
        # self.fact_conv2 = ParafacConvolution2D(nn.Conv2d(16, 32, 3, stride=2, padding=1), R=R)
        # self.fact_conv3 = ParafacConvolution2D(nn.Conv2d(32, 64, 7), R=R)
        # self.bn1 = nn.BatchNorm2d(16)
        # self.bn2 = nn.BatchNorm2d(32)
        # self.bn3 = nn.BatchNorm2d(64)

        # self.params = 0
        # for i in range(4):
        #     self.params += self.fact_conv1.weight.factors[i].shape[0] * self.fact_conv1.weight.factors[i].shape[1]
        #     self.params += self.fact_conv2.weight.factors[i].shape[0] * self.fact_conv2.weight.factors[i].shape[1]
        #     self.params += self.fact_conv3.weight.factors[i].shape[0] * self.fact_conv3.weight.factors[i].shape[1]

        # print(f'number of parameters: {self.params}')
        # self.fact_conv1 = ParafacConvolution2D(self.conv1, R=R)
        # self.fact_conv2 = ParafacConvolution2D(self.conv2, R=R)
        # self.fact_conv3 = ParafacConvolution2D(self.conv3, R=R)

        self.gelu = nn.GELU() 
         
    def forward(self, x):
        x = self.fact_conv1(x) 
        # x = self.bn1(x)
        x = self.gelu(x)
        x = self.fact_conv2(x)
        # x = self.bn2(x)
        x = self.gelu(x)
        x = self.fact_conv3(x)
        x = self.gelu(x) 
        # x = self.bn3(x)
        # print(f'aftet encoder: {x.shape}')
        # print(x.shape)
        # x = self.encoder(x)
        return x


class Decoder28_CPD(nn.Module):
    def __init__(self, R, factorization='cp'):
        super(Decoder28_CPD, self).__init__()
        
        # self.conv1 = nn.Conv2d(64, 32, 7, padding=3) 
        # self.conv2 = nn.Conv2d(32, 16, 3, stride=1, padding=1)
        # self.conv3 = nn.Conv2d(16, 1, 3, stride=1, padding=1)
        # self.tryout1 = FactorizedConvTranspose(64, 32, 7, order=2, rank=R)
        # self.tryout2 = FactorizedConvTranspose(32, 16, 3, order=2, stride=2, padding=1, output_padding=1, rank=R)
        # self.tryout3 = FactorizedConvTranspose(16, 1, 3,  order=2, stride=2, padding=1, output_padding=1, rank=R)

        # print(type(self.tryout1))
        # print(self.tryout1.weight.shape)

        # self.transconv1 = nn.ConvTranspose2d(64, 32, 7)
        # self.transconv2 = nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1)
        # self.transconv3 = nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1)

        # # self.tucker_layer2 = TuckerTransposeConv(in_channels=32, out_channels=16, kernel_size=(3, 3), stride=2, padding=1, output_padding=1, ranks=[15, 15])
        # self.tucker_layer3 = TuckerTransposeConv(in_channels=16, out_channels=1, kernel_size=(3, 3), stride=2, padding=1, output_padding=1, ranks=[15, 15])
        # print(type(self.tucker_layer))
        # self.tryout1 = ParafacConvolutionTranspose2D(nn.ConvTranspose2d(64, 32, 7), R=R)
        # self.tryout2 = ParafacConvolutionTranspose2D(nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), R=R)
        # self.tryout3 = ParafacConvolutionTranspose2D(nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1), R=R)

        self.tryout1 = TuckerConvolutionTranspose2D(nn.ConvTranspose2d(64, 32, 7), rank_s=R, rank_t=R)
        self.tryout2 = TuckerConvolutionTranspose2D(nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), rank_s=R, rank_t=R)
        self.tryout3 = TuckerConvolutionTranspose2D(nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1), rank_s=R, rank_t=R)

        # self.fact_conv1 = tltorch.FactorizedConv.from_conv(self.conv1, rank=R, decompose_weights=True, factorization=factorization)
        # self.fact_conv2 = tltorch.FactorizedConv.from_conv(self.conv2, rank=R, decompose_weights=True, factorization=factorization)
        # self.fact_conv3 = tltorch.FactorizedConv.from_conv(self.conv3, rank=R, decompose_weights=True, factorization=factorization)
        # self.fact_convtrans1 = tltorch.FactorizedConv.from_conv(self.transconv1, rank=R, decompose_weights=True, factorization=factorization)
        # self.fact_convtrans2 = tltorch.FactorizedConv.from_conv(self.transconv2, rank=R, decompose_weights=True, factorization=factorization)
        # self.fact_convtrans3 = tltorch.FactorizedConv.from_conv(self.transconv3, rank=R, decompose_weights=True, factorization=factorization)
        # print(self.fact_convtrans.weight.shape)

        # self.up1 = nn.Upsample(size=(7, 7), mode='bilinear', align_corners=False)
        # self.up2 = nn.Upsample(size=(28, 28), mode='bilinear', align_corners=False)

        self.gelu = nn.GELU() 
        # self.sigmoid = nn.Sigmoid()

        # self.fact_conv1 = ParafacConvolution2D(self.conv1, R=R)
        # self.fact_conv2 = ParafacConvolution2D(self.conv2, R=R)
        # self.fact_conv3 = ParafacConvolution2D(self.conv3, R=R)


        # self.decoder = nn.Sequential(
        #     nn.ConvTranspose2d(64, 32, 7),
        #     nn.GELU(),
        #     nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
        #     nn.GELU(),
        #     nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
        #     nn.GELU()
        # )

    def forward(self, x):

        x = self.tryout1(x)
        x = self.gelu(x)
        x = self.tryout2(x)
        x = self.gelu(x)
        x = self.tryout3(x)
        x = self.gelu(x)
        # x = self.decoder(x)
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
        self.norm = nn.LayerNorm(input_dim, eps=1e-6) 
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # nn.Linear(input_dim, num_classes),
            nn.Linear(input_dim, input_dim//2),
            nn.GELU(),
            nn.Linear(input_dim//2, num_classes)
            # nn.Linear(input_dim, num_classes)
        )

    def forward(self, x):
        x = self.encoder(x)
        # x = self.norm(x.mean([-2,-1]))
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
            # return x