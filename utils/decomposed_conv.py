import tltorch
import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tensorly.decomposition import parafac
import time
import matplotlib.pyplot as plt
import tqdm
from fvcore.nn import FlopCountAnalysis
import sys, os

sys.path.append(os.path.abspath(os.path.join('..')))

from models._11am_back import AutoEncoder11
from models._11am_un import AutoEncoder11_UN, Classifier_UN

_CONVOLUTION = {1: F.conv1d, 2: F.conv2d, 3: F.conv3d}

def general_conv1d(x, kernel, mode, bias=None, stride=1, padding=0, groups=1, dilation=1, verbose=False):
    """General 1D convolution along the mode-th dimension

    Uses an ND convolution under the hood

    Parameters
    ----------
    x : batch-dize, in_channels, K1, ..., KN
    kernel : out_channels, in_channels/groups, K{mode}
    mode : int
        weight along which to perform the decomposition
    stride : int
    padding : int
    groups : 1
        typically would be equal to the number of input-channels
        at least for CP convolutions

    Returns
    -------
    x convolved with the given kernel, along dimension `mode`
    """
    if verbose:
        print(f'Convolving {x.shape} with {kernel.shape} along mode {mode}, '
              f'stride={stride}, padding={padding}, groups={groups}')

    def _pad_value(value, mode, order, padding=1):
        return tuple([value if i == (mode - 2) else padding for i in range(order)])

    ndim = np.ndim(x)
    order = ndim - 2
    for i in range(2, ndim):
        if i != mode:
            kernel = kernel.unsqueeze(i)

    return _CONVOLUTION[order](x, kernel, bias=bias, 
                               stride=_pad_value(stride, mode, order),
                               padding=_pad_value(padding, mode, order, padding=0), 
                               dilation=_pad_value(dilation, mode, order), 
                               groups=groups)


class TensorlyConv(nn.Module):
    def __init__(self, conv, R):
        super(TensorlyConv, self).__init__()
        self.cp_tensor = parafac(conv.weight.detach(), init='svd', rank=R)
        self.shape = self.cp_tensor.shape
        self.rank = self.cp_tensor.rank
        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.bias = conv.bias
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels

        # Define the four convolution steps as separate modules for compatibility with fvcore
        self.first_conv = nn.Conv1d(in_channels=self.in_channels, out_channels=self.rank, kernel_size=1)
        self.general_conv1 = nn.Conv1d(in_channels=self.rank, out_channels=self.rank, kernel_size=1)
        self.general_conv2 = nn.Conv1d(in_channels=self.rank, out_channels=self.rank, kernel_size=1)
        self.last_conv = nn.Conv1d(in_channels=self.rank, out_channels=self.out_channels, kernel_size=1)

    def forward(self, x):
        batch_size = x.shape[0]
        order = len(self.shape) - 2
        padding = self.padding
        stride = self.stride
        dilation = self.dilation

        if isinstance(padding, int):
            padding = (padding, )*order
        if isinstance(stride, int):
            stride = (stride, )*order
        if isinstance(dilation, int):
            dilation = (dilation, )*order

        # Change the number of channels to the rank
        x_shape = list(x.shape)
        x = x.reshape((batch_size, x_shape[1], -1)).contiguous()

        # First conv == tensor contraction
        x = self.first_conv(x)

        x_shape[1] = self.rank
        x = x.reshape(x_shape)

        # Convolve over non-channels
        for i in range(order):
            kernel = self.cp_tensor.factors[i+2].transpose(0, 1).unsqueeze(1)
            x = general_conv1d(x.contiguous(), kernel, i+2, stride=stride[i], padding=padding[i], dilation=dilation[i], groups=self.rank)

        # Revert back number of channels from rank to output_channels
        x_shape = list(x.shape)
        x = x.reshape((batch_size, x_shape[1], -1))

        # Last conv == tensor contraction
        x = self.last_conv(x * self.cp_tensor.weights.unsqueeze(1).unsqueeze(0))

        x_shape[1] = self.out_channels
        x = x.reshape(x_shape)

        return x


class CPDConvolution2D(nn.Module):
    def __init__(self, conv, R):
        super(CPDConvolution2D,self).__init__()
        weights, factors = parafac(conv.weight.detach(), init='svd', rank=R)
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


class UNConvModel(nn.Module):
    def __init__(self):
        super(UNConvModel, self).__init__()
        self.un_conv = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=7, stride=1, padding=1)
        )
        self.norm = nn.BatchNorm2d(64)
    
    def forward(self, x):
        x = self.un_conv(x)
        x = self.norm(x)
        return x

class TLConvModel(nn.Module):
    def __init__(self):
        super(TLConvModel, self).__init__()
        un_conv = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=7, stride=1, padding=1)
        )
        self.tlconv = nn.Sequential(
            tltorch.FactorizedConv.from_conv(un_conv[0], rank=100, factorization='cp', implementation='factorized'),
        )
        self.norm = nn.BatchNorm2d(64)
    
    def forward(self, x):
        x = self.tlconv(x)
        x = self.norm(x)
        return x


if __name__ == "__main__":
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    batches = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    times_un = []
    times_tl = []
    times_self = []

    flops_un_ls = []
    flops_cp_ls = []


    for i in tqdm.tqdm(batches):
        random_input = torch.randn(i, 1, 128, 128)


        # selfconv = CPDConvolution2D(un_conv, R=10)


        flops_un = FlopCountAnalysis(AutoEncoder11_UN(), random_input)
        print(flops_un.total())
        flops_un_ls.append(flops_un.total())

        flops_cp = FlopCountAnalysis(AutoEncoder11(R=100), random_input)
        print(flops_cp.total())
        flops_cp_ls.append(flops_cp.total())

        # time_start_un = time.time()
        # ret = un_conv(random_input)
        # time_end_un = time.time()

        # time_start_tlconv = time.time()
        # ret_tl = tlconv(random_input)
        # time_stop_tlconv = time.time()

        # time_start_self = time.time()
        # ret_self = selfconv(random_input)
        # time_stop_self = time.time()

        # elapsed_time_un = time_end_un - time_start_un
        # elapsed_time_tlconv = time_stop_tlconv - time_start_tlconv
        # elapsed_time_self = time_stop_self - time_start_self
        # print(f"Elapsed time for un_conv: {elapsed_time_un:.6f} seconds")
        # print(f"Elapsed time for tlconv: {elapsed_time_tlconv:.6f} seconds")
        # print(f"Elapsed time for selfconv: {elapsed_time_self:.6f} seconds")
        # times_un.append(elapsed_time_un)
        # times_tl.append(elapsed_time_tlconv)
        # times_self.append(elapsed_time_self)

    plt.figure(figsize=(10,9))
    plt.plot(batches, flops_un_ls, 'o-',label='uncompresssed')
    plt.plot(batches, flops_cp_ls,'o-', label='tensorly')
    # plt.plot(batches, times_self, 'o-',label='self')
    plt.xlabel('batch size')
    plt.ylabel('FLOPS')
    plt.yscale('log')
    plt.legend()
    plt.show()



    