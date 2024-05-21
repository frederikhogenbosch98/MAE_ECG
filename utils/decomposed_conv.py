import tltorch
import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tensorly.decomposition import parafac
import time
import matplotlib.pyplot as plt
import tqdm

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


if __name__ == "__main__":
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    batches = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    times_un = []
    times_tl = []
    times_self = []


    for i in tqdm.tqdm(batches):
        random_input = torch.randn(i, 64, 112, 112)

        un_conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=7, stride=1, padding=1)
        tlconv = tltorch.FactorizedConv.from_conv(un_conv, rank=10, factorization='cp')
        selfconv = CPDConvolution2D(un_conv, R=10)


        time_start_un = time.time()
        ret = un_conv(random_input)
        time_end_un = time.time()

        time_start_tlconv = time.time()
        ret_tl = tlconv(random_input)
        time_stop_tlconv = time.time()

        time_start_self = time.time()
        ret_self = selfconv(random_input)
        time_stop_self = time.time()

        elapsed_time_un = time_end_un - time_start_un
        elapsed_time_tlconv = time_stop_tlconv - time_start_tlconv
        elapsed_time_self = time_stop_self - time_start_self
        # print(f"Elapsed time for un_conv: {elapsed_time_un:.6f} seconds")
        # print(f"Elapsed time for tlconv: {elapsed_time_tlconv:.6f} seconds")
        # print(f"Elapsed time for selfconv: {elapsed_time_self:.6f} seconds")
        times_un.append(elapsed_time_un)
        times_tl.append(elapsed_time_tlconv)
        times_self.append(elapsed_time_self)
    
    plt.plot(batches, times_un, 'o-',label='uncompresssed')
    plt.plot(batches, times_tl,'o-', label='tensorly')
    plt.plot(batches, times_self, 'o-',label='self')
    plt.xlabel('batch size')
    plt.ylabel('seconds')
    plt.yscale('log')
    plt.legend()
    plt.show()



    