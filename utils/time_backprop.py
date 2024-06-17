import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
from tensorly.decomposition import parafac
import time

S = 64
T = 64
D = 7
R = np.arange(1,50)


class CPDConvolution2D(nn.Module):
    def __init__(self, conv, R):
        super(CPDConvolution2D,self).__init__()
        conv_weight = conv.weight.detach()
        conv_weight_numpy = conv_weight.cpu().numpy()
        weights, factors = parafac(conv_weight_numpy, rank=R, init='random')
        factors = [torch.tensor(f).float() for f in factors]
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
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=7, stride=1, padding=3)
        )
    
    def forward(self, x):
        return self.un_conv(x)


D = 7
def  FLOPs_sr(R, S, H, W):
    return 4 * R * S * H * W
def  FLOPs_w(R, H, W):
    return  4 * D * H * W * R**2
def  FLOPs_h(R, H, W):
    return  4 * D * H * W * R**2
def  FLOPs_rt(R, H, W, T):
    return  4 * R * T * H * W
def FLOPs_og(S, T, H, W):
    return 4 * S * D**2 * T * H * W


T = 64
S = 64

H = 32
W = 32

criterion = nn.MSELoss()


conv = nn.Conv2d(64, 64, kernel_size=7, stride=1, padding=3)
un_conv = UNConvModel()

# Create a random input tensor
input_tensor = torch.randn(1, 64, 32, 32, requires_grad=True)

# Forward pass
output = un_conv(input_tensor)

# Dummy loss for backpropagation
loss = criterion(output, input_tensor)

# Backward pass to calculate gradients
start_time = time.time()
loss.backward()
end_time = time.time()

print(f"Uncompressed backprop timing: {end_time - start_time}")

print(f'FLOPGS OG: {FLOPs_og(S, T, H, W)}')
og_flops = FLOPs_og(S, T, H, W)
og_time = end_time - start_time

times = []
flops = []

for r in R:
    crit = nn.MSELoss()
    cpd_conv = CPDConvolution2D(conv, r)
    output = cpd_conv(input_tensor)
    loss = crit(output, input_tensor)
    start_time_cp = time.time()
    loss.backward()
    end_time_cp = time.time()
    print(f"CPD {r} backprop timing: {end_time_cp - start_time_cp}")
    times.append(end_time_cp-start_time_cp)



    print(f'FLOPGS OG for R: {r}: {FLOPs_sr(r, S, H, W) + FLOPs_w(r, H, W)+ FLOPs_h(r, H, W)+ FLOPs_rt(r, T, H, W)}')
    flops.append(FLOPs_sr(r, S, H, W) + FLOPs_w(r, H, W)+ FLOPs_h(r, H, W)+ FLOPs_rt(r, T, H, W))

flops_plot = [i/og_flops for i in flops]
times_plot = [i/og_time for i in times]

plt.plot(np.arange(len(R)), flops_plot, label='flops')
plt.plot(np.arange(len(R)), times_plot, label='time')
plt.show()