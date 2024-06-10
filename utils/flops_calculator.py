import torch
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis
import sys, os
from pprint import pprint
# from torch_flops import TorchFLOPsByFX

sys.path.append(os.path.abspath(os.path.join('..', 'models')))
sys.path.append(os.path.abspath(os.path.join('..')))

from print_funs import count_parameters
# Now you can import the models
from resnet50 import ResNet
from UNet import UNet
from convnext import ConvNext
from _11am_un import AutoEncoder11_UN
from _11am_back import AutoEncoder11

x = torch.randn(1, 1, 128, 128)


flops = FlopCountAnalysis(AutoEncoder11_UN(), x)
print(f"FLOPs: {flops.total()}")
current_pams = count_parameters(AutoEncoder11_UN())
print(f'num params: {current_pams}')

for r in [5, 10, 15, 20, 25, 35, 50, 75, 100, 125, 150, 175, 200]:

    flops = FlopCountAnalysis(AutoEncoder11(R=r), x)

    # print(f"FLOPs: {flops.total()}, R: {r}")
    current_pams = count_parameters(AutoEncoder11(R=r))
    print(f'num params: {current_pams}')
    # print(flops.by_module())
