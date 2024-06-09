import torch
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis
import sys, os
from pprint import pprint
# from torch_flops import TorchFLOPsByFX

sys.path.append(os.path.abspath(os.path.join('..', 'models')))

# Now you can import the models
from resnet50 import ResNet
from UNet import UNet
from convnext import ConvNext
from _11am_un import AutoEncoder11_UN

x = torch.randn(1, 1, 128, 128)


flops = FlopCountAnalysis(ConvNext(), x)

print(f"FLOPs: {flops.total()}")
print(flops.by_module())
