import torch
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis
import sys, os
sys.path.append(os.path.abspath(os.path.join('..', 'models')))

# Now you can import the models
from resnet50 import ResNet
from UNet import UNet
from convnext import ConvNext
from _11am_un import AutoEncoder11_UN

input_tensor = torch.randn(256, 1, 128, 128)

# Initialize FlopCountAnalysis with the model and the input tensor
flops = FlopCountAnalysis(ConvNext(), input_tensor)

# Print the total FLOPs
print(f"FLOPs: {flops.total()}")