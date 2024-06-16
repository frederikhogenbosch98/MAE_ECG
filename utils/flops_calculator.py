import torch
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis, flop_count
import sys, os
from pprint import pprint
# from torch_flops import TorchFLOPsByFX
from tabulate import tabulate
from collections import Counter
from decomposed_conv import CPDConvolution2D, UNConvModel, TensorlyConv, TLConvModel

sys.path.append(os.path.abspath(os.path.join('..', 'models')))
sys.path.append(os.path.abspath(os.path.join('..')))

from print_funs import count_parameters
# Now you can import the models
from resnet50 import ResNet
from UNet import UNet
from convnext import ConvNext
from _11am_un import AutoEncoder11_UN
# from _11am_back import AutoEncoder11
from _11am_corr import AutoEncoder11


def get_flops(model, x, criterion, backward=False):
    flops = FlopCountAnalysis(model, x)
    forward_flops = flops.total()

    if backward:
        # Register hooks to count FLOPs for backward pass
        def conv_hook(module, grad_input, grad_output):
            if isinstance(module, nn.Conv2d):
                flops_count = flop_count(module, (x,))
                return flops_count

        handles = []
        for layer in model.modules():
            if isinstance(layer, nn.Conv2d):
                handle = layer.register_backward_hook(conv_hook)
                handles.append(handle)

        output = model(x)
        target = torch.randn_like(output)
        loss = criterion(output, target)
        loss.backward()

        # Calculate backward FLOPs
        backward_flops = sum(handle.backward_flops for handle in handles)

        # Remove hooks
        for handle in handles:
            handle.remove()

        return forward_flops, backward_flops
    else:
        return forward_flops


x = torch.randn(1, 1, 128, 128)
num_params_uncompressed = 9411649

# flops = FlopCountAnalysis(AutoEncoder11_UN(), x)
# print(f"FLOPs: {flops.total()}")
# current_pams = count_parameters(AutoEncoder11_UN())
# print(f'num params: {current_pams}')

criterion = nn.MSELoss()

for r in [5, 10, 15, 20, 25, 35, 50, 75, 100, 125, 150, 175, 200]:

    # conv = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=1, padding=1) 
    # if r == 0:
    #    flops = FlopCountAnalysis(AutoEncoder11_UN(), x) 
    # #    flops = FlopCountAnalysis(UNConvModel(), x) 
    # else:
    #     flops = FlopCountAnalysis(AutoEncoder11(R=r), x)
    #     flops = FlopCountAnalysis(CPDConvolution2D(conv=conv,R=r), x)

    if r == 0:
        model = AutoEncoder11_UN()
    else:
        model = AutoEncoder11(R=r)

    forward_flops, backward_flops = get_flops(model, x, criterion, backward=True)

    print(r)


    
    print(f"R: {r}")
    print(f"Forward FLOPs: {forward_flops}")
    print(f"Backward FLOPs: {backward_flops}")

    current_pams = count_parameters(model)
    print(f'Number of parameters: {current_pams}')
    comp_ratio = num_params_uncompressed / current_pams
    print(f'Compression ratio: {comp_ratio}')

    flops_by_operator = forward_flops.by_operator()
    flops_by_module = forward_flops.by_module()
    flops_by_module_and_operator = forward_flops.by_module_and_operator()

    # Pretty print the total FLOPs
    print("Total FLOPs:", forward_flops + backward_flops)

    # Pretty print FLOPs by operator
    print("\nFLOPs by Operator:")
    print(tabulate(flops_by_operator.items(), headers=["Operator", "FLOPs"], tablefmt="pretty")) 

    # print(f"FLOPs: {flops.total()}, R: {r}")
    # current_pams = count_parameters(AutoEncoder11(R=r))
    # print(f'num params: {current_pams}')
    # comp_ratio = num_params_uncompressed/current_pams
    # print(f'compression ratio: {comp_ratio}')
    # # print(flops.by_module())
    # # Get the results
    # total_flops = flops.total()
    # flops_by_operator = flops.by_operator()
    # flops_by_module = flops.by_module()
    # flops_by_module_and_operator = flops.by_module_and_operator()

    # # Pretty print the total FLOPs
    # print("Total FLOPs:", total_flops)

    # # Pretty print FLOPs by operator
    # print("\nFLOPs by Operator:")
    # print(tabulate(flops_by_operator.items(), headers=["Operator", "FLOPs"], tablefmt="pretty"))

    # Pretty print FLOPs by module
    # print("\nFLOPs by Module:")
    # print(tabulate(flops_by_module.items(), headers=["Module", "FLOPs"], tablefmt="pretty"))

