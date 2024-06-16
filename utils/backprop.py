import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from tensorly.decomposition import parafac

class CPDConvolution2DFunction(Function):
    @staticmethod
    def forward(ctx, x, s_to_r_weight, depth_vert_weight, depth_hor_weight, r_to_t_weight, r_to_t_bias, stride, padding):
        ctx.save_for_backward(x, s_to_r_weight, depth_vert_weight, depth_hor_weight, r_to_t_weight, r_to_t_bias)
        ctx.stride = stride
        ctx.padding = padding

        x = F.conv2d(x, s_to_r_weight, stride=1, padding=0)
        x = F.conv2d(x, depth_vert_weight, groups=depth_vert_weight.shape[0], stride=(stride[0], 1), padding=(padding[0], 0))
        x = F.conv2d(x, depth_hor_weight, groups=depth_hor_weight.shape[0], stride=(1, stride[1]), padding=(0, padding[1]))
        x = F.conv2d(x, r_to_t_weight, bias=r_to_t_bias, stride=1, padding=0)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        x, s_to_r_weight, depth_vert_weight, depth_hor_weight, r_to_t_weight, r_to_t_bias = ctx.saved_tensors
        stride = ctx.stride
        padding = ctx.padding

        # Gradients with respect to r_to_t_weight and r_to_t_bias
        grad_r_to_t_weight = F.conv2d(grad_output.permute(1, 0, 2, 3), x.permute(1, 0, 2, 3), bias=None, stride=1, padding=0).permute(1, 0, 2, 3)
        grad_r_to_t_bias = grad_output.sum((0, 2, 3))

        # Gradients with respect to depth_hor_weight
        grad_depth_hor_input = F.conv2d(grad_output, r_to_t_weight.permute(1, 0, 2, 3), stride=1, padding=0)
        grad_depth_hor_weight = F.conv2d(grad_depth_hor_input.permute(1, 0, 2, 3), x.permute(1, 0, 2, 3), stride=1, padding=0).permute(1, 0, 2, 3)

        # Gradients with respect to depth_vert_weight
        grad_depth_vert_input = F.conv2d(grad_depth_hor_input, depth_hor_weight.permute(1, 0, 2, 3), groups=depth_hor_weight.shape[0], stride=(stride[0], 1), padding=(padding[0], 0))
        grad_depth_vert_weight = F.conv2d(grad_depth_vert_input.permute(1, 0, 2, 3), x.permute(1, 0, 2, 3), stride=1, padding=0).permute(1, 0, 2, 3)

        # Gradients with respect to s_to_r_weight
        grad_s_to_r_input = F.conv2d(grad_depth_vert_input, depth_vert_weight.permute(1, 0, 2, 3), groups=depth_vert_weight.shape[0], stride=1, padding=0)
        grad_s_to_r_weight = F.conv2d(grad_s_to_r_input.permute(1, 0, 2, 3), x.permute(1, 0, 2, 3), stride=1, padding=0).permute(1, 0, 2, 3)

        # Gradients with respect to input x
        grad_x = F.conv2d(grad_s_to_r_input, s_to_r_weight.permute(1, 0, 2, 3), stride=1, padding=0)

        return grad_x, grad_s_to_r_weight, grad_depth_vert_weight, grad_depth_hor_weight, grad_r_to_t_weight, grad_r_to_t_bias, None, None

class CPDConvolution2D(nn.Module):
    def __init__(self, conv, R):
        super(CPDConvolution2D, self).__init__()
        conv_weight = conv.weight.detach().cpu().numpy()
        weights, factors = parafac(conv_weight, rank=R, init='svd')
        
        self.s_to_r = nn.Parameter(torch.tensor(factors[1].astype(float)).permute(1, 0).unsqueeze(-1).unsqueeze(-1).to(conv.weight.device, dtype=torch.float))
        self.depth_vert = nn.Parameter(torch.tensor(factors[2].astype(float)).permute(1, 0).unsqueeze(1).unsqueeze(-1).to(conv.weight.device, dtype=torch.float))
        self.depth_hor = nn.Parameter(torch.tensor(factors[3].astype(float)).permute(1, 0).unsqueeze(1).unsqueeze(1).to(conv.weight.device, dtype=torch.float))
        self.r_to_t = nn.Parameter(torch.tensor(factors[0].astype(float)).unsqueeze(-1).unsqueeze(-1).to(conv.weight.device, dtype=torch.float))

        self.bias = nn.Parameter(conv.bias.detach()) if conv.bias is not None else None
        self.stride = conv.stride
        self.padding = conv.padding

    def forward(self, x):
        return CPDConvolution2DFunction.apply(x, self.s_to_r, self.depth_vert, self.depth_hor, self.r_to_t, self.bias, self.stride, self.padding)

# Test the implementation
conv = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
R = 4  # Rank for CP decomposition
cpd_conv = CPDConvolution2D(conv, R)

# Create a random input tensor
input_tensor = torch.randn(1, 3, 32, 32, requires_grad=True)

# Forward pass
output = cpd_conv(input_tensor)

# Dummy loss for backpropagation
loss = output.sum()

# Backward pass to calculate gradients
loss.backward()

# Access gradients
grad_input = input_tensor.grad
print(f"Gradient with respect to the input: {grad_input}")
