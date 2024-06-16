import torch
import torch.nn as nn
from tensorly.decomposition import parafac
import tensorly as tl
import torchviz

tl.set_backend('pytorch')  # Ensure TensorLy is using PyTorch backend

class CPDConvolution2D(nn.Module):
    def __init__(self, conv, R):
        super(CPDConvolution2D, self).__init__()
        conv_weight = conv.weight.detach()
        weights, factors = parafac(conv_weight, rank=R, init='svd')
        
        self.s_to_r = nn.Conv2d(conv.in_channels, R, (1, 1), stride=1, padding=0, bias=False)
        self.s_to_r.weight.data = factors[1].permute(1, 0).unsqueeze(-1).unsqueeze(-1)

        self.depth_vert = nn.Conv2d(R, R, (factors[2].shape[0], 1), groups=R, stride=(conv.stride[0], 1), padding=(conv.padding[0], 0), bias=False)
        self.depth_vert.weight.data = factors[2].permute(1, 0).unsqueeze(1).unsqueeze(-1)

        self.depth_hor = nn.Conv2d(R, R, (1, factors[3].shape[0]), groups=R, stride=(1, conv.stride[1]), padding=(0, conv.padding[1]), bias=False)
        self.depth_hor.weight.data = factors[3].permute(1, 0).unsqueeze(1).unsqueeze(1)

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
    def __init__(self, conv):
        super(UNConvModel, self).__init__()
        self.un_conv = conv
    def forward(self, x):
        x = self.un_conv(x)
        return x

# Create dummy input and target
input_tensor = torch.randn(1, 64, 32, 32)  # Batch size of 1, 64 channels, 32x32 image
target_tensor = torch.randn(1, 64, 32, 32)

# Instantiate the models
conv_layer = nn.Conv2d(64, 64, kernel_size=7, stride=1, padding=3)
cpd_model = CPDConvolution2D(conv_layer, R=10)
unconv_model = UNConvModel(conv=conv_layer)

# Define a simple loss function
criterion = nn.MSELoss()

# Forward pass
cpd_output = cpd_model(input_tensor)
unconv_output = unconv_model(input_tensor)

# # Calculate loss
# cpd_loss = criterion(cpd_output, target_tensor)
# unconv_loss = criterion(unconv_output, target_tensor)

# # Backward pass
# cpd_loss.backward(retain_graph=True)
# unconv_loss.backward(retain_graph=True)

# Visualize the computational graphs including the backward pass
cpd_graph = torchviz.make_dot(cpd_output, params=dict(cpd_model.named_parameters()))
cpd_graph.render("CPDConvolution2D_backward", format="png")

unconv_graph = torchviz.make_dot(unconv_output, params=dict(unconv_model.named_parameters()))
unconv_graph.render("UNConvModel_backward", format="png")
