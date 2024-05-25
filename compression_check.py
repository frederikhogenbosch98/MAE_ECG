import torch
from torchvision import datasets, transforms
from models.model_56x56_TD import AutoEncoder56_TD, Classifier56_TD
from models.model_56x56 import AutoEncoder56, Classifier56
import torch.nn as nn
from models.convnext import ConvNext
from models.UNet import UNet
from models.UNet_TD import UNet_TD
import numpy as np

if __name__ == '__main__':

    num_params_uncompressed = 3122977

    rank = 100
    # model = UNet_TD(R=rank, factorization='cp')

    num_params_TD = 492625 # count_parameters(mae)

    print(f'compression ratio for rank {rank}: {np.round(num_params_uncompressed/num_params_TD,2)}x')