import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import wfdb
from model import AutoEncoder
from torch.utils.data import Dataset, DataLoader
from train import train
from eval import eval


if __name__ == "__main__":

    dtype = torch.float
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    TRAIN = True
    SAVE_MODEL = True

    LEARNING_RATE = 0.001
    BATCH_SIZE = 32
    NUM_EPOCHS = 40
    RANDOM_SEED = 123

    torch.manual_seed(RANDOM_SEED)    

    # TRAINING
    model = train(LEARNING_RATE, BATCH_SIZE, NUM_EPOCHS, SAVE_MODEL, device, TRAIN)

    # EVALUATION
    eval(model, device)
    
