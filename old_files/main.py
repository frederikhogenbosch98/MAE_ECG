import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import wfdb
from old_files.model import AutoEncoder
from torch.utils.data import Dataset, DataLoader
from old_files.train import train
from old_files.eval import eval


if __name__ == "__main__":

    dtype = torch.float
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    TRAIN = True
    SAVE_MODEL = False

    LEARNING_RATE = 0.001
    BATCH_SIZE = 64 #512
    NUM_EPOCHS = 5 #500
    RANDOM_SEED = 123

    torch.manual_seed(RANDOM_SEED)    

    # TRAINING
    model = train(LEARNING_RATE, BATCH_SIZE, NUM_EPOCHS, SAVE_MODEL, device, TRAIN)

    # EVALUATION
    eval(model, device)
    
