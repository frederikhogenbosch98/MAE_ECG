import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import wfdb
from model import AutoEncoder
from torch.utils.data import Dataset, DataLoader
import glob
from create_dataset import ECGDataset

# def plot_ecg(i):
#     record = wfdb.rdrecord(f'{physio_root}/01/010/JS00{i:03}') 
#     wfdb.plot_wfdb(record=record, title=f'01 010 i')
    
    

def plot_results(model, test_tensor):
    output_tensor = model(test_tensor[0, :, :])
    selected_sample_channel = output_tensor[0,:]

    selected_sample_channel_np = selected_sample_channel.detach().numpy()

    time_points = range(test_tensor.size(dim=2))

    plt.plot(time_points, selected_sample_channel_np)
    plt.plot(time_points, test_tensor[0, 0, :])
    plt.title(f"Sample {0+1}, Channel {0+1}")
    plt.xlabel("Time Point")
    plt.ylabel("Amplitude")
    plt.show()


if __name__ == "__main__":

    LR_RATE = 0.001
    BATCH_SIZE = 32
    NUM_EPOCHS = 20
    RANDOM_SEED = 123

    torch.manual_seed(RANDOM_SEED)    

    dtype = torch.float
    device = torch.device("mps")  # mps for Apple Metal, torch.device("cuda") for Nvidia GPU's or otherwise torch.device("cpu") to use CPU

    dataset = torch.load('data/datasets/train_dataset.pt')
    print(dataset)

    test_tensor = torch.load('data/datasets/test_dataset.pt')


    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    print(dataloader)


    model = AutoEncoder()
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR_RATE)


    print(f'selected device: {device}')

    for epoch in range(NUM_EPOCHS):

        model.train()
        running_loss = 0.0
        
        for inputs, _ in dataloader:
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = loss_function(outputs, inputs)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(dataloader)
        print(f'Epoch {epoch+1}, Loss: {epoch_loss:.4f}')

    # plot_results(model, test_tensor)
