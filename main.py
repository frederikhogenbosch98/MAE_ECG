import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import wfdb
from model import AutoEncoder
from torch.utils.data import Dataset, DataLoader
from create_dataset import ECGDataset
import time

# def plot_ecg(i):
#     record = wfdb.rdrecord(f'{physio_root}/01/010/JS00{i:03}') 
#     wfdb.plot_wfdb(record=record, title=f'01 010 i')
    
    

def plot_results(model, test_tensor):
    output_tensor = model(test_tensor[0, :, :])
    selected_sample_channel = output_tensor[0,:]
    selected_sample_channel = selected_sample_channel.cpu()
    selected_sample_channel_np = selected_sample_channel.detach().numpy()

    time_points = range(test_tensor.size(dim=2))
    test_tensor = test_tensor.cpu()

    plt.plot(time_points, selected_sample_channel_np)
    plt.plot(time_points, test_tensor[0, 0, :])
    plt.title(f"Sample {0+1}, Channel {0+1}")
    plt.xlabel("Time Point")
    plt.ylabel("Amplitude")
    plt.show()



def print_epoch_info(epoch, loss, t_int):
    print(f'epoch {epoch+1}, Loss: {loss:.4f}, Duration: {np.round(t_int, 2)}s')


if __name__ == "__main__":

    LR_RATE = 0.001
    BATCH_SIZE = 32
    NUM_EPOCHS = 10
    RANDOM_SEED = 123

    torch.manual_seed(RANDOM_SEED)    

    dtype = torch.float
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu") 

    dataset = torch.load('data/datasets/train_dataset.pt')
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


    test_tensor = torch.load('data/datasets/test_dataset.pt').to(device)

    model = AutoEncoder()
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR_RATE)


    # move everything to mps device
    print(f'moving model and dataset to device: {device}')
    model = model.to(device)

    is_on_mps = next(model.parameters()).device.type == 'mps'
    print(f"Model is on MPS: {is_on_mps}")  
    # dataset = dataset.move_device(device)

    # print(dataset.get_device())

    for epoch in range(NUM_EPOCHS):
        t_start = time.time()

        model.train()
        running_loss = 0.0
        
        for inputs, _ in dataloader:

            inputs = inputs.to(device)
            inputs.get_device()
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = loss_function(outputs, inputs)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        t_end = time.time()
        epoch_loss = running_loss / len(dataloader)
        print_epoch_info(epoch, epoch_loss, t_end-t_start)


    plot_results(model, test_tensor)
