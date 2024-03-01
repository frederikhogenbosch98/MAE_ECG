import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import wfdb
from model import AutoEncoder
from torch.utils.data import Dataset, DataLoader
import time
import random


class ECGDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], 0

def plot_results(model, test_tensor, losses, NUM_EPOCHS):
    output_tensor = model(test_tensor[8, :, :])
    selected_sample_channel = output_tensor[8,:]
    selected_sample_channel = selected_sample_channel.cpu()
    selected_sample_channel_np = selected_sample_channel.detach().numpy()

    time_points = range(test_tensor.size(dim=2))
    test_tensor = test_tensor.cpu()

    plt.plot(time_points, selected_sample_channel_np, label='predicted')
    plt.plot(time_points, test_tensor[8, 8, :], label='true')
    plt.title(f"Sample {0+1}, Channel {0+1}")
    plt.xlabel("Time Point")
    plt.ylabel("Amplitude")
    plt.show()

    plt.plot(np.arange(NUM_EPOCHS), losses)
    plt.title('loss function')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()


def masking(tensor, ratio):
    N = tensor.size(dim=2)
    N_r = int(ratio*N)
    zero_idxs = random.sample(range(N),N_r)
    print(zero_idxs)
    mask = np.ones(N, dtype=bool)
    for i in zero_idxs:
        mask[i] = False
    return mask



def print_epoch_info(epoch, loss,  t_int):
    print(f'epoch {epoch+1}: Loss: {loss:.4f}, Duration: {np.round(t_int, 2)}s')


if __name__ == "__main__":

    LR_RATE = 0.001
    BATCH_SIZE = 32
    NUM_EPOCHS = 2
    RANDOM_SEED = 123

    torch.manual_seed(RANDOM_SEED)    

    dtype = torch.float
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu") 

    train_tensor = torch.load('data/datasets/train_dataset.pt').to(device)
    test_tensor = torch.load('data/datasets/test_dataset.pt').to(device)

    mask = masking(train_tensor, 0.5)

    mt = torch.masked_tensor(train_tensor, mask)

    dataset = ECGDataset(mt)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

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

    losses = []
    # TRAINING
    for epoch in range(NUM_EPOCHS):
        t_start = time.time()

        model.train()
        running_loss = 0.0
        
        for idx, (inputs, _) in enumerate(dataloader):

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
        losses.append(epoch_loss)
        print_epoch_info(epoch, epoch_loss, t_end-t_start)


    # TESTING
    

    plot_results(model, test_tensor, losses, NUM_EPOCHS)
