import numpy as np
import torch
import torch.nn as nn
from model import AutoEncoder
from torch.utils.data import Dataset, DataLoader
import time
import random
import matplotlib.pyplot as plt

class ECGDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], 0

def print_epoch_info(epoch, loss,  t_int, mem):
    print(f'epoch {epoch+1}: Loss: {loss:.4f}, Duration: {np.round(t_int, 2)}s, memory usage: {mem}')


def masking(tensor, ratio):
    N = tensor.size(dim=2)
    N_r = int(ratio*N)
    zero_idxs = random.sample(range(N),N_r)
    mask = np.ones(N, dtype=bool)
    for i in zero_idxs:
        mask[i] = False
    return mask


def train(LEARNING_RATE, BATCH_SIZE, NUM_EPOCHS, SAVE_MODEL, device, TRAIN):


    train_tensor = torch.load('data/datasets/train_dataset.pt').to(device)
    train_tensor = train_tensor.permute(0, 3, 2, 1)

    dataset_train = ECGDataset(train_tensor)
    dataloader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)

    model = AutoEncoder(in_chans=12, dims=[24, 48], depths=[1, 1], decoder_embed_dim=512)


    # N, C, H, W  -> 512 batch, 12 leads, 300 samples, 6 cycles
    print(train_tensor.size())

    if TRAIN:
        loss_function = nn.MSELoss()
        # optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)


        # move everything to mps device
        print(f'moving model and dataset to device: {device}')
        model = model.to(device)

        is_on_mps = next(model.parameters()).device.type == 'mps'
        print(f"Model is on MPS: {is_on_mps}")  
        # dataset = dataset.move_device(device)

        # print(dataset.get_device())

        losses = []
        # TRAINING
        print("STARTING TRAINING")
        for epoch in range(NUM_EPOCHS):
            t_start = time.time()

            model.train()
            running_loss = 0.0
            
            for idx, (inputs, _) in enumerate(dataloader_train):

                inputs = inputs.to(device)
                mask = masking(train_tensor, 1) # 0 is fully masked
                # inputs = inputs.masked_fill(torch.from_numpy(mask).to(device) == True, 0)
                optimizer.zero_grad()

                outputs = model(inputs)
                loss = loss_function(outputs, inputs)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
            t_end = time.time()

            epoch_loss = running_loss / len(dataloader_train)
            losses.append(epoch_loss)
            print_epoch_info(epoch, epoch_loss, t_end-t_start, torch.mps.current_allocated_memory())

        

        if SAVE_MODEL:
            save_folder = 'data/models/MAE.pth'
            torch.save(model.state_dict(), save_folder)
            print(f'model saved to {save_folder}')

        plt.plot(np.arange(NUM_EPOCHS), losses)
        plt.title('loss function')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.show()

    else:
        model.load_state_dict(torch.load('data/models/MAE.pth'))

    return model