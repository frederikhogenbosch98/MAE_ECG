
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from train import ECGDataset


def plot_results(model, test_tensor):
    output_tensor = model(test_tensor[27:29, :, :, :])
    selected_sample_channel = output_tensor[1,  11, :, 0]
    selected_sample_channel = selected_sample_channel.cpu()
    print(selected_sample_channel.shape)
    selected_sample_channel_np = selected_sample_channel.detach().numpy()

    time_points = range(test_tensor.size(dim=2))
    test_tensor = test_tensor.cpu()

    torch.set_printoptions(threshold=10)
    print(test_tensor)
    plt.plot(time_points, selected_sample_channel_np, label='predicted')
    plt.plot(time_points, test_tensor[28,11, :, 0], label='true')
    plt.title(f"Sample {0+1}, Channel {0+1}")
    plt.xlabel("Time Point")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.show()


def eval(model, device):

    model.to(device)
    model.eval()

    test_tensor = torch.load('data/datasets/test_dataset.pt').to(device)
    test_tensor = test_tensor.permute(0, 3, 2, 1)
    # torch.set_printoptions(threshold=10)
    # print(test_tensor)
    print(test_tensor.shape)

    # dataset_test = ECGDataset(test_tensor)
    # dataloader_test = DataLoader(dataset_test, batch_size=32, shuffle=True)

    # mse_loss = nn.MSELoss(reduction='mean')
    # total_loss = 0.0
    # count = 0

    # with torch.no_grad():  
    #     for inputs, _ in dataloader_test:
    #         inputs = inputs.to(device)  
    #         reconstructed = model(inputs)  
    #         loss = mse_loss(reconstructed, inputs)  
    #         total_loss += loss.item() 
    #         count += 1

    # average_loss = np.round(total_loss / count, 6)

    # print(f'Average MSE Loss on Test Set: {average_loss}')

    plot_results(model, test_tensor)