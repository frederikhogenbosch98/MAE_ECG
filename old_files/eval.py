import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from old_files.train import ECGDataset


def plot_results(model, test_tensor, i, lead=0):
    output_tensor = model(test_tensor[i:i+1, :, :, :])
    # print(output_tensor.shape)
    selected_sample_channel = output_tensor[0, :, :, lead]
    selected_sample_channel = selected_sample_channel.cpu()
    # print(selected_sample_channel.shape)
    selected_sample_channel_np = selected_sample_channel.detach().numpy()
    # print(selected_sample_channel_np[0])

    time_points = range(test_tensor.size(dim=2))
    test_tensor = test_tensor.cpu()
    # output_tensor = output_tensor.cpu()

    # print(test_tensor.shape)
    # print(selected_sample_channel_np.shape)
    torch.set_printoptions(threshold=10)
    # print(test_tensor)
    plt.plot(time_points, test_tensor[i, 0, :, lead], label='true')
    plt.plot(time_points, selected_sample_channel_np[0], label='predicted')
    # plt.plot(time_points, output_tensor[0, 0, :, lead], label='predicted')
    plt.title(f"Sample {0+1}, Channel {0+1}")
    plt.xlabel("Time Point")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.show()

def plot_img(model, tensor, i, lead):
    output_tensor = model(tensor[i:i+1, :, :, :])

    output_tensor = output_tensor.cpu()
    # print(tensor.device)
    tensor = tensor.cpu()
    output_tensor = output_tensor.detach().numpy()
    # print(tensor[i,lead,:,:].shape)
    # print(np.transpose(output_tensor[:,lead,:,:],(1,2,0)).shape)
    plt.imshow(tensor[i,lead,:,:])
    plt.imshow(np.transpose(output_tensor[:,lead,:,:],(1,2,0)))
    plt.show()

def eval(model, device):

    model.to(device)
    model.eval()

    test_tensor = torch.load('data/datasets/test_dataset_img.pt').to(device)
    # test_tensor = test_tensor.permute(0, 3, 2, 1)
    # test_tensor = test_tensor[:,None,:,:]
    added_layer = 255*torch.ones([test_tensor.shape[0], test_tensor.shape[1], 1, test_tensor.shape[3]], dtype=torch.float32)
    added_layer = added_layer.to(device)

    test_tensor = torch.cat((test_tensor, added_layer), dim=2)
    # torch.set_printoptions(threshold=10)
    # print(test_tensor)
    print(test_tensor.shape)

    dataset_test = ECGDataset(test_tensor)
    dataloader_test = DataLoader(dataset_test, batch_size=32, shuffle=True)

    mse_loss = nn.MSELoss(reduction='mean')
    total_loss = 0.0
    count = 0

    with torch.no_grad():  
        for inputs, _ in dataloader_test:
            inputs = inputs.to(device)  
            reconstructed = model(inputs)  
            loss = mse_loss(reconstructed, inputs)  
            total_loss += loss.item() 
            count += 1

    average_loss = np.round(total_loss / count, 6)

    print(f'Average MSE Loss on Test Set: {average_loss}')
    for i in range(100,110):
        plot_img(model, test_tensor, i, lead=0)
        # plot_results(model, test_tensor,i, lead=11)