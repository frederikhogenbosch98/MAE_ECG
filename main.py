import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import wfdb
from model import AutoEncoder

physio_root = 'data/physionet/files/ecg-arrhythmia/1.0.0/WFDBRecords'

dtype = torch.float
device = torch.device("mps")  # mps for Apple Metal, torch.device("cuda") for Nvidia GPU's or otherwise torch.device("cpu") to use CPU


def plot_ecg(i):
    record = wfdb.rdrecord(f'{physio_root}/01/010/JS00{i:03}') 
    wfdb.plot_wfdb(record=record, title=f'01 010 i')
    

def create_input_tensor():
    print(f'creating input tensor')
    init_sample, init_field = wfdb.rdsamp(f'{physio_root}/01/010/JS00001')
    input_tensor = torch.Tensor(np.array(init_sample)).unsqueeze(0)
    for i in range(2, 104):
        try: 
            sample_values, sample_field = wfdb.rdsamp(f'{physio_root}/01/010/JS00{i:03}')
            sample_tensor = torch.Tensor(np.array(sample_values)).unsqueeze(0)
            input_tensor = torch.cat([input_tensor, sample_tensor], dim=0)

        except FileNotFoundError:
            print(f"FileNotFoundError for sample {i}, skipping...")
            continue

    return input_tensor

    


if __name__ == "__main__":

    # input_tensor = create_input_tensor()

    init_sample, init_field = wfdb.rdsamp(f'{physio_root}/01/010/JS00001')
    input_tensor = torch.Tensor(np.array(init_sample)).unsqueeze(0)
    input_tensor = input_tensor.permute(0,2,1)

    # print(input_tensor.size())

    model = AutoEncoder()
    print(model)
    output_tensor = model(input_tensor)
    print(output_tensor.size())

    # selected_sample_channel = output_tensor[0, 0, :]

    # selected_sample_channel_np = selected_sample_channel.detach().numpy()

    # time_points = range(len(selected_sample_channel_np))

    # plt.plot(time_points, selected_sample_channel_np)
    # plt.title(f"Sample {0+1}, Channel {0+1}")
    # plt.xlabel("Time Point")
    # plt.ylabel("Amplitude")
    # plt.show()

