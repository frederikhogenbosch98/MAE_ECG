import os
import argparse
import numpy as np
import torch
import torch.nn as nn
# from utils import setup_seed
import matplotlib.pyplot as plt
import wfdb

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

    print(input_tensor.size())

    



if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--seed', type=int, default=42)


    # args = parser.parse_args()
    # setup_seed(args.seed)
    
    # plot_ecg()
    create_input_tensor()

