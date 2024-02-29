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


def plot_ecg():
    record = wfdb.rdrecord(f'{physio_root}/01/010/JS00001') 
    wfdb.plot_wfdb(record=record, title='01 010')
    

def read_ecg():
    signals, fields = wfdb.rdsamp(f'{physio_root}/01/010/JS00001')
    input_tensor = torch.Tensor(signals)
    print(input_tensor.size())
    print(signals)

    



if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--seed', type=int, default=42)


    # args = parser.parse_args()
    # setup_seed(args.seed)
    
    # plot_ecg()
    read_ecg()

