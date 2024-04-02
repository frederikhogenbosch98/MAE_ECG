import wfdb
import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
from scipy.signal import butter, lfilter
import pandas as pd

physio_root = 'data/physionet/files/ecg-arrhythmia/1.0.0/WFDBRecords'


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band', analog=False)
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def create_input_tensor():
    print(f'creating datasets')
    
    low_cut = 1
    high_cut = 20
    fs = 500

    init_sample, init_field = wfdb.rdsamp(f'{physio_root}/01/010/JS00001')
    init_filtered = butter_bandpass_filter(np.array(init_sample), low_cut, high_cut, fs, order=5)
    input_tensor = torch.Tensor(init_filtered).unsqueeze(0)

    directory_path = Path(f'{physio_root}')

    mat_files = [file for file in directory_path.rglob('*.mat')]

    mat_files_without_extension = [str(file)[:-4] for file in mat_files]
    mat_files_without_extension = mat_files_without_extension[1:]

    for idx, file in enumerate(mat_files_without_extension):
        if idx % 1000 == 0:
            print(f'processing folder {(idx//1000)+1}/10')
        sample_values, sample_field = wfdb.rdsamp(f'{file}')
        sample_values = np.array(sample_values)
        filtered_data = np.zeros((sample_values.shape[0], sample_values.shape[1]))
        for l in range(sample_values.shape[1]):
            filtered_data[:,l] = butter_bandpass_filter(sample_values[:,l], low_cut, high_cut, fs, order=5)

        sample_tensor = torch.Tensor(filtered_data).unsqueeze(0)
        input_tensor = torch.cat([input_tensor, sample_tensor], dim=0)

    print(input_tensor.size())
    return input_tensor


def create_label_tensors():

    Y = pd.read_csv(path+'ptbxl_database.csv', index_col='ecg_id')
    Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

    return label_tensor


def train_test_split(tensor, split):
    split_idx = int(tensor.size(dim=0)*split)
    train_tensor = tensor[1:split_idx,:,:]
    test_tensor = tensor[split_idx+1:-1,:,:]

    return train_tensor, test_tensor

if __name__ == "__main__":
    input_tensor = create_input_tensor().permute(0,2,1)
    input_tensor = input_tensor[:, :, 0:4992]

    # shuffle tensors with some sort of seed

    train_tensor, test_tensor = train_test_split(input_tensor, 0.7)

    print(train_tensor.size())
    print(test_tensor.size())

    # train_dataset = ECGDataset(train_tensor)
    # test_dataset = ECGDataset(test_tensor)

    torch.save(train_tensor, 'data/datasets/train_dataset.pt')
    torch.save(test_tensor, 'data/datasets/test_dataset.pt')