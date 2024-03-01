import wfdb
import torch
import glob
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path

physio_root = 'data/physionet/files/ecg-arrhythmia/1.0.0/WFDBRecords'


class ECGDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], 0


def create_input_tensor():
    print(f'creating datasets')

    init_sample, init_field = wfdb.rdsamp(f'{physio_root}/01/010/JS00001')
    input_tensor = torch.Tensor(np.array(init_sample)).unsqueeze(0)

    directory_path = Path(f'{physio_root}')

    mat_files = [file for file in directory_path.rglob('*.mat')]

    mat_files_without_extension = [str(file)[:-4] for file in mat_files]
    mat_files_without_extension = mat_files_without_extension[1:]

    for idx, file in enumerate(mat_files_without_extension):
        if idx % 1000 == 0:
            print(f'at iteration: {idx}')
        sample_values, sample_field = wfdb.rdsamp(f'{file}')
        sample_tensor = torch.Tensor(np.array(sample_values)).unsqueeze(0)
        input_tensor = torch.cat([input_tensor, sample_tensor], dim=0)

    print(input_tensor.size())
    return input_tensor



def train_test_split(tensor, split):
    split_idx = int(tensor.size(dim=0)*split)
    train_tensor = tensor[1:split_idx,:,:]
    test_tensor = tensor[split_idx+1:-1,:,:]

    return train_tensor, test_tensor

if __name__ == "__main__":
    input_tensor = create_input_tensor().permute(0,2,1)
    input_tensor = input_tensor[:, :, 0:4992]

    train_tensor, test_tensor = train_test_split(input_tensor, 0.7)

    print(train_tensor.size())
    print(test_tensor.size())

    train_dataset = ECGDataset(train_tensor)
    # test_dataset = ECGDataset(test_tensor)

    torch.save(train_dataset, 'data/datasets/train_dataset.pt')
    torch.save(test_tensor, 'data/datasets/test_dataset.pt')