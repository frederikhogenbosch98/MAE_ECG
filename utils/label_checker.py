import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from pathlib import Path
import os, time
import wfdb
from scipy.signal import butter, lfilter
import numpy as np
from scipy.signal import resample, find_peaks

def plot_single_img(img, i):
    plt.imshow(img[i, :, :, :].permute(1,2,0).detach().numpy(),cmap="gray")
    plt.title(f'index: {i}')
    plt.show()


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


def normalize(segment):
    segment_min = np.min(segment)
    segment_max = np.max(segment)
    return (segment - segment_min) / (segment_max - segment_min)


def get_r_idx(data):
    r_idx, _ = find_peaks(data, distance=250) 
    return r_idx

def extract_segments(data, r_idx):
    segments = []
    for idx in r_idx:
        start = max(idx-100, 0)
        end = min(idx+200, len(data))
        segment = list(data[start:end])
        segments.append(segment)
        
    return segments


def averaging(segments):
    averaged_signal = []
    seg = 0
    for j in range(len(segments[seg])):
        mean_vec = []
        for i in range(len(segments)):
            mean_vec.append(segments[i][j])
        
        averaged_signal.append(np.average(mean_vec))

    return averaged_signal

class SupervisedDataset(torch.utils.data.Dataset):
    def __init__(self, data_path_un, data_path_sup, resize_shape=(56,56)):
        loaded_data = torch.load(data_path_un)
        label_data = torch.load(data_path_sup)
        self.data = loaded_data
        self.data = self.data[:,0,:,:].unsqueeze(1)
        self.labels = label_data['labels']
        self.labels = self.labels.long() 
        self.valid_idx = label_data['valid_idx']

        self.shortened_data = torch.empty((len(self.valid_idx),) + self.data[0].shape)

        for idx, i in enumerate(self.valid_idx):
            self.shortened_data[idx] = self.data[i]

        assert len(self.shortened_data) == len(self.labels), f'data ({len(self.shortened_data)}) and labels ({len(self.labels)}) must have the same length'

        self.transform = transforms.Compose([
            transforms.ToPILImage(),  
            transforms.Resize(resize_shape),
            transforms.ToTensor()  
        ])
        

    def __len__(self):
        return len(self.shortened_data)

    def __getitem__(self, index):
        data_item = self.transform(self.shortened_data[index])
        label_item = self.labels[index]
        return data_item, label_item
    
def plot_ecg(i):
    record = wfdb.rdrecord(f'{physio_root}/{i:05}_hr') 
    sample_values, sample_field = wfdb.rdsamp(f'{physio_root}/{i:05}_hr')
    sample_values = np.array(sample_values)
    # first_lead_record = wfdb.Record(
    # p_signal=record.p_signal[:, 0].reshape(-1, 1),  # Reshape for single column matrix
    # fs=record.fs,
    # sig_name=[record.sig_name[0]],
    # n_sig=1,
    # units=[record.units[0]],
    # comments=record.comments
    # )
    # sig = np.array(record)
    # print(sig)
    low_cut = 0.1
    high_cut = 100
    fs = 500
    filtered_data = butter_bandpass_filter(sample_values[:,0], low_cut, high_cut, fs, order=5)
    r_idx = get_r_idx(filtered_data)
    segs = extract_segments(filtered_data, r_idx)
    if segs and len(segs) > 7:
        mid_idx = len(segs) // 2
        strt_idx = max(0, mid_idx-4)
        end_idx = strt_idx+8
        segs = segs[strt_idx:end_idx]
        del segs[0], segs[-1]
        signal = normalize(np.mean(np.array(segs), axis=0))



    plt.plot(signal)
    plt.show()
    # return filtered_data
    # wfdb.plot_wfdb(record=first_lead_record, title=f'{i-1}', ecg_grids=[0])


if __name__ == "__main__":

    physio_root = 'data/physionet/ptbxl/records500/00000'
    # directory_path = Path(f'{physio_root}')
    # num_folders = len(next(os.walk(f'{physio_root}'))[1])
    # mat_files = [file for file in directory_path.rglob('*.dat')]
    # mat_files = sorted(mat_files)
    # mat_files_without_extension = [str(file)[:-4] for file in mat_files]
    # print(mat_files_without_extension)
        
        

    ds = SupervisedDataset(data_path_un='data/datasets/unsupervised_dataset_22k_224.pt' ,data_path_sup='data/datasets/supervised_dataset_22k.pt')

    train_loader = torch.utils.data.DataLoader(ds, 
                                    batch_size=256, 
                                    shuffle=False, num_workers=2)

    for idx, (image, label) in enumerate(train_loader):
        for i in range(200,256):
            print(f'idx: {i}, label: {label[i]}')
            plot_single_img(image, i)
            # plot_ecg(i+1)
            