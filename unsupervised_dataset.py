import wfdb
import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
from scipy.signal import butter, lfilter
import pandas as pd
from scipy.signal import resample, find_peaks
import matplotlib.pyplot as plt
import os
import io
from PIL import Image
import time
from print_funs import plot_single_img
import argparse

physio_root = 'data/physionet/ptbxl/records500'


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


def create_input_tensor(folder):
    print(f'creating datasets')
    lower_limit = 1000*folder
    upper_limit = 1000*(folder+1)
    
    low_cut = 0.1
    high_cut = 100
    fs = 500

    directory_path = Path(f'{physio_root}')
    num_folders = len(next(os.walk(f'{physio_root}'))[1])
    mat_files = [file for file in directory_path.rglob('*.dat')]
    mat_files = sorted(mat_files)
    mat_files_without_extension = [str(file)[:-4] for file in mat_files]

    
    t_start = time.time()

    for idx, file in enumerate(mat_files_without_extension):
        print(file)
        if idx < lower_limit:
            continue
        elif idx >= upper_limit:
            continue

        if idx % 1000 == 0:
            print(f'processing folder {(idx//1000)+1}/{num_folders}')

        ### signal data
        sample_values, sample_field = wfdb.rdsamp(f'{file}')
        sample_values = np.array(sample_values)
        filtered_data = []
        resampled_data = []
        segs = []

        output_data = np.zeros((300, 12))
        
        for l in range(sample_values.shape[1]):
            filtered_data = butter_bandpass_filter(sample_values[:,l], low_cut, high_cut, fs, order=5)
            resampled_data = resample(filtered_data, num=3600)

            if l == 0:
                r_idx = get_r_idx(resampled_data)

            segs = extract_segments(resampled_data, r_idx)
            if segs and len(segs) > 7:
                mid_idx = len(segs) // 2
                strt_idx = max(0, mid_idx-4)
                end_idx = strt_idx+8
                segs = segs[strt_idx:end_idx]
                del segs[0], segs[-1]
                output_data[:,l] = normalize(np.mean(np.array(segs), axis=0))

            buf = create_img(output_data[:,l], 224, 224)
            buf.seek(0)
            image = Image.open(buf).convert('L')
            image_array = np.array(image)
            if l == 0:  
                img_tensor = torch.tensor(image_array[0:224, 0:224])
                img_tensor = img_tensor[None, :, :]
            else:
                temp_img_tensor = torch.tensor(image_array[0:224, 0:224]) 
                temp_img_tensor = temp_img_tensor[None, :, :]
                img_tensor = torch.cat([img_tensor, temp_img_tensor], dim=0)


        if idx == 21:
            for l in range(11):
                plt.plot(output_data[:,l])
                plt.show()

        if idx % 100 == 0 and idx != 0:
            print(idx)

        if idx == lower_limit:
            input_tensor = img_tensor.unsqueeze(0)
        else:
            input_tensor = torch.cat([input_tensor, img_tensor.unsqueeze(0)], dim=0)


    print(input_tensor.size())
    return input_tensor




def create_img(signal, width, height):

    dpi = 230 
    fig_width_in = width / dpi
    fig_height_in = height / dpi
    t = np.linspace(0, 1, len(signal))  

    fig, ax = plt.subplots(figsize=(fig_width_in, fig_height_in), dpi=dpi)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')


    ax.plot(t, signal, color='black', linewidth=0.5)
    ax.axis('off')
    buf = io.BytesIO()

    plt.savefig(buf, dpi=300, bbox_inches='tight', pad_inches=0)
    # plt.show()
    plt.close(fig)
    
    return buf



def plot_resulting_tensors(tensor, i):
    plt.plot(tensor[i,0,:,0])
    plt.plot(tensor[i,1,:,0])
    plt.plot(tensor[i,2,:,0])
    plt.plot(tensor[i,3,:,0])
    plt.plot(tensor[i,4,:,0])
    plt.plot(tensor[i,5,:,0])

    plt.show()

def train_test_split(tensor, split):
    split_idx = int(tensor.size(dim=0)*split)
    train_tensor = tensor[0:split_idx,:,:,:]
    test_tensor = tensor[split_idx+1:-1,:,:,:]

    return train_tensor, test_tensor 

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
                    prog='UnsupervisedDataset',
                    description='Create dataset tensors',
                    epilog='Help')

    parser.add_argument('-f', '--folder')
    args = parser.parse_args()
    SAVE = False

    st = time.time()
    input_tensor = create_input_tensor(int(args.folder))
    end = time.time()
    print(f'total tensor creation time: {end-st}s')
    series = (int(args.folder)+1) * 1000
    if SAVE:
        save_dir = 'data/datasets/'
        torch.save(input_tensor, f'{save_dir}/subsets/dataset_20k_224_{series}.pt')
        # torch.save(test_tensor, f'{save_dir}test_dataset_20k_224.pt')
        print(f'tensors saved to {save_dir}')

