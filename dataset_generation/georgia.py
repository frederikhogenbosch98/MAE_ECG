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
# from print_funs import plot_single_img
import argparse
import cv2
import tqdm
from scipy.io import loadmat

# physio_root = 'data/physionet/georgia_raw/'
physio_root = '../datasets/WFDB/'
_directory = '../../extra_reps/data/mitbih/'
_dataset_dir = 'data/physionet/georgia/class'

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


def create_input_tensor():
    print(f'creating datasets')
    # lower_limit = 1000*folder
    # upper_limit = 1000*(folder+1)
    
    low_cut = 0.1
    high_cut = 100
    fs = 500

    directory_path = Path(f'{physio_root}')
    num_folders = len(next(os.walk(f'{physio_root}'))[1])
    mat_files = [file for file in directory_path.rglob('*.mat')]
    mat_files = sorted(mat_files)
    mat_files_without_extension = mat_files#[str(file)[:-4] for file in mat_files]

    size = (112,112)
    
    t_start = time.time()

    for idx, file in enumerate(tqdm.tqdm(mat_files_without_extension)):
        # print(file)
        # if idx < lower_limit:
        #     continue
        # elif idx >= upper_limit:
        #     continue

        # if idx % 1000 == 0:
        #     print(f'processing folder {(idx//1000)+1}/{num_folders}')

        ### signal data
        # sample_values, sample_field = wfdb.rdsamp(f'{file}')
        sample_values = loadmat(f'{file}')
        # print(sample_values['val'].shape)
        sample_values = sample_values['val'][0]
        sample_values = np.array(sample_values)
        filtered_data = []
        segs = []

        
        filtered_data = butter_bandpass_filter(sample_values, low_cut, high_cut, fs, order=5)
        resampled_data = resample(filtered_data, num=3600)
        r_idx = get_r_idx(resampled_data)

        segs = extract_segments(filtered_data, r_idx)
        if segs and len(segs) > 7:
            mid_idx = len(segs) // 2
            strt_idx = max(0, mid_idx-4)
            end_idx = strt_idx+8
            segs = segs[strt_idx:end_idx]
            del segs[0], segs[-1]
            segs = normalize(np.array(segs))


        for i in range(len(segs)):


            filename = '{}/{}_{}.png'.format(_dataset_dir, str(file)[-8:-3], i)            
            buf = create_img(segs[i], 224, 224)
            image_pil = Image.open(buf)
            image_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2GRAY)
            cv2.imwrite(filename, image_cv)
            plt.cla()
            plt.clf()
            plt.close('all')

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




if __name__ == "__main__":

    # parser = argparse.ArgumentParser(
    #                 prog='UnsupervisedDataset',
    #                 description='Create dataset tensors',
    #                 epilog='Help')

    # parser.add_argument('-f', '--folder')
    # args = parser.parse_args()
    SAVE = False

    st = time.time()
    # input_tensor = create_input_tensor(int(args.folder))
    create_input_tensor()
    end = time.time()
    print(f'total tensor creation time: {end-st}s')
    # series = (int(args.folder)+1) * 1000
    # if SAVE:
    #     save_dir = 'data/datasets/'
    #     torch.save(input_tensor, f'{save_dir}/subsets/dataset_20k_224_{series}.pt')
    #     # torch.save(test_tensor, f'{save_dir}test_dataset_20k_224.pt')
    #     print(f'tensors saved to {save_dir}')

