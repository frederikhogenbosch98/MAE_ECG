import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import wfdb
from scipy.signal import butter, lfilter, filtfilt
import matplotlib.pyplot as plt
from scipy.signal import resample, find_peaks
from scipy.fft import fft, fftfreq
from PIL import Image, ImageDraw
import cv2

# physio_root = 'data/physionet/files/ecg-arrhythmia/1.0.0/WFDBRecords'
physio_root = 'data/physionet/ptbxl/records500'


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band', analog=False)
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def plot_ecg(i):
    record = wfdb.rdrecord(f'{physio_root}/01/010/JS00{i:03}') 
    wfdb.plot_wfdb(record=record, title=f'01 010 i')


def filter_signal(i, lead):
    # sample_values, sample_field = wfdb.rdsamp(f'{physio_root}/01/010/JS00{i:03}')
    sample_values, sample_field = wfdb.rdsamp(f'{physio_root}/00000/00{i:03}_hr')
    sample_values = np.array(sample_values)

    low_cut = 0.1
    high_cut = 100 
    fs = 500

    # print(sample_values.shape)
    filtered_data = butter_bandpass_filter(sample_values[:,lead], low_cut, high_cut, fs, order=5)

    return sample_values, filtered_data, low_cut, high_cut


def normalize(segment):
    segment_min = np.min(segment)
    segment_max = np.max(segment)
    return (segment - segment_min) / (segment_max - segment_min)


def extract_segments(data):
    r_idx, _ = find_peaks(data, distance=250)
    segments = []

    for idx in r_idx:
        start = max(idx-100, 0)
        end = min(idx+200, len(data))
        segment = list(data[start:end])
        segments.append(segment)
        

    return r_idx, segments


def averaging(segments):
    averaged_signal = []
    seg = 0
    for j in range(len(segments[seg])):
        mean_vec = []
        seg += 1
        for i in range(len(segments)):
            mean_vec.append(segments[i][j])
        averaged_signal.append(np.average(mean_vec))

    return averaged_signal


def inspect_freqs(ecg_signal, fs):

    """
    Plot the FFT of an ECG signal.

    Parameters:
    - ecg_signal: Array-like, the ECG signal samples.
    - fs: Sampling frequency in Hz.
    """
    # Number of samples in the ECG signal
    N = len(ecg_signal)
    
    # Compute the FFT of the ECG signal
    fft_ecg = np.fft.fft(ecg_signal)
    # Compute the frequencies for each FFT component
    freqs = np.fft.fftfreq(N, d=1/fs)
    
    # Magnitude of the FFT (2-sided spectrum)
    magnitude = np.abs(fft_ecg)
    
    # Only plot the one-sided spectrum (up to Nyquist frequency)
    # Nyquist frequency is fs/2
    n = N // 2
    plt.figure(figsize=(10, 6))
    plt.plot(freqs[:n], magnitude[:n])
    plt.title('FFT of ECG Signal')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.grid(True)
    plt.show()


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

    plt.savefig('signal_image_pixels.png', dpi=300, bbox_inches='tight', pad_inches=0)
    # plt.show()





def plot_filtered_signal(sample_values, filtered_data, low_cut, high_cut):
    x = np.linspace(0,2000, len(sample_values[:,0]))
    plt.plot(x, sample_values[:,0], label='true')
    plt.plot(x, filtered_data, label='filtered')
    plt.title(f'Bandpass filter (cutoff low: {low_cut}Hz, cutoff high: {high_cut}Hz)')
    plt.xlabel('time (ms)')
    plt.ylabel('voltage (mV)')
    plt.legend()
    plt.show()


def plot_peaks(resampled_data):

    plt.plot(resampled_data)
    plt.plot(extracted_segments, resampled_data[extracted_segments], "x")
    plt.plot(np.zeros_like(resampled_data), "--", color="gray")
    plt.show()


def plot_segments(segments):
    for i in range(len(segments)):
        plt.plot(segments[i],label = 'id %s'%i)
    plt.show()

def plot_resulting_tensors(tensor):
    plt.plot(tensor[0,0,:,0])
    plt.plot(tensor[0,1,:,0])
    plt.plot(tensor[0,2,:,0])
    plt.plot(tensor[0,3,:,0])
    plt.plot(tensor[0,4,:,0])
    plt.plot(tensor[0,5,:,0])

    plt.show()

if __name__ == "__main__":
    # plot_ecg(27)
    true_data, filtered_data, low_cut, high_cut = filter_signal(47, 1)
    num_segs = []
    for lead in range(0,12):
        # print(lead)
        try:
            true_data, filtered_data, low_cut, high_cut = filter_signal(27, lead)
            # true_data, filtered_data, low_cut, high_cut = filter_signal(37) 
            plot_filtered_signal(true_data, filtered_data, low_cut, high_cut)

            resampled_data = resample(filtered_data, num=3600)
            # inspect_freqs(resampled_data, 360)
            # plt.plot(filtered_data)
            # plt.plot(resampled_data)
            # plt.show()


            r_idx, extracted_segments = extract_segments(resampled_data)
            # plot_peaks(resampled_data)
            # plt.plot(resampled_data)
            # plt.plot(r_idx, resampled_data[r_idx], "x")
            # plt.plot(np.zeros_like(resampled_data), "--", color="gray")
            # plt.show()
            del extracted_segments[0], extracted_segments[-1]
            # plot_segments(extracted_segments)

            # averaged_signal = averaging(extracted_segments)
            averaged_signal = np.mean(np.array(extracted_segments), axis=0)
            # averaged_signal = np.array(extracted_segments)
            num_segs.append(averaged_signal.shape[0])
            # normalized_data = np.zeros((averaged_signal.shape))
            # for j in range(averaged_signal.shape[0]):
                # normalized_data[j,:] = normalize(averaged_signal[j,:])
            normalized_data = normalize(averaged_signal)
            create_img(normalized_data, width=224, height=224)
        except FileNotFoundError:
            continue
        
    # print(np.unique(num_segs, return_counts=True))

    # train_tensor = torch.load('data/datasets/train_dataset.pt')
    # train_tensor = train_tensor.permute(0, 3, 2, 1)
    # plot_resulting_tensors(train_tensor)
    # plt.plot(normalized_data[5,:])    
    # plt.plot(normalized_data[2,:])    
    # plt.plot(normalized_data[0,:])    
    # plt.show()


    # train_tensor = torch.load('data/datasets/train_dataset.pt')
    # print(train_tensor[0,0,:,0])



