import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import wfdb
from scipy.signal import butter, lfilter, filtfilt
import matplotlib.pyplot as plt
from scipy.signal import resample, find_peaks

physio_root = 'data/physionet/files/ecg-arrhythmia/1.0.0/WFDBRecords'


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


def filter_signal(i):
    sample_values, sample_field = wfdb.rdsamp(f'{physio_root}/01/010/JS00{i:03}')
    sample_values = np.array(sample_values)

    low_cut = 1
    high_cut = 100
    fs = 500

    filtered_data = butter_bandpass_filter(sample_values[:,0], low_cut, high_cut, fs, order=5)

    return sample_values, filtered_data, low_cut, high_cut


def normalize(segment):
    segment_min = np.min(segment)
    segment_max = np.max(segment)
    return (segment - segment_min) / (segment_max - segment_min)

def extract_segments(data):

    r_idx, _ = find_peaks(data, height=0.65)
    segments = []
    for idx in r_idx:
        start = max(idx-360, 0)
        end = min(idx+120, len(data))
        segment = list(data[start:end])
        segments.append(segment)
        

    return segments
 
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

if __name__ == "__main__":

    true_data, filtered_data, low_cut, high_cut = filter_signal(47)
    # plot_filtered_signal(true_data, filtered_data, low_cut, high_cut)

    normalized_data = normalize(filtered_data)
    resampled_data = resample(normalized_data, num=3600)

    extracted_segments = extract_segments(resampled_data)
    plot_segments(extracted_segments[1:])




