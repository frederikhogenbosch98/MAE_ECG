import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import wfdb
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt

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


def plot_ecg(i):
    record = wfdb.rdrecord(f'{physio_root}/01/010/JS00{i:03}') 
    wfdb.plot_wfdb(record=record, title=f'01 010 i')


def filter_signal_and_plot(i):
    sample_values, sample_field = wfdb.rdsamp(f'{physio_root}/01/010/JS00{i:03}')
    sample_values = np.array(sample_values)

    low_cut = 1
    high_cut = 20
    fs = 500

    filtered_data = butter_bandpass_filter(sample_values[:,0], low_cut, high_cut, fs, order=5)
    plt.plot(filtered_data)
    plt.plot(sample_values[:,0])
    plt.show()

    

if __name__ == "__main__":
    # plot_ecg(1)
    filter_signal_and_plot(8)
    filter_signal_and_plot(23)
    filter_signal_and_plot(94)
