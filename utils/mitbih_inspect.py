'''
Original file from ECGNeuralNetwork repository by lorenzobrusco.
Modifications were made.
Original repository: https://github.com/lorenzobrusco/ECGNeuralNetwork
'''


import json
import matplotlib.pyplot as plt
import tqdm
import random
import cv2
from os.path import isfile, join
from os import listdir
import os
import wfdb
import numpy as np
from PIL import Image
import io

_range_to_ignore = 30
# _directory = '../../extra_reps/data/mitbih/'
# _directory = 'data/physionet/mitbih_raw/'
_dataset_dir = '../data/physionet/mitbih_224/'
# _dataset_dir = 'data/physionet/mitbih/'
_dataset_ann_dir = '../extra_reps/data/dataset_ann/'
_split_percentage = .50
_split_validation_percentage = 0.3
_split_test_percentage = 0.50
_directory = '../../datasets/physionet.org/files/files/'
_dataset_dir = 'data/physionet/incartdb_rr/'


def nearest_integers(lst, index, num_neighbors=4):
    # if index < 1 or index >= len(lst):
        # raise ValueError("Index out of bounds")
    
    start_index = max(1, index - num_neighbors // 2)
    end_index = min(len(lst), start_index + num_neighbors + 1)
    
    if end_index - start_index < num_neighbors + 1:
        start_index = max(1, end_index - num_neighbors - 1)
    
    return np.arange(start_index, end_index)


def create_img_from_sign(lblabels, lbrevert_labels, lboriginal_labels, size=(224, 224), augmentation=True):
    """
       For each beat for each patient creates img apply some filters
       :param size: the img size
       :param augmentation: create for each image another nine for each side
    """

    if not os.path.exists(_directory):
        os.makedirs(_directory)

    files = [f[:-4] for f in listdir(_directory) if isfile(join(_directory, f)) if (f.find('.dat') != -1)]

    # to_remove = ['104', '102', '107', '217']
    # for fl in to_remove:
    #     files.remove(fl)
    files = sorted(files)
    print(len(files))

    # random.shuffle(files)
    # ds1 = files[: int(len(files) * _split_percentage)]
    # ds2 = files[int(len(files) * _split_percentage):]
    # ds11 = ds1[int(len(ds1) * _split_validation_percentage):]
    # ds12 = ds1[: int(len(ds1) * _split_validation_percentage)]
    ds11 = ['101', '106', '108', '109', '112', '114', '115', '116', '119', '122', '124', '203', '205', '208', '209', '215', '220', '223']
    ds12 = ['118', '201', '207', '230']
    ds2 = ['100', '103', '105', '111', '113', '117', '121', '123', '200', '202', '210', '212', '213',\
           '214', '219', '221', '222', '228', '231', '232', '233', '234']
    N_std = []
    S_std = []
    F_std = []
    V_std = []
    Q_std = [] 

    for file in files:
        sig, _ = wfdb.rdsamp(_directory + file)
        ann = wfdb.rdann(_directory + file, extension='atr')
        len_sample = len(ann.sample)
        rr_intervals = []
        for k in range(1, len_sample-1):
            rr_intervals.append((ann.sample[k] - ann.sample[k-1])/360)
         
        rr_mean = np.mean(rr_intervals)

        for i in tqdm.tqdm(range(2, len_sample - 2)):
            if ann.symbol[i] not in lboriginal_labels:
                continue
            label = lboriginal_labels[ann.symbol[i]]



            # if label == 'S':
            # print(f'--------{i}---------')
            # print(f'--------{label}---------')

            rr_interval = (ann.sample[i] - ann.sample[i-1])/360


            sdnn = rr_mean - rr_interval 
            if sdnn > 0.2:
                sdnn = sdnn
            else:
                sdnn = 0
            # if (rr_mean - rr_interval) > (30*0.001):
            #     print(label)
                


            if label == "N":
                N_std.append(sdnn)
            elif label == "S":
                S_std.append(sdnn)
            elif label == "V":
                V_std.append(sdnn)
            elif label == "F":
                F_std.append(sdnn)
            elif label == "Q":
                Q_std.append(sdnn)


    print(f'N: {np.mean(N_std)}')
    print(f'S: {np.mean(S_std)}')
    print(f'V: {np.mean(V_std)}')
    print(f'F: {np.mean(F_std)}')
    print(f'Q: {np.mean(Q_std)}')



if __name__ == "__main__":
    labels_json = '{ ".": "N", "N": "N", "V": "V", "/": "Q", "L": "N", "R": "N", "A": "S", "a": "S", "J": "S", "S":"S", "F":"F", "e":"N", "j":"N", "E":"V", "f":"Q", "Q":"Q"}'
    labels_to_float = '{ "N": "0", "S" : "1", "V": "2", "F": "3", "Q": "4"}'
    float_to_labels = '{ "0": "N", "1" : "S", "2": "V", "3": "F", "4": "Q"}'
    labels = json.loads(labels_to_float)
    revert_labels = json.loads(float_to_labels)
    original_labels = json.loads(labels_json)

    create_img_from_sign(lblabels=labels, lbrevert_labels=revert_labels, lboriginal_labels=original_labels)
