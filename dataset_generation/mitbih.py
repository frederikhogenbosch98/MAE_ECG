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

_range_to_ignore = 20
_directory = '../../extra_reps/data/mitbih/'
# _directory = 'data/physionet/mitbih_raw/'
_dataset_dir = '../data/physionet/mitbih_224/'
# _dataset_dir = 'data/physionet/mitbih/'
_dataset_ann_dir = '../extra_reps/data/dataset_ann/'
_split_percentage = .50
_split_validation_percentage = 0.3
_split_test_percentage = 0.50


def nearest_integers(lst, index, num_neighbors=4):
    if index < 0 or index >= len(lst):
        raise ValueError("Index out of bounds")
    
    start_index = max(1, index - num_neighbors // 2)
    end_index = min(len(lst), start_index + num_neighbors + 1)
    
    if end_index - start_index < num_neighbors + 1:
        start_index = max(1, end_index - num_neighbors - 1)
    
    return lst[start_index:end_index]


def create_img_from_sign(lblabels, lbrevert_labels, lboriginal_labels, size=(224, 224), augmentation=True):
    """
       For each beat for each patient creates img apply some filters
       :param size: the img size
       :param augmentation: create for each image another nine for each side
    """

    if not os.path.exists(_directory):
        os.makedirs(_directory)

    files = [f[:-4] for f in listdir(_directory) if isfile(join(_directory, f)) if (f.find('.dat') != -1)]

    random.shuffle(files)
    ds1 = files[: int(len(files) * _split_percentage)]
    ds2 = files[int(len(files) * _split_percentage):]
    ds11 = ds1[int(len(ds1) * _split_validation_percentage):]
    ds12 = ds1[: int(len(ds1) * _split_validation_percentage)]
    print(f'ds11 files: {ds11}')
    print(f'ds12 files: {ds12}')
    print(f'ds2 files: {ds2}')
    N_std = []
    S_std = []
    F_std = []
    V_std = []
    Q_std = [] 

    for file in files:
        sig, _ = wfdb.rdsamp(_directory + file)
        ann = wfdb.rdann(_directory + file, extension='atr')
        len_sample = len(ann.sample)
        print(len_sample)
        for i in tqdm.tqdm(range(len_sample-10, len_sample - 2)):
            if ann.symbol[i] not in lboriginal_labels:
                continue
            label = lboriginal_labels[ann.symbol[i]]

            if file in ds11:
                dir = '{}DS11/{}'.format(_dataset_dir, label)
            elif file in ds12:
                dir = '{}DS12/{}'.format(_dataset_dir, label) 
            else:
                dir = '{}DS2/{}'.format(_dataset_dir, label)

            if not os.path.exists(dir):
                os.makedirs(dir)

            # if label == 'S':
            # print(f'--------{i}---------')
            # print(f'--------{label}---------')


            rr_intervals = []
            for j in nearest_integers(np.arange(1, len_sample), i):
                print(j+1)
                print(nearest_integers(np.arange(1, len_sample)))[:-3]
                rr_intervals.append((ann.sample[j+1] - ann.sample[j])/360)

            mean_RR = np.mean(rr_intervals)
            sdnn = np.std(rr_intervals, ddof=1)
            # print(sdnn)

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


            ''' Get the Q-peak intervall '''
            start = ann.sample[i - 1] + _range_to_ignore
            end = ann.sample[i + 1] - _range_to_ignore

            ''' Get the signals '''
            plot_x = [sig[i][0] for i in range(start, end)]

            ''' Convert in gray scale and resize img '''
            if file in ds11:
                filename = '{}DS11/{}/{}_{}{}{}.png'.format(_dataset_dir, label, label, file[-3:], start, end)
                filename_std = '{}DS11/{}/{}_{}{}{}{}.txt'.format(_dataset_dir, label, label, file[-3:], start, end, 'std')
            elif file in ds12:
                filename = '{}DS12/{}/{}_{}{}{}.png'.format(_dataset_dir, label, label, file[-3:], start, end) 
                filename_std = '{}DS12/{}/{}_{}{}{}{}.txt'.format(_dataset_dir, label, label, file[-3:],start, end, 'std')
            else:
                filename = '{}DS2/{}/{}_{}{}{}.png'.format(_dataset_dir, label, label, file[-3:], start, end)            
                filename_std = '{}DS2/{}/{}_{}{}{}{}.txt'.format(_dataset_dir, label, label, file[-3:], start, end, 'std')
            buf = create_img(plot_x, 224, 224)
            image_pil = Image.open(buf)
            image_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2GRAY)
            cv2.imwrite(filename, image_cv)
            plt.cla()
            plt.clf()
            plt.close('all')
            with open(filename_std, 'w') as file2: 
                file2.write(str(sdnn))

        print(f'N: {np.mean(N_std)}')
        print(f'S: {np.mean(S_std)}')
        print(f'V: {np.mean(V_std)}')
        print(f'F: {np.mean(F_std)}')
        print(f'Q: {np.mean(Q_std)}')
    print(f'N: {np.mean(N_std)}')
    print(f'S: {np.mean(S_std)}')
    print(f'V: {np.mean(V_std)}')
    print(f'F: {np.mean(F_std)}')
    print(f'Q: {np.mean(Q_std)}')



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
    labels_json = '{ ".": "N", "N": "N", "V": "V", "/": "Q", "L": "N", "R": "N", "A": "S", "a": "S", "J": "S", "S":"S", "F":"F", "e":"N", "j":"N", "E":"V", "f":"Q", "Q":"Q"}'
    labels_to_float = '{ "N": "0", "S" : "1", "V": "2", "F": "3", "Q": "4"}'
    float_to_labels = '{ "0": "N", "1" : "S", "2": "V", "3": "F", "4": "Q"}'
    labels = json.loads(labels_to_float)
    revert_labels = json.loads(float_to_labels)
    original_labels = json.loads(labels_json)

    create_img_from_sign(lblabels=labels, lbrevert_labels=revert_labels, lboriginal_labels=original_labels)
