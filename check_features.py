from nn_funcs import CosineAnnealingwithWarmUp, EarlyStopper, MITBIHImageWithFeatureDataset, INCARTDBImageWithFeatureDataset
import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np

if __name__ == "__main__":    


    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((128, 128)), 
        transforms.ToTensor(),         
        ])
    

    mitbih_ds11_dir = 'data/physionet/mitbih_rr/DS11/'
    mitbih_ds12_dir = 'data/physionet/mitbih_rr/DS12/'
    mitbih_ds2_dir = 'data/physionet/mitbih_rr/DS2/'
    mitbih_dataset_train = MITBIHImageWithFeatureDataset(root_dir=mitbih_ds11_dir, transform=transform)
    mitbih_dataset_val = MITBIHImageWithFeatureDataset(root_dir=mitbih_ds12_dir, transform=transform)
    mitbih_dataset_test = MITBIHImageWithFeatureDataset(root_dir=mitbih_ds2_dir, transform=transform) 
    # print(len(mitbih_dataset_train))
    # print(len(mitbih_dataset_val))
    # print(len(mitbih_dataset_test))
    incartdb_dir = 'data/physionet/incartdb_rr/'
    incartdb_dataset = INCARTDBImageWithFeatureDataset(root_dir=incartdb_dir, transform=transform)


    set = torch.utils.data.ConcatDataset([mitbih_dataset_train, mitbih_dataset_test, mitbih_dataset_val, incartdb_dataset])
    test_loader = torch.utils.data.DataLoader(set, 
                                            batch_size=256, 
                                            shuffle=True)
    
    Nl = []
    Sl = []
    Vl = []
    Fl = []
    Ql = []

    for images, features, labels in test_loader:
        for i in range(len(labels)):

            if labels[i].item() == 1:
                Nl.append(features[i].item())
            elif labels[i].item() == "3":
                Sl.append(features[i].item())
            elif labels[i].item() == "4":
                Vl.append(features[i].item())
            elif labels[i].item() == "0":
                Fl.append(features[i].item())
            elif labels[i].item() == "2":
                Ql.append(features[i].item())
    

    print(f'N: {np.mean(Nl)}')
    print(f'S: {np.mean(Sl)}')
    print(f'V: {np.mean(Vl)}')
    print(f'F: {np.mean(Fl)}')
    print(f'Q: {np.mean(Ql)}')