import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
import ast
import torch

def aggregate_supclass_diagnostic(y_dic):
    agg_df = pd.read_csv('data/physionet/ptbxl/scp_statements.csv', index_col=0)
    agg_df = agg_df[agg_df.diagnostic == 1]
    tmp = []
    for key in y_dic.keys():
        if key in agg_df.index:
            tmp.append(agg_df.loc[key].diagnostic_class)
    return list(set(tmp))
    


def create_label_tensor():

    Y = pd.read_csv('data/physionet/ptbxl/ptbxl_database.csv', index_col='ecg_id')
    Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))
    # print(Y[['scp_codes']])
    # print(Y.describe().transpose())
    

   # Apply diagnostic superclass
    Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_supclass_diagnostic)
    Y['diagnostic_superclass_len'] = Y['diagnostic_superclass'].apply(len)
    # print(Y.loc[Y.diagnostic_superclass_len > 1, 'diagnostic_superclass'])
    print(Y[0:50])


    labels = []
    for idx, row in Y.iterrows():
        label = []
        if 'NORM' in row['diagnostic_superclass']:
            label.append(0)
        if 'CD' in row['diagnostic_superclass']:
            label.append(1)
        if 'STTC' in row['diagnostic_superclass']:
            label.append(2) 
        if 'HYP' in row['diagnostic_superclass']:
            label.append(3) 
        if 'MI' in row['diagnostic_superclass']:
            label.append(4) 
        labels.append(label)

    return labels 



if __name__ == "__main__":

    SAVE = True
    save_path = 'data/datasets/supervised_dataset_22k.pt'

    unsupervised_dataset = torch.load('data/datasets/unsupervised_dataset_22k_224.pt')
    # unsupervised_dataset_last_bit = torch.load('data/datasets/subsets/dataset_20k_224_b22000.pt')

    st = time.time()
    labels = create_label_tensor()
    end = time.time()
    print(f'total tensor creation time: {end-st}s')
    print(len(labels))
    print(len(unsupervised_dataset))
    
    # supervised_dataset = unsupervised_dataset[0].unsqueeze(0)
    labels_corr = []
    # labels_corr.append(0)
    # for i in range(1,len(labels)):
    # for i in range(len(labels)):
    #     if not labels[i] or len(labels[i]) > 1:
    #         continue
    #     labels_corr.append(labels[i])
    #     supervised_dataset = torch.cat([supervised_dataset, unsupervised_dataset[i].unsqueeze(0)], dim=0)
    valid_indices = [i for i in range(len(labels)) if len(labels[i]) == 1]
    # supervised_dataset = torch.empty((len(valid_indices),) + unsupervised_dataset[0].shape)
    labels_corr = torch.empty(len(valid_indices))
    for idx, i in enumerate(valid_indices):
        # supervised_dataset[idx] = unsupervised_dataset[i]
        labels_corr[idx] = labels[i][0]
        # print(idx, i+1, labels[i][0])
        # if idx == 50:
        #     break
        

    # for i in range(25):
    #     print(i)
    #     print(labels_corr[i])
    # print(supervised_dataset.shape)

    # print(labels_corr[0:50])
    
    if SAVE:
        torch.save({'labels': labels_corr, 'valid_idx': valid_indices}, save_path)
        print(f'Supervised dataset saved to {save_path}')
    # if SAVE:
    #     torch.save({'data': supervised_dataset, 'labels': labels_corr}, save_path)
    #     print(f'Supervised dataset saved to {save_path}')