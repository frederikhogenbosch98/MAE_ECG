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
    print(Y[['scp_codes']])
    print(Y.describe().transpose())
    

   # Apply diagnostic superclass
    Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_supclass_diagnostic)
    Y['diagnostic_superclass_len'] = Y['diagnostic_superclass'].apply(len)
    print(Y.loc[Y.diagnostic_superclass_len > 1, 'diagnostic_superclass'])
    print(Y[0:50])


    labels = []
    for idx, row in Y.iterrows():
        label = []
        if 'NORM' in row['diagnostic_superclass']:
            label.append(0)
        if 'CD' in row['diagnostic_superclass']:
            label.append(1)
        if 'STCC' in row['diagnostic_superclass']:
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

    unsupervised_dataset = torch.load('data/datasets/subsets/dataset_20k_224_1000.pt')

    st = time.time()
    labels = create_label_tensor()
    end = time.time()
    print(f'total tensor creation time: {end-st}s')
    
    supervised_dataset = unsupervised_dataset[0].unsqueeze(0)
    labels_corr = []
    labels_corr.append(0)
    # for i in range(1,len(labels)):
    for i in range(1,1000):
        if not labels[i] or len(labels[i]) > 1:
            continue
        labels_corr.append(labels[i])
        supervised_dataset = torch.cat([supervised_dataset, unsupervised_dataset[i].unsqueeze(0)], dim=0)
        
    # print(len(labels_corr))
    # print(supervised_dataset.shape)

    # print(labels_corr[0:50])
    
    if SAVE:
        torch.save({'data': supervised_dataset, 'labels': labels_corr}, save_path)
        print(f'Supervised dataset saved to {save_path}')