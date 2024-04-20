import torch
import os

pathstr = f'data/datasets/subsets/' 
files =  sorted(os.listdir(pathstr))



for idx, file in enumerate(files):
    if idx == 0:
        combined_tensor = torch.Tensor(torch.load(pathstr+file))
    else:
        combined_tensor = torch.cat([combined_tensor, torch.Tensor(torch.load(pathstr+file))], dim=0)

save_dir = 'data/datasets/'
torch.save(combined_tensor, f'{save_dir}/dataset_22k_224.pt')

print(f'combined tensor shape: {combined_tensor.shape}')
