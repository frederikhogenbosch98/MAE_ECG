import numpy as np
import matplotlib.pyplot as plt
import time


def train_test_split(tensor, split):
    split_idx = int(tensor.size(dim=0)*split)
    train_tensor = tensor[0:split_idx,:,:,:]
    test_tensor = tensor[split_idx+1:-1,:,:,:]

    return train_tensor, test_tensor 





if __name__ == "__main__":

    SAVE = True

    st = time.time()
    input_tensor = create_input_tensor()
    end = time.time()
    print(f'total tensor creation time: {end-st}s')
    # input_tensor = input_tensor.permute(0,2,1)
    # input_tensor = input_tensor[:, :, 0:4992]

    # shuffle tensors with some sort of seed

    train_tensor, test_tensor = train_test_split(input_tensor, 0.7)

    print(train_tensor.size())
    print(test_tensor.size())

    # train_dataset = ECGDataset(train_tensor)
    # test_dataset = ECGDataset(test_tensor)
    # for i in range(50):
        # plot_resulting_tensors(train_tensor,i)
    if SAVE:
        save_dir = 'data/datasets/'
        torch.save(train_tensor, f'{save_dir}train_dataset.pt')
        torch.save(test_tensor, f'{save_dir}test_dataset.pt')
        print(f'tensors saved to {save_dir}')
