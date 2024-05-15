
from torchvision import datasets, transforms
import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import ImageFolder 
import matplotlib.pyplot as plt
import time
import numpy as np
import os
from PIL import Image

class CosineAnnealingwithWarmUp():

    def __init__(self, optimizer, n_warmup_epochs, warmup_lr, start_lr, lower_lr, alpha, epoch_int, num_epochs):
        self._optimizer = optimizer
        self.n_steps = 0
        self.n_warmup_steps = n_warmup_epochs
        self.warmup_lr = warmup_lr
        self.start_lr = start_lr
        self.epoch_int = epoch_int
        self.num_epochs = num_epochs - n_warmup_epochs
        self.current_epoch = 0
        self.lower_lr = lower_lr

        self.warmup = np.linspace(self.warmup_lr, self.start_lr, self.n_warmup_steps)
        # assert epoch_int % num_epochs  != 0, "num_epochs should be a multiple of epoch interval"

        self.alpha = np.power(alpha, np.arange(num_epochs // epoch_int))
        self.lrs = self.get_cosine_epoch()
        

    def step(self):
        self._update_learning_rate()
        self._optimizer.step()

    def normalize(self,array):
        return np.interp(array, (array.min(), array.max()), (self.lower_lr, self.start_lr))

    def zero_grad(self):
        self._optimizer.zero_grad()

    def print_seq(self):
        plt.plot(np.concatenate((self.warmup, self.get_cosine_epoch())))
        plt.title('Learning Rate Custom Scheduler')
        plt.xlabel('epochs')
        plt.ylabel('learning rate')
        plt.show()

    def get_cosine_epoch(self):
        full_ls = np.zeros(self.num_epochs)
        for i in range(self.num_epochs // self.epoch_int):
            full_ls[i*self.epoch_int:(i+1)*self.epoch_int] = self.start_lr * self.alpha[i] * np.cos(np.linspace(0, np.pi/2, self.epoch_int))

        return np.array(self.normalize(full_ls))
    
    def _update_learning_rate(self):

        if self.n_steps < self.n_warmup_steps:
            lr = self.warmup[self.n_steps]
            self.current_epoch = 0
        else:
            lr = self.lrs[self.n_steps-self.n_warmup_steps]
            
        self.n_steps += 1

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr


class EarlyStopper:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False





def apply_mask(x, ratio, p):
    x = x.permute(0,5,1,2,3,4)
    rand_mask = torch.rand(x.shape[0], x.shape[1], x.shape[2], x.shape[3]) < ratio
    rand_mask = rand_mask.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, 1, p, p)
    x = torch.where(rand_mask, torch.zeros_like(x), x) 
    return x
 

def patchify(imgs, ratio, p=4):
    """
    imgs: (N, 3, H, W)
    x: (N, L, patch_size**2 *3)
    """
    assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

    h = w = imgs.shape[2] // p
    x = imgs.reshape(shape=(imgs.shape[0], 1, h, p, w, p))
    x = torch.einsum('nchpwq->nhwpqc', x)
    x = apply_mask(x, ratio, p)
    x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 1))
    return x


def unpatchify(x, p=4):
    """
    x: (N, L, patch_size**2 *3)
    imgs: (N, 3, H, W)
    """
    h = w = int(x.shape[1]**.5)
    assert h * w == x.shape[1]
    
    x = x.reshape(shape=(x.shape[0], h, w, p, p, 1))
    x = torch.einsum('nhwpqc->nchpwq', x)
    imgs = x.reshape(shape=(x.shape[0], 1, h * p, h * p))
    return imgs


def mask(batch, ratio, p):
    x = patchify(batch.cpu(), ratio, p)
    imgs = unpatchify(x, p)
    return imgs
    # plt.subplot(2,1,1)
    # plt.imshow(batch[8,0,:,:].cpu().detach().numpy(), cmap="gray")
    # plt.subplot(2,1,2)
    # plt.imshow(imgs[8,0,:,:].cpu().detach().numpy(), cmap="gray")
    # plt.show()

class ImageWithFeatureDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None, scale_method='normalize'):
        self.root_dir = root_dir
        self.transform = transform
        self.scale_method = scale_method

        # Gather all image and feature paths
        self.image_paths = []
        self.feature_paths = []
        self.features = []

        # Traverse the directory structure
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith('.png'):
                    image_path = os.path.join(root, file)
                    feature_path = os.path.splitext(image_path)[0] + 'std.txt'
                    print(image_path) 
                    if os.path.exists(feature_path):
                        self.image_paths.append(image_path)
                        self.feature_paths.append(feature_path)
                        
                        # Load feature to collect all features for scaling
                        with open(feature_path, 'r') as f:
                            feature = float(f.read().strip())
                            print(feature)
                            self.features.append(feature)
                            print(len(self.features))

        # Print collected paths for debugging
        # print(f"Collected {len(self.image_paths)} image paths and {len(self.feature_paths)} feature paths")

        # Convert features to tensor
        print(len(self.features))
        self.features = torch.tensor(self.features, dtype=torch.float32)
        print(self.features.shape)
        print(len(self.image_paths))
        
        # Calculate scaling parameters
        if scale_method == 'normalize':
            self.feature_min = self.features.min().item()
            self.feature_max = self.features.max().item()

        elif scale_method == 'standardize':
            self.feature_mean = self.features.mean().item()
            self.feature_std = self.features.std().item()
        else:
            raise ValueError("scale_method should be either 'normalize' or 'standardize'")

        self.label_to_index = {label: idx for idx, label in enumerate(sorted(set(os.path.basename(os.path.dirname(path)) for path in self.image_paths)))}

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        feature_path = self.feature_paths[index]



        # Load image
        image = Image.open(image_path).convert('L')  # 'L' mode for grayscale
        if self.transform:
            image = self.transform(image)

        if not isinstance(image, torch.Tensor):
            image = transforms.ToTensor()(image)
        # Load feature
        with open(feature_path, 'r') as f:
            feature = float(f.read().strip())

        # Scale feature
        if self.scale_method == 'normalize':
            feature = (feature - self.feature_min) / (self.feature_max - self.feature_min)
        elif self.scale_method == 'standardize':
            feature = (feature - self.feature_mean) / self.feature_std

        # Convert feature to tensor
        feature = torch.tensor([feature], dtype=torch.float32)


        # Extract label from the path (assuming the structure is root/class_name/filename)
        label = os.path.basename(os.path.dirname(image_path))
        label_idx = self.label_to_index[label]
        # if not isinstance(label_idx, torch.Tensor):
        #     label_idx = transforms.ToTensor()(label_idx)

        return image, feature, label_idx



class UnsupervisedDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, resize_shape=(112,112)):
        loaded_data = torch.load(data_path)
        # print(type(loaded_data))
        self.data = loaded_data
        self.data = self.data[:,0,:,:].unsqueeze(1)


        self.transform = transforms.Compose([
            transforms.ToPILImage(),  
            transforms.Resize(resize_shape),
            transforms.ToTensor()  
        ])

        self.resize_shape = resize_shape
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # data_item = self.data[index].float() / 255.0  # Normalize if still in 0-255 range
        # data_item = torchvision.transforms.functional.resize(data_item, self.resize_shape)
        data_item = self.transform(self.data[index])
        return data_item


class SupervisedDataset(torch.utils.data.Dataset):
    def __init__(self, data_path_un, data_path_sup, resize_shape=(224,224)):
        self.data = torch.load(data_path_un)
        label_data = torch.load(data_path_sup)
        self.data = self.data[:,0,:,:].unsqueeze(1)
        self.labels = label_data['labels']
        self.labels = self.labels.long() 
        self.valid_idx = label_data['valid_idx']

        self.shortened_data = torch.empty((len(self.valid_idx),) + self.data[0].shape)

        for idx, i in enumerate(self.valid_idx):
            self.shortened_data[idx] = self.data[i]

        assert len(self.shortened_data) == len(self.labels), f'data ({len(self.shortened_data)}) and labels ({len(self.labels)}) must have the same length'

        self.transform = transforms.Compose([
            transforms.ToPILImage(),  
            transforms.Resize(resize_shape),
            transforms.ToTensor()  
        ])
        
        self.resize_shape = resize_shape
    def __len__(self):
        return len(self.shortened_data)

    def __getitem__(self, index):
        data_item = self.shortened_data[index].float() / 255.0  # Normalize if still in 0-255 range
        data_item = torchvision.transforms.functional.resize(data_item, self.resize_shape)
        label_item = self.labels[index]
        return data_item, label_item

