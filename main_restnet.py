from torchvision import datasets, transforms
import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import ImageFolder 
from models.model_56x56_TD import AutoEncoder56_CPD, Classifier56_CPD
from models.model_56x56 import AutoEncoder56, Classifier56
import matplotlib.pyplot as plt
import time
import numpy as np
from print_funs import plot_losses, plotimg, plot_single_img, count_parameters
from torch.optim.lr_scheduler import StepLR
from ptflops import get_model_complexity_info
import math
from models.resnet50 import ResNet



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



if __name__ == "__main__":

    dtype = torch.float32
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")


    ## ECG SUPERVISED
    dataset_sup = SupervisedDataset(data_path_un='data/datasets/unsupervised_dataset_22k_224.pt', data_path_sup='data/datasets/supervised_dataset_22k.pt')
    # print(len(dataset_sup))
    trainset_sup, testset_sup = torch.utils.data.random_split(dataset_sup, [12000, 4244])    
    trainset_sup, valset_sup = torch.utils.data.random_split(trainset_sup, [10000, 2000]) 

    batch_size=512

    train_loader = torch.utils.data.DataLoader(trainset_sup, 
                                                batch_size=batch_size, 
                                                shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(valset_sup, 
                                        batch_size=batch_size, 
                                        shuffle=False)

    test_loader = torch.utils.data.DataLoader(testset_sup, 
                                                batch_size=batch_size, 
                                                shuffle=True, num_workers=2)
                        
    ### RESNET TRAINING

    num_classes = 10
    num_epochs = 20
    batch_size = 16
    learning_rate = 0.01

    model = ResNet([3, 4, 6, 3]).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay = 0.001, momentum = 0.9)  

    # Train the model
    total_step = len(train_loader)

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):  
            # Move tensors to the configured device
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            del images, labels, outputs
            torch.cuda.empty_cache()
            # gc.collect()

        print ('Epoch [{}/{}], Loss: {:.4f}' 
                    .format(epoch+1, num_epochs, loss.item()))
                
        # Validation
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                del images, labels, outputs
        
            print('Accuracy of the network on the {} validation images: {} %'.format(5000, 100 * correct / total))


    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            del images, labels, outputs

        print('Accuracy of the network on the {} test images: {} %'.format(10000, 100 * correct / total)) 