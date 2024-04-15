import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_losses(NUM_EPOCHS, losses):
    plt.plot(np.arange(NUM_EPOCHS), losses)
    plt.title('validation loss function')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()


def plotimg(test_tensor, recon):
    test_tensor = test_tensor.cpu()#.detach().numpy()
    # print(test_tensor[0,:,:])
    plt.subplot(2, 2, 1)
    plt.imshow(unnormalize(test_tensor[:,:,:]))#, cmap="gray")
    plt.subplot(2, 2, 2)
    # print(recon)
    plt.imshow(unnormalize(recon))#, cmap="gray")
    plt.show()


def plot_single_img(img, i, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    unnormalized_image = unnormalize(img[i], mean, std)
    plt.imshow(unnormalized_image[:, :, :])#,cmap="gray")
    plt.show()


def unnormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):

    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    tensor = tensor * std + mean  
    tensor = tensor.clamp(0, 1)  
    return tensor.detach().numpy().transpose(1, 2, 0) 