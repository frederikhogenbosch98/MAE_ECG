import matplotlib.pyplot as plt
import numpy as np


def plot_losses(NUM_EPOCHS, losses):
    plt.plot(np.arange(NUM_EPOCHS), losses)
    plt.title('loss function')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()


def plotimg(test_tensor, recon):
    test_tensor = test_tensor.cpu().detach().numpy()
    plt.subplot(2, 2, 1)
    plt.imshow(test_tensor[0,:,:], cmap="gray")
    plt.subplot(2, 2, 2)
    plt.imshow(recon, cmap="gray")
    plt.show()


def plot_single_img(img, i):
    plt.imshow(img[i, 0, :, :],cmap="gray")
    plt.show()