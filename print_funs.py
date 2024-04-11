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
    plt.imshow(test_tensor[0,:,:])
    plt.subplot(2, 2, 2)
    plt.imshow(recon)
    plt.show()