import matplotlib.pyplot as plt
import numpy as np


def plot_losses(NUM_EPOCHS, losses):
    plt.plot(np.arange(NUM_EPOCHS), losses)
    plt.title('loss function')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()