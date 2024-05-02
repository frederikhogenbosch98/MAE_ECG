import matplotlib.pyplot as plt
import numpy as np

def plot_compression_cp():
    R = [8, 12, 15, 20, 25]
    num_params = [9257, 13213, 16180, 21125, 26070]
    full_params = 2881921
    a = [full_params/i for i in num_params]
    plt.plot(R, a)
    plt.title('parameter compression ratio per rank (cp)')
    plt.xlabel('ranks')
    plt.ylabel('compression ratios')
    plt.xticks(R)
    plt.show()


def plot_compression_tucker():
    R = [8, 12, 15, 20, 25]
    num_params = [15229, 26805, 37566, 54845, 74300]
    full_params = 2881921
    a = [full_params/i for i in num_params]
    plt.plot(R, a)
    plt.title('parameter compression ratio per rank (tucker)')
    plt.xlabel('ranks')
    plt.ylabel('compression ratios')
    plt.xticks(R)
    plt.show()


if __name__ == '__main__':
    plot_compression_cp()
    plot_compression_tucker()