import matplotlib.pyplot as plt
import numpy as np

def plot_compression_cp():
    R = [5, 10, 15, 20, 25]
    num_params = [6290, 11235, 16180, 21125, 26070]
    full_params = 2881921
    a = [full_params/i for i in num_params]
    plt.plot(R, a)
    plt.title('parameter compression ratio per rank (cp)')
    plt.xlabel('ranks')
    plt.ylabel('compression ratios')
    plt.xticks(R)
    plt.show()

def plot_accs_cp():
    R = [15, 20, 25]
    # num_params = [6290, 11235, 16180, 21125, 26070]
    # full_params = 2881921
    accuracies = [0.895795, 0.903269, 0.902985]
    
    # Calculate compression ratios
    # a = [full_params / i for i in num_params]

    
    plt.plot(R, accuracies, 'o-', label='Accuracies')  # 'r-' is for red solid line
    plt.xlabel('rank')
    plt.ylabel('Accuracies')
    plt.tick_params('y')
    plt.ylim(0.85, 1.0)  # Set the range of the accuracy axis
    
    # Add horizontal lines for literature and 250 epochs benchmarks
    # plt.axhline(0.9439, color='gray', linestyle='-', label='Literature (0.9439)')
    plt.axhline(0.896515, color='red', linestyle='-', label='uncompressed')
    plt.title('CP accuracies with 50 MAE epoch and 25 CLASSIFIER epochs')
    plt.xticks(R)
    plt.legend() 
    plt.show()


def plot_compression_tucker():
    R = [5, 10, 15, 20, 25]
    num_params = [8626, 20621, 37566, 54845, 74300]
    full_params = 2881921
    a = [full_params/i for i in num_params]
    plt.plot(R, a)
    plt.title('parameter compression ratio per rank (tucker)')
    plt.xlabel('ranks')
    plt.ylabel('compression ratios')
    plt.xticks(R)
    plt.show()

def plot_accs_tucker():
    R = [15, 20, 25]
    # num_params = [6290, 11235, 16180, 21125, 26070]
    # full_params = 2881921
    accuracies = [0.900821, 0.900894, 0.898160]

    # Calculate compression ratios
    # a = [full_params / i for i in num_params]

    
    plt.plot(R, accuracies, 'o-', label='Accuracies')  # 'r-' is for red solid line
    plt.xlabel('rank')
    plt.ylabel('Accuracies')
    plt.tick_params('y')
    plt.ylim(0.85, 1.0)  # Set the range of the accuracy axis
    
    # Add horizontal lines for literature and 250 epochs benchmarks
    # plt.axhline(0.9439, color='gray', linestyle='-', label='Literature (0.9439)')
    plt.axhline(0.896515, color='red', linestyle='-', label='uncompressed')
    plt.title('Tucker accuracies with 50 MAE epoch and 25 CLASSIFIER epochs')
    plt.xticks(R)
    plt.legend() 
    plt.show()


def plot_mses_cp():
    params_un = 722113
    params_cp = [12114, 21539, 30964, 40389, 50000, 70000]
    # num_params = [6290, 11235, 16180, 21125, 26070]
    # full_params = 2881921
    mses = [0.003701, 0.002618, 0.00215, 0.0005, 0.0001, 0.00024]
    
    # Calculate compression ratios
    # a = [full_params / i for i in num_params]

    R = [np.round(params_un / i, 0) for i in params_cp]
    plt.plot(R, mses, 'o-', label='Accuracies')  # 'r-' is for red solid line
    plt.xlabel('compression ratio')
    plt.ylabel('Accuracies')
    plt.tick_params('y')
    plt.xlim(70, 0)
    plt.ylim(0.0000, 0.004)  # Set the range of the accuracy axis
    
    # Add horizontal lines for literature and 250 epochs benchmarks
    # plt.axhline(0.9439, color='gray', linestyle='-', label='Literature (0.9439)')
    plt.axhline(0.000204, color='red', linestyle='-', label='uncompressed')
    plt.title('Mean Squared Error (MSE) CPD at various ranks')
    plt.xticks(R)
    plt.legend() 
    plt.show()


def plot_sgd_loss():
    num_epochs = 30
    epochs = np.arange(num_epochs)
    loss_un = []
    loss_cp = []

    plt.plot(epochs, loss_un, label='uncompressed')
    plt.plot(epochs, loss_cp, label='cpd')
    plt.title('SGD validation loss uncompressed vs cpd')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.show()

def plot_adam_loss():
    num_epochs = 30
    epochs = np.arange(num_epochs)
    loss_un = [0.0238849, 0.0022764, 0.0016823, 0.0008592, 0.0006434,
    0.0005292, 0.0004069, 0.0004447, 0.0004904, 0.0004808,
    0.0002920, 0.0003081, 0.0003146, 0.0002733, 0.0003427,
    0.0003022, 0.0003125, 0.0005739, 0.0003186, 0.0002659,
    0.0002083, 0.0002131, 0.0002058, 0.0002133, 0.0002020,
    0.0002014, 0.0002023, 0.0002001, 0.0002032, 0.0002037]
    loss_cp = [0.0098841, 0.0055452, 0.0048496, 0.0054096, 0.0036840,
    0.0034440, 0.0030900, 0.0030042, 0.0028229, 0.0027689,
    0.0029659, 0.0026524, 0.0025764, 0.0025059, 0.0023866,
    0.0024003, 0.0024250, 0.0023358, 0.0025048, 0.0022827,
    0.0021646, 0.0021549, 0.0021544, 0.0021449, 0.0021587,
    0.0021498, 0.0021429, 0.0021406, 0.0021506, 0.0021425]

    plt.plot(epochs, loss_un, label='uncompressed')
    plt.plot(epochs, loss_cp, label='cpd')
    plt.title('Adam validation loss uncompressed vs cpd (24x compressed)')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.yscale('log')
    plt.show()

if __name__ == '__main__':
    # plot_compression_cp()
    # plot_accs_cp()
    # plot_compression_tucker()
    # plot_accs_tucker()
    plot_mses_cp()
    plot_adam_loss()