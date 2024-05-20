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
    accuracies = [90.0821, 90.0894, 94.213]

    # Calculate compression ratios
    # a = [full_params / i for i in num_params]

    
    plt.plot(R, accuracies, 'o-', label='Accuracies')  # 'r-' is for red solid line
    plt.xlabel('rank')
    plt.ylabel('Accuracies')
    plt.tick_params('y')
    plt.ylim(90, 100)  # Set the range of the accuracy axis
    
    # Add horizontal lines for literature and 250 epochs benchmarks
    # plt.axhline(0.9439, color='gray', linestyle='-', label='Literature (0.9439)')
    plt.axhline(94.211, color='red', linestyle='-', label='uncompressed')
    plt.title('Tucker accuracies with 50 MAE epoch and 25 CLASSIFIER epochs')
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


def classifier_adam_vs_sgd():
    epoch_adam = np.arange(14)
    epoch_sgd = np.arange(20)

    adam_train_loss_list = [
    0.5779353, 0.1946356, 0.1225612, 0.0946694, 0.0799573,
    0.0701714, 0.0627757, 0.0576435, 0.0539650, 0.0498278,
    0.0477157, 0.0439790, 0.0415124, 0.0389845
    ]
    adam_val_loss_list = [
    1.3850335, 1.1258610, 0.9338873, 0.7984684, 0.6788499,
    0.7822869, 0.8170707, 0.8465877, 0.8286164, 0.9640266,
    0.8102707, 0.8555469, 0.8640891, 0.8114818
    ]
    sgd_val_loss_list = [
    1.8872204, 1.6689397, 1.6294034, 1.4174828, 1.3488336,
    1.3591400, 1.3538532, 1.6205519, 1.2042281, 1.1175311,
    1.0546582, 1.0380862, 1.0226835, 1.0744919, 1.0292719,
    1.0530600, 0.9721900, 0.9735496, 1.0213716, 0.9780238
    ]
    sgd_train_loss_list = [
    1.0305209, 0.7625475, 0.6436381, 0.5631530, 0.5000646,
    0.4512779, 0.4124015, 0.3801628, 0.3540864, 0.3323303,
    0.3133786, 0.2974768, 0.2824948, 0.2718632, 0.2639334,
    0.2542969, 0.2463553, 0.2412917, 0.2379314, 0.2346631 
    ]


    plt.plot(epoch_adam, adam_train_loss_list, label='adam train')
    plt.plot(epoch_adam, adam_val_loss_list, label='adam val')
    plt.plot(epoch_sgd, sgd_val_loss_list, label='sgd train')
    plt.plot(epoch_sgd, sgd_train_loss_list, label='sgd val')
    plt.title('sgd vs adam loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.show()


def plot_mses_cp():
    params_un = 722113
    params_cp = [12114, 21539, 30964, 40389, 49814, 70000]
    # num_params = [6290, 11235, 16180, 21125, 26070]
    # full_params = 2881921
    mses = [0.003701, 0.002618, 0.00215, 0.002287,0.001866, 0.00024]

    print([np.round(params_un / i, 1) for i in params_cp])
    
    # Calculate compression ratios
    # a = [full_params / i for i in num_params]

    R = [np.round(params_un / i, 1) for i in params_cp]
    plt.plot(R, mses, 'o-', label='MSE')  # 'r-' is for red solid line
    plt.xlabel('compression ratio')
    plt.ylabel('Mean Squared Error')
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
    # plot_mses_cp()
    # plot_adam_loss()
    classifier_adam_vs_sgd()