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
    params_cp = [12114, 21539, 30964, 40389, 49814, 191189]
    # num_params = [6290, 11235, 16180, 21125, 26070]
    # full_params = 2881921
    mses = [0.003701, 0.002618, 0.00215, 0.002287, 0.001866, 0.0015]

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


    plt.plot(epochs, loss_un, label='uncompressed adam')
    plt.plot(epochs, loss_cp, label='cpd adam')
    plt.title('Adam training and validation loss uncompressed vs cpd (24x compressed)')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.yscale('log')
    plt.show()


def plot_sgd_loss():
    num_epochs = 26
    epochs = np.arange(num_epochs)
    loss_sgd_un = [
    0.0732750, 0.0407066, 0.0296485, 0.0244884, 0.0216104,
    0.0196357, 0.0182382, 0.0172183, 0.0164245, 0.0157614,
    0.0151992, 0.0147583, 0.0143244, 0.0139463, 0.0136187,
    0.0133186, 0.0130292, 0.0127703, 0.0125228, 0.0122977,
    0.0122899, 0.0122647, 0.0122336, 0.0122135, 0.0122034,
    0.0121941
    ]
    loss_sgd_cp = [
            0.0213847, 0.0170526, 0.0155148, 0.0146263, 0.0139791,
    0.0133995, 0.0129530, 0.0125421, 0.0121045, 0.0117482,
    0.0114239, 0.0111400, 0.0108710, 0.0106306, 0.0104390,
    0.0102340, 0.0100281, 0.0098654, 0.0097105, 0.0095532,
    0.0095301, 0.0095284, 0.0095184, 0.0094924, 0.0094900,
    0.0094835
    ]

    plt.plot(epochs, loss_sgd_un, label='uncompressed sgd')
    plt.plot(epochs, loss_sgd_cp, label='cpd sgd')
    plt.title('SGD training and validation loss uncompressed vs cpd (24x compressed)')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.yscale('log')
    plt.show()


def sgd_vs_adam_loss():
    num_epochs = 26
    epochs = np.arange(num_epochs)
    loss_sgd = [
    0.0732750, 0.0407066, 0.0296485, 0.0244884, 0.0216104,
    0.0196357, 0.0182382, 0.0172183, 0.0164245, 0.0157614,
    0.0151992, 0.0147583, 0.0143244, 0.0139463, 0.0136187,
    0.0133186, 0.0130292, 0.0127703, 0.0125228, 0.0122977,
    0.0122899, 0.0122647, 0.0122336, 0.0122135, 0.0122034,
    0.0121941
    ]
    loss_adam = [0.0238849, 0.0022764, 0.0016823, 0.0008592, 0.0006434,
    0.0005292, 0.0004069, 0.0004447, 0.0004904, 0.0004808,
    0.0002920, 0.0003081, 0.0003146, 0.0002733, 0.0003427,
    0.0003022, 0.0003125, 0.0005739, 0.0003186, 0.0002659,
    0.0002083, 0.0002131, 0.0002058, 0.0002133, 0.0002020,
    0.0002014]

    plt.figure(figsize=(10,6))
    plt.plot(epochs, loss_sgd, label='SGD')
    plt.plot(epochs, loss_adam, label='Adam')
    plt.title('Val loss: Adam vs SGD for uncompressed model')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.yscale('log')
    plt.show()


def exp1_val_comp():
    epochs = np.arange(30)
    basic = [0.00512783, 0.00209671, 0.00132064, 0.00091725, 0.00072729, 0.00065543,
              0.00057368, 0.00046878, 0.00037255, 0.00038711, 0.00035563, 0.00046265,
              0.00029022, 0.00058916, 0.00034564, 0.0002656, 0.00031416, 0.00026493,
              0.00028163, 0.00023158, 0.00035372, 0.00022482, 0.00024224, 0.00022552,
              0.00021913, 0.00023666, 0.00022781, 0.00022489, 0.0002809, 0.0002232] 

    unet_32 = [4.15628565e-03, 1.10233537e-03, 5.68807312e-04, 3.92444178e-04,
           3.97415496e-04, 3.10695140e-04, 1.88708095e-04, 1.25053622e-04,
           1.01659260e-04, 1.35047512e-04, 1.67822185e-04, 5.95923086e-05,
           4.91981446e-05, 4.73952480e-05, 3.95220733e-05, 3.55118872e-05,
           3.44159929e-05, 3.39198590e-05, 3.19500427e-05, 2.72047988e-05,
           2.89163696e-05, 2.57240348e-05, 2.90727466e-05, 2.79348852e-05,
           2.53166067e-05, 2.82199899e-05, 2.86391404e-05, 3.18887552e-05,
           2.78725783e-05, 3.00213562e-05] # 0 was best run

    resnet = [0.00190439, 0.00195894, 0.00196355, 0.00200424, 0.00204703, 0.00194016,
            0.00190386, 0.00244688, 0.00238602, 0.00190779, 0.00191106, 0.00204274,
            0.00187112, 0.0020549,  0.00229959, 0.00167884, 0.00172789, 0.00170544,
            0.00169515, 0.00167654, 0.00167611, 0.00173861, 0.00169013, 0.00165769,
            0.00180312, 0.00170194, 0.00179243, 0.00166265, 0.00165337, 0.00169349]

    

    plt.figure(figsize=(10,6))
    plt.plot(epochs, basic, label='Basic')
    plt.plot(epochs, unet_32, label='U-Net')
    plt.plot(epochs, resnet, label='ResNet')
    plt.title('Validation loss comparison for best run of all models')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.yscale('log')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # plot_compression_cp()
    # plot_accs_cp()
    # plot_compression_tucker()
    # plot_accs_tucker()
    # plot_mses_cp()
    # plot_adam_loss()
    # plot_sgd_loss()
    # classifier_adam_vs_sgd()
    # sgd_vs_adam_loss()
    exp1_val_comp()