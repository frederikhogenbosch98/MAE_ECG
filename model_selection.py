import argparse
import matplotlib.pyplot as plt
import os
import time
import numpy as np
import tqdm
from datetime import datetime

import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder 
from torch.optim.lr_scheduler import StepLR
from ptflops import get_model_complexity_info

from models.model_56x56_TD import AutoEncoder56_TD, Classifier56_TD
from models.model_56x56 import AutoEncoder56, Classifier56
from models.resnet50 import ResNet
from models.UNet import UNet, Classifier56Unet
from models.model_self_TD import AutoEncoder_self_TD, Classifier_self_TD
from models._11am_back import AutoEncoder11
from models._11am_down import AutoEncoder11_DOWN
from models.convnext import ConvNext
from models._11am_un import AutoEncoder11_UN, Classifier_UN

from main import train_mae, train_classifier, eval_mae, eval_classifier
from print_funs import plot_losses, plotimg, plot_single_img, count_parameters
from nn_funcs import CosineAnnealingwithWarmUp, EarlyStopper, MITBIHImageWithFeatureDataset, INCARTDBImageWithFeatureDataset



def get_args_parser():
    parser = argparse.ArgumentParser('training', add_help=False)

    # Model parameters
    parser.add_argument('--model', default='default', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--batch_size_mae', default=256, type=int,
                        help='Per GPU batch size')
    parser.add_argument('--epochs_mae', default=50, type=int)
    parser.add_argument('--warmup_epochs_mae', type=int, default=0,
                        help='epochs to warmup LR')


    parser.add_argument('--batch_size_class', default=256, type=int,
                        help='Per GPU batch size')
    parser.add_argument('--epochs_class', default=20, type=int)
    parser.add_argument('--warmup_epochs_class', type=int, default=0,
                        help='epochs to warmup LR') 
    
    # Optimizer parameters
    parser.add_argument('--weight_decay_mae', type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--lr_mae', type=float, default=5e-4, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--min_lr_mae', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--weight_decay_class', type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--lr_class', type=float, default=5e-4, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--min_lr_class', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    

    parser.add_argument('--gpu', default='all', type=str,
                        help='single or all')

    # save parameters
    parser.add_argument('--contrun', action='store_true', help='flag continue from last run')
    parser.add_argument('--no_train_mae', action='store_false',  help='Train MAE', dest='train_mae')
    parser.add_argument('--no_train_class', action='store_false', help='Train Classifier', dest='train_class')
    parser.add_argument('--no_save_mae', action='store_false',  help='Save MAE model', dest='save_mae')
    parser.add_argument('--no_save_class', action='store_false', help='Save Classifier model', dest='save_class')

    return parser



if __name__ == "__main__":
    torch.manual_seed(42)

    parser = get_args_parser()
    args = parser.parse_args()

    dtype = torch.float32
    device_ids = [0, 2, 3]
    main_device = device_ids[0]
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{main_device}')
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f'SELECTED DEVICE: {device}')


    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((128, 128)), 
        transforms.ToTensor(),         
        ])


    # UNLABELED
    ptbxl_dir = 'data/physionet/ptbxl_full_224/'
    ptbxl_dataset = datasets.ImageFolder(root=ptbxl_dir, transform=transform)
    georgia_dir = 'data/physionet/georgia/'
    georgia_dataset = datasets.ImageFolder(root=georgia_dir, transform=transform)
    china_dir = 'data/physionet/china/'
    china_dataset = datasets.ImageFolder(root=china_dir, transform=transform)
    combined_unsupervised_train = torch.utils.data.ConcatDataset([ptbxl_dataset, georgia_dataset, china_dataset])
    trainset_un, testset_un, valset_un = torch.utils.data.random_split(combined_unsupervised_train, [190000, 25000, 17076])

    # LABELED
    # mitbih_ds11_dir = 'data/physionet/mitbih_224/DS11/'
    # mitbih_ds12_dir = 'data/physionet/mitbih_224/DS12/'
    # mitbih_ds2_dir = 'data/physionet/mitbih_224/DS2/'
    # mitbih_dataset_train = MITBIHImageWithFeatureDataset(root_dir=mitbih_ds11_dir, transform=transform)
    # mitbih_dataset_val = MITBIHImageWithFeatureDataset(root_dir=mitbih_ds12_dir, transform=transform)
    # mitbih_dataset_test = MITBIHImageWithFeatureDataset(root_dir=mitbih_ds2_dir, transform=transform) 
    # # print(len(mitbih_dataset_train))
    # # print(len(mitbih_dataset_val))
    # # print(len(mitbih_dataset_test))
    # incartdb_dir = 'data/physionet/incartdb_224/render/imgs/'
    # incartdb_dataset = INCARTDBImageWithFeatureDataset(root_dir=incartdb_dir, transform=transform)


    # trainset_sup = torch.utils.data.ConcatDataset([mitbih_dataset_train, incartdb_dataset])
    # valset_sup = mitbih_dataset_val
    # testset_sup = mitbih_dataset_test

    mses = []
    accuracies = []

    # MAE
    num_warmup_epochs_mae = args.warmup_epochs_mae
    num_epochs_mae = args.epochs_mae + num_warmup_epochs_mae

    # CLASSIFIER
    num_warmup_epochs_classifier = args.warmup_epochs_class
    num_epochs_classifier = args.epochs_class + num_warmup_epochs_classifier

    mae_losses_run = np.zeros((4, num_epochs_mae))
    mae_val_losses_run = np.zeros((4, num_epochs_mae))
    class_losses_run = np.zeros((4, num_epochs_classifier))
    class_val_losses_run = np.zeros((4, num_epochs_classifier))

    models = [ UNet(), AutoEncoder11_UN()] # ConvNext(),
    model_strs = ['unet', 'basic'] # 'convnext', 
    lr = [2e-5, 1e-4]

    now = datetime.now()
    run_dir = f'trained_models/model_comparison/RUN_{now.day}_{now.month}_{now.hour}_{now.minute}'
    for i, model in enumerate(models):
        os.makedirs(run_dir, exist_ok=True)
        model = nn.DataParallel(model, device_ids=device_ids).to(device)
        mae, mae_losses, mae_val_losses = train_mae(model=model, 
                                                    trainset=trainset_un,
                                                    valset=valset_un,
                                                    learning_rate=lr[i],
                                                    min_lr = args.min_lr_mae,
                                                    weight_decay = args.weight_decay_mae,
                                                    num_epochs=num_epochs_mae,
                                                    n_warmup_epochs=num_warmup_epochs_mae,
                                                    TRAIN_MAE=args.train_mae,
                                                    SAVE_MODEL_MAE=args.save_mae,
                                                    R=0,
                                                    batch_size=args.batch_size_mae,
                                                    fact=model_strs[i],
                                                    run_dir = run_dir,
                                                    contrun = args.contrun,
                                                    device = device)
        
        mae_losses_run[i,:] = mae_losses
        mae_val_losses_run[i,:] = mae_val_losses
        train_save_folder = f'{run_dir}/MAE_losses_{model_strs[i]}_train.npy'
        val_save_folder = f'{run_dir}/MAE_losses_{model_strs[i]}_val.npy'
        np.save(train_save_folder, mae_losses)
        np.save(val_save_folder, mae_val_losses)

        mses.append(eval_mae(mae, testset_un, device=device))

    
    print('\n'.join([f'{model_strs[i]} mse: {mses[i]}' for i in range(4)]))

    






