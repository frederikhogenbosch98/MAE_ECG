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
from models.UNet import UNet, ClassifierUnet 
from models.model_self_TD import AutoEncoder_self_TD, Classifier_self_TD
from models._11am_corr import AutoEncoder11
from models._11am_down import AutoEncoder11_DOWN
from models.convnext import ConvNext
from models._11am_un import AutoEncoder11_UN, Classifier_UN

from main import train_mae, train_classifier, eval_mae, eval_classifier, reconstruct_img
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
    parser.add_argument('--num_runs', type=int, default=3,
                        help='Number of repeat runs')


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

    parser.add_argument('--rank', default=25, type=int,
                        help='rank')

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
    device_ids = [0,  2, 3]
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
    # ptbxl_dir = 'data/physionet/ptbxl_full_224/'
    ptbxl_dir = 'data/physionet/ptbxl_wide/'
    ptbxl_dataset = datasets.ImageFolder(root=ptbxl_dir, transform=transform)
    # georgia_dir = 'data/physionet/georgia/'
    georgia_dir = 'data/physionet/georgia_wide/'
    georgia_dataset = datasets.ImageFolder(root=georgia_dir, transform=transform)
    # china_dir = 'data/physionet/china/'
    china_dir = 'data/physionet/china_wide/'
    china_dataset = datasets.ImageFolder(root=china_dir, transform=transform)
    combined_unsupervised_train = torch.utils.data.ConcatDataset([ptbxl_dataset, georgia_dataset, china_dataset])
    trainset_un, testset_un, valset_un = torch.utils.data.random_split(combined_unsupervised_train, [190000, 25000, 17077])

    # LABELED
    mitbih_ds11_dir = 'data/physionet/mitbih_rr/DS11/'
    mitbih_ds12_dir = 'data/physionet/mitbih_rr/DS12/'
    mitbih_ds2_dir = 'data/physionet/mitbih_rr/DS2/'
    mitbih_dataset_train = MITBIHImageWithFeatureDataset(root_dir=mitbih_ds11_dir, transform=transform)
    mitbih_dataset_val = MITBIHImageWithFeatureDataset(root_dir=mitbih_ds12_dir, transform=transform)
    mitbih_dataset_test = MITBIHImageWithFeatureDataset(root_dir=mitbih_ds2_dir, transform=transform) 
    # print(len(mitbih_dataset_train))
    # print(len(mitbih_dataset_val))
    # print(len(mitbih_dataset_test))
    incartdb_dir = 'data/physionet/incartdb_rr/'
    incartdb_dataset = INCARTDBImageWithFeatureDataset(root_dir=incartdb_dir, transform=transform)
    # print(f'incartdb num beats: {len(incartdb_dataset)}')


    trainset_sup = torch.utils.data.ConcatDataset([mitbih_dataset_train, incartdb_dataset])
    # trainset_sup = mitbih_dataset_train 
    valset_sup = mitbih_dataset_val
    testset_sup = mitbih_dataset_test

    mega_mses = []
    accuracies = []

    # MAE
    num_warmup_epochs_mae = args.warmup_epochs_mae
    num_epochs_mae = args.epochs_mae + num_warmup_epochs_mae

    # CLASSIFIER
    num_warmup_epochs_classifier = args.warmup_epochs_class
    num_epochs_classifier = args.epochs_class + num_warmup_epochs_classifier

    num_params_uncompressed = 9411649#2355745

    
    R_LIST = [0, args.rank] #5, 10, 15, 20, 25, 35, 50, 75, 100,125,  
    # R_LIST = [args.rank]
    CLASSIFY = True
    fact = 'cp'

    now = datetime.now()
    run_dir = f'trained_models/compressed/RUN_{now.day}_{now.month}_{now.hour}_{now.minute}_adam_check'
    os.makedirs(f'{run_dir}/', exist_ok=True)
    for i, R in enumerate(R_LIST):
        print(f'RUN R: {R}')
        if R == 0:
            model = AutoEncoder11_UN(channels=[16,32,64,128])
            num_params_uncompressed = count_parameters(model)
        else:
            model = AutoEncoder11(R=R, factorization='cp', channels=[16,32,64,128])
        model = nn.DataParallel(model, device_ids=device_ids).to(device)
        mses = []
        current_pams = count_parameters(model)
        print(f'num params: {current_pams}')
        comp_ratio = num_params_uncompressed/current_pams
        print(f'compression ratio: {comp_ratio}')

        for j in range(args.num_runs):
            os.makedirs(f'{run_dir}/R_{R}/{j}', exist_ok=True)
            mae, mae_losses, mae_val_losses, epoch_time = train_mae(model=model, 
                                                        trainset=trainset_un,
                                                        valset=valset_un,
                                                        learning_rate=args.lr_mae,
                                                        min_lr = args.min_lr_mae,
                                                        weight_decay = args.weight_decay_mae,
                                                        num_epochs=num_epochs_mae,
                                                        n_warmup_epochs=num_warmup_epochs_mae,
                                                        TRAIN_MAE=args.train_mae,
                                                        SAVE_MODEL_MAE=args.save_mae,
                                                        R=R,
                                                        batch_size=args.batch_size_mae,
                                                        fact=fact,
                                                        run_dir = run_dir,
                                                        contrun = args.contrun,
                                                        device = device,
                                                        step_size=15)
            
            train_save_folder = f'{run_dir}/R_{R}/{j}/MAE_losses_train.npy'
            val_save_folder = f'{run_dir}/R_{R}/{j}/MAE_losses_val.npy'
            np.save(train_save_folder, mae_losses)
            np.save(val_save_folder, mae_val_losses)

            mses.append(eval_mae(mae, testset_un,R,device=device))

            # if j == 0:
            #     reconstruct_img(mae, R=R, run_dir=run_dir)

        #     if CLASSIFY:
        #         num_classes = 5
        #         if args.model == 'default':
        #             classifier = Classifier_UN(autoencoder=mae.module, in_features=2048, out_features=num_classes)
        #             # classifier = ClassifierUnet(autoencoder=mae.module, in_features=2048, out_features=num_classes)


        #         if args.gpu == 'all':
        #             classifier = nn.DataParallel(classifier, device_ids=device_ids).to(device) 

        #         classifier, class_losses, class_val_losses = train_classifier(classifier=classifier, 
        #                                     trainset=trainset_sup, 
        #                                     valset=valset_sup, 
        #                                     num_epochs=num_epochs_classifier, 
        #                                     n_warmup_epochs=num_warmup_epochs_classifier, 
        #                                     learning_rate=args.lr_class,
        #                                     min_lr = args.min_lr_class,
        #                                     weight_decay = args.weight_decay_class,
        #                                     batch_size=args.batch_size_class, 
        #                                     TRAIN_CLASSIFIER=args.train_class, 
        #                                     SAVE_MODEL_CLASSIFIER=args.save_class,
        #                                     R=R,
        #                                     fact='cp',
        #                                     run_dir = run_dir,
        #                                     device = device,
        #                                     testset=testset_sup)

        #         accuracy = eval_classifier(classifier, testset_sup, device=device)
        #         accuracies.append(accuracy)


        # mega_mses.append(np.mean(mses))

        # save_str = f'''compression ratio: {comp_ratio} | num_params: {current_pams} | R: {R} |
        #             fact: {fact} | avg accuracy: {np.mean(accuracies)} | avg test mse: {np.mean(mses)} 
        #             (last) epoch time: {epoch_time}'''
        # with open(f"{run_dir}/run_info_{R}.txt", "w") as file:
        #     file.write(save_str)

    print(mega_mses)

    






