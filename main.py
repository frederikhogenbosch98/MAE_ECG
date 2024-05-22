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
from models.UNet import AutoEncoder56Unet, Classifier56Unet
from models.model_self_TD import AutoEncoder_self_TD, Classifier_self_TD
from models._11am_back import AutoEncoder11
from models._11am_down import AutoEncoder11_DOWN
from models.convnext import ConvNext
from models._11am_un import AutoEncoder11_UN, Classifier_UN

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
    
    # decomposition parameters
    parser.add_argument('--channel_start', default=16, type=int,
                        help='channels to start with')
    parser.add_argument('--rank_start', default=10, type=int)
    parser.add_argument('--rank_increment', type=int, default=5)
    parser.add_argument('--rank_stop', default='25', type=int,
                        help='stop rank')
    parser.add_argument('--gpu', default='all', type=str,
                        help='single or all')

    # save parameters
    parser.add_argument('--contrun', action='store_true', help='flag continue from last run')
    parser.add_argument('--no_train_mae', action='store_false',  help='Train MAE', dest='train_mae')
    parser.add_argument('--no_train_class', action='store_false', help='Train Classifier', dest='train_class')
    parser.add_argument('--no_save_mae', action='store_false',  help='Save MAE model', dest='save_mae')
    parser.add_argument('--no_save_class', action='store_false', help='Save Classifier model', dest='save_class')

    


    return parser


def train_mae(model, trainset, run_dir, device, min_lr=1e-5, valset=None, weight_decay=1e-4, MASK_RATIO=0.0, num_epochs=50, n_warmup_epochs=5, batch_size=128, learning_rate=5e-4, TRAIN_MAE=True, SAVE_MODEL_MAE=True, R=None, fact=None, contrun=False):
    now = datetime.now()

    if TRAIN_MAE:

        if contrun:
            model.load_state_dict(torch.load('trained_models/last/last_run.pth'))

        criterion = nn.MSELoss() # mean square error loss
        optimizer = torch.optim.Adam(model.parameters(),
                                    lr=1e-4, 
                                    weight_decay=1e-4)

        # optimizer = torch.optim.SGD(model.parameters(),
        #                             lr=learning_rate,
        #                             weight_decay=weight_decay)

        train_loader = torch.utils.data.DataLoader(trainset, 
                                                batch_size=batch_size, 
                                                shuffle=True)#, num_workers=4)
        num_iters = len(train_loader)
        if valset:
            val_loader = torch.utils.data.DataLoader(valset, 
                                        batch_size=batch_size, 
                                        shuffle=False)#, num_workers=2)

        # scheduler = CosineAnnealingwithWarmUp(optimizer, 
        #                                     n_warmup_epochs=n_warmup_epochs,
        #                                     warmup_lr=5e-5,
        #                                     start_lr=5e-4,
        #                                     lower_lr=2e-5,
        #                                     alpha=0.85,
        #                                     epoch_int=20,
        #                                     num_epochs=num_epochs)


            # early_stopper = EarlyStopper(patience=6)
        scheduler = StepLR(optimizer, step_size=20, gamma=0.1)
        outputs = []
        losses = []
        val_losses = []

        print(f"Start MAE training for {n_warmup_epochs} warm-up epochs and {num_epochs-n_warmup_epochs} training epochs")
        t_start = time.time()
        for epoch in range(num_epochs):
            running_loss = 0.0
            t_epoch_start = time.time()
            model.train()
            with tqdm.tqdm(train_loader, unit="batch", leave=False) as tepoch:
                for data in tepoch:
                    tepoch.set_description(f"epoch {epoch+1}")
                    img, _ = data
                    img = img.to(device)
                    unmasked_img = img
                    img = img.to(device)
                    recon = model(img)
                    optimizer.zero_grad()
                    loss = criterion(recon, unmasked_img)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                    tepoch.set_postfix(loss=running_loss / (batch_size*(epoch+1)))
                scheduler.step()

            if (epoch + 1) % 20 == 0 and epoch != 0 and SAVE_MODEL_MAE:
                torch.save(model.state_dict(), f'{run_dir}/MAE_RUN_{fact}_R{R}_{now.day}_{now.month}_{now.hour}_{now.minute}_epoch_{epoch}.pth')
                torch.save(model.state_dict(), 'trained_models/last/last_run.pth')

            if valset:
                model.eval()
                validation_loss = 0.0

                with torch.no_grad():
                    for data in val_loader:
                        imgs, _  = data
                        imgs = imgs.to(device)
                        outputs = model(imgs)
                        loss = criterion(outputs, imgs)
                        validation_loss += loss.item() * imgs.size(0)

                validation_loss /= len(val_loader.dataset)
                
                # if early_stopper.early_stop(validation_loss):             
                #     print(f"EARLY STOPPING AT EPOCH: {epoch}")
                #     break
            else:
                validation_loss = 0

            epoch_loss = running_loss / len(train_loader)
            losses.append(epoch_loss)
            val_losses.append(validation_loss)
            t_epoch_end = time.time()

            print('epoch {}: training loss: {:.7f}, val loss: {:.7f}, duration: {:.2f}s, lr: {:.2e}'.format(epoch+1, epoch_loss, validation_loss, t_epoch_end - t_epoch_start, optimizer.param_groups[0]['lr']))

           
        t_end = time.time()
        print(f"End of MAE training. Training duration: {np.round((t_end-t_start)/60.0,2)}m. Training loss: {loss}.")

        if SAVE_MODEL_MAE:
            save_folder = f'{run_dir}/MAE_RUN_{fact}_R{R}_{now.day}_{now.month}_{now.hour}_{now.minute}.pth'
            # save_folder = 'trained_models/tranpose_02_05_10am.pth'
            # save_folder = 'data/models_/MAE_TESTRUN.pth'
            torch.save(model.state_dict(), save_folder)
            torch.save(model.state_dict(), 'trained_models/last/last_run.pth')
            print(f'MAE model saved to {save_folder}')

        # plot_losses(epoch+1, losses)        
        print("\n")


    else:
        # model.load_state_dict(torch.load('data/models_mnist/MAE_TESTRUN.pth'))
        # model.load_state_dict(torch.load('trained_models/MAE_RUN_cp_R0_8_5_4_38.pth', map_location=torch.device('cpu')))
        # model.load_state_dict(torch.load('trained_models/last/last_run.pth'))
        model.load_state_dict(torch.load('trained_models/RUN_21_5_8_54/MAE_RUN_cp_R100_21_5_8_56_epoch_19.pth')) #uncompressed
        # model.load_state_dict(torch.load('trained_models/RUN_19_5_23_14/MAE_RUN_cp_R25_20_5_11_41.pth')) #R25
        # model.load_state_dict(torch.load('trained_models/RUN_14_5_22_16/MAE_RUN_default_R0_14_5_22_16.pth'))
        # model.load_state_dict(torch.load('trained_models/tranpose_02_05_10am.pth', map_location=torch.device('cpu')))
        print(f'dataset loaded')
        losses = np.zeros(num_epochs)
        val_losses = np.zeros(num_epochs)


    return model, losses, val_losses 



def eval_mae(model, testset, batch_size=128):
    model.to(device)
    model.eval()

    test_loader = torch.utils.data.DataLoader(testset, 
                                            batch_size=batch_size, 
                                            shuffle=True)

    mse_loss = nn.MSELoss(reduction='mean')
    total_loss = 0.0
    count = 0

    with torch.no_grad():  
        for inputs, _ in test_loader:
            inputs = inputs.to(device)  
            reconstructed = model(inputs)  
            loss = mse_loss(reconstructed, inputs)  
            total_loss += loss.item() 
            count += 1

    average_loss = np.round(total_loss / count, 6)

    print(f'Average MSE Loss on Test Set: {average_loss}')

    # data_list = []
    # target_list = []

    # for data, _ in testset:
    #     data_list.append(data.unsqueeze(0))

    # test_data_tensor = torch.cat(data_list, dim=0)

    # test_data_tensor = test_data_tensor.to(device)


    # x = model(test_data_tensor[0:64,:,:,:])
    # embedding = model.encoder(x)
    # e1 = embedding
    # recon = model.decoder(e1)
    # # print(recon.shape)
    # # print(recon)
    # for i in range(10):
    #     recon_cpu = recon[i,:,:,:]#.detach().numpy()
    #     recon_cpu = recon_cpu.cpu()
    #     print(test_data_tensor[i,:,:,:].shape)
    #     print(recon_cpu.shape)
    #     plotimg(test_data_tensor[i,:,:,:], recon_cpu)
        

    return average_loss

def train_classifier(classifier, trainset, run_dir, weight_decay = 1e-4, min_lr=1e-6, valset=None, num_epochs=25, n_warmup_epochs=5, learning_rate=5e-4, batch_size=128, TRAIN_CLASSIFIER=True, SAVE_MODEL_CLASSIFIER=True, R=None, fact=None):

    now = datetime.now()
    classifier.to(device)
    if TRAIN_CLASSIFIER:
        # for param in classifier.enc1.parameters():
        #     param.requires_grad = False
        # for param in classifier.enc2.parameters():
        #     param.requires_grad = False
        # for param in classifier.enc3.parameters():
        #     param.requires_grad = False
        # for param in classifier.pool1.parameters():
        #     param.requires_grad = False
        # for param in classifier.pool2.parameters():
        #     param.requires_grad = False
        # for param in classifier.pool3.parameters():
        #     param.requires_grad = False

        for param in classifier.encoder.parameters():
            param.requires_grad = False


        

        train_loader = torch.utils.data.DataLoader(trainset, 
                                            batch_size=batch_size, 
                                            shuffle=True, num_workers=2)
        

        # optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, classifier.parameters()),
        #                     lr=learning_rate,
        #                     weight_decay=weight_decay)


        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, classifier.parameters()),
                                    lr=1e-4, 
                                    weight_decay=1e-4)
        
        if valset:
            val_loader = torch.utils.data.DataLoader(valset, 
                                batch_size=batch_size, 
                                shuffle=False, num_workers=2)    


        # scheduler = CosineAnnealingwithWarmUp(optimizer, 
        #                                         n_warmup_epochs=n_warmup_epochs, 
        #                                         warmup_lr=1e-4, 
        #                                         start_lr=learning_rate, 
        #                                         lower_lr=min_lr,
        #                                         alpha=0.75, 
        #                                         epoch_int=20, 
        #                                         num_epochs=num_epochs)



        loss_function =  nn.CrossEntropyLoss().to(device)

        # early_stopper = EarlyStopper(patience=10, min_delta=0.0001)

        losses = []
        val_losses = []
        print(f"Start CLASSIFIER training for {n_warmup_epochs} warm-up epochs and {num_epochs-n_warmup_epochs} training epochs")        
        t_start = time.time()
        for epoch in range(num_epochs):
            running_loss = 0.0
            classifier.train()
            t_epoch_start = time.time()
            with tqdm.tqdm(train_loader, unit="batch", leave=False) as tepoch:
                for inputs, features, labels in tepoch: 
                    tepoch.set_description(f"epoch {epoch+1}")
                    inputs = inputs.to(device)
                    features = features.to(device)
                    labels = labels.to(device)
                    outputs = classifier(inputs, features)
                    loss = loss_function(outputs, labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                    tepoch.set_postfix(loss=running_loss / (batch_size*(epoch+1)))
            # scheduler.step()

            if epoch % 20 == 0 and epoch != 0:
                torch.save(classifier.state_dict(), f'{run_dir}/CLASSIFIER_RUN_{fact}_R{R}_{now.day}_{now.month}_{now.hour}_{now.minute}_epoch_{epoch}.pth')


            if valset:
                classifier.eval()  
                validation_loss = 0.0
                correct = 0
                total = 0
                with torch.no_grad():  
                    for data, features, target in val_loader:
                        data, features, target = data.to(device), features.to(device), target.to(device)
                        output = classifier(data, features)
                        loss = loss_function(output, target)
                        validation_loss += loss.item() * data.size(0)
                        _, predicted = torch.max(output.data, 1)
                        total += target.size(0)
                        correct += (predicted == target).sum().item()

                validation_loss /= len(val_loader.dataset)
                accuracy = correct / total * 100


                # if early_stopper.early_stop(validation_loss):             
                #     print(f"EARLY STOPPING AT EPOCH: {epoch}")
                #     break
            else:
                validation_loss = 0


            epoch_loss = running_loss / len(train_loader)
            t_epoch_end = time.time()
            epoch_loss = running_loss / len(train_loader)

            print('epoch {}: training loss: {:.7f}, val loss: {:.7f}, accuracy: {:.2f}, duration: {:.2f}s, lr: {:.2e}'.format(epoch+1, epoch_loss, validation_loss, accuracy, t_epoch_end - t_epoch_start, optimizer.param_groups[0]['lr']))

            losses.append(epoch_loss)
            val_losses.append(validation_loss)


        t_end = time.time()
        print(f"End of CLASSIFIER training. Training duration: {np.round((t_end-t_start)/60.0,2)}m. final loss: {loss}.")

        if SAVE_MODEL_CLASSIFIER:
            save_folder = f'{run_dir}/CLASSIFIER_RUN_{fact}_R{R}_{now.day}_{now.month}_{now.hour}_{now.minute}.pth'
            torch.save(classifier.state_dict(), save_folder)
            print(f'classifier model saved to {save_folder}')

        print("\n")
    else:
        # classifier.load_state_dict(torch.load('data/models_mnist/CLASSIFIER_MR_02.pth'))
        print('classifier model loaded')
        classifier.load_state_dict(torch.load('trained_models/RUN_16_5_22_32/CLASSIFIER_RUN_default_R0_17_5_1_55.pth', map_location=torch.device('cpu')))
        losses = np.zeros(num_epochs)
        val_losses = np.zeros(num_epochs)

    return classifier, losses, val_losses


def eval_classifier(model, testset, batch_size=128):

    model.to(device)
    model.eval()

    test_loader = torch.utils.data.DataLoader(testset, 
                                            batch_size=batch_size, 
                                            shuffle=True)
    correct = 0
    total = 0
    test_accuracy = []
    with torch.no_grad():
        for images, features, labels in test_loader:
            images, features, labels = images.to(device), features.to(device), labels.to(device)
            outputs = model(images, features)
            # _, predicted = torch.max(F.softmax(outputs, dim=1).data, 1)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            test_accuracy.append((predicted == labels).sum().item() / predicted.size(0))
        
    print(f'acc: {np.mean(test_accuracy)}')

    accuracy = 100 * correct / total
    print(f'Accuracy: {np.round(accuracy,3)}%')

    # ## MNIST
    # for idx, (images, labels) in enumerate(test_loader):
    #     images, labels = images.to(device), labels.to(device)
    #     x = model(images)
    #     _, predicted = torch.max(x.data, 1)
    #     images = images.cpu()
    #     plot_single_img(images, 0)
    #     print(f'prediction: {predicted[0].item()} ----- label: {labels[0].item()}')
    #     if idx == 10:
    #         break

    return np.mean(test_accuracy)



if __name__ == "__main__":
    torch.manual_seed(42)

    args = get_args_parser()
    args = args.parse_args()


    dtype = torch.float32
    if torch.cuda.is_available():
        device = torch.device("cuda")
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


    ptbxl_dir = 'data/physionet/ptbxl_full_224/'
    ptbxl_dataset = datasets.ImageFolder(root=ptbxl_dir, transform=transform)
    georgia_dir = 'data/physionet/georgia/'
    georgia_dataset = datasets.ImageFolder(root=georgia_dir, transform=transform)
    china_dir = 'data/physionet/china/'
    china_dataset = datasets.ImageFolder(root=china_dir, transform=transform)
    combined_unsupervised_train = torch.utils.data.ConcatDataset([ptbxl_dataset, georgia_dataset, china_dataset])
    # print(len(combined_unsupervised_train))
    # trainset_un, testset_un, valset_un = torch.utils.data.random_split(ptbxl_dataset, [40000, 10000, 2656])    
    # trainset_un, testset_un, valset_un = torch.utils.data.random_split(combined_unsupervised_train, [10000, 4000, 2926])
    trainset_un, testset_un, valset_un = torch.utils.data.random_split(combined_unsupervised_train, [190000, 25000, 17076])

    mitbih_ds11_dir = 'data/physionet/mitbih_224/DS11/'
    mitbih_ds12_dir = 'data/physionet/mitbih_224/DS12/'
    mitbih_ds2_dir = 'data/physionet/mitbih_224/DS2/'
    mitbih_dataset_train = MITBIHImageWithFeatureDataset(root_dir=mitbih_ds11_dir, transform=transform)
    mitbih_dataset_val = MITBIHImageWithFeatureDataset(root_dir=mitbih_ds12_dir, transform=transform)
    mitbih_dataset_test = MITBIHImageWithFeatureDataset(root_dir=mitbih_ds2_dir, transform=transform) 
    # print(len(mitbih_dataset_train))
    # print(len(mitbih_dataset_val))
    # print(len(mitbih_dataset_test))
    incartdb_dir = 'data/physionet/incartdb_224/render/imgs/'
    incartdb_dataset = INCARTDBImageWithFeatureDataset(root_dir=incartdb_dir, transform=transform)
    print(f'incartdb num beats: {len(incartdb_dataset)}')


    trainset_sup = torch.utils.data.ConcatDataset([mitbih_dataset_train, incartdb_dataset])
    # trainset_sup = mitbih_dataset_train 
    valset_sup = mitbih_dataset_val
    testset_sup = mitbih_dataset_test

    R_LIST = np.arange(args.rank_start, args.rank_stop + args.rank_increment, args.rank_increment).tolist()

    if args.model == 'cp':
        fact_list = ['cp']
    elif args.model == 'tucker':
        fact_list = ['tucker']
    elif args.model == 'both':
        fact_list = ['cp', 'tucker']
    elif args.model == 'default':
        fact_list = ['default']
        R_LIST = [0]
    elif args.model == 'combine':
        fact_list = ['default', 'cp']

    else:
        fact_list = ['default']
        R_LIST = [0]
    


    channels = [args.channel_start, 2*args.channel_start, 4*args.channel_start, 8*args.channel_start]
    print(channels)
    mses = []
    accuracies = []

    # MAE
    num_warmup_epochs_mae = args.warmup_epochs_mae
    num_epochs_mae = args.epochs_mae + num_warmup_epochs_mae

    # CLASSIFIER
    num_warmup_epochs_classifier = args.warmup_epochs_class
    num_epochs_classifier = args.epochs_class + num_warmup_epochs_classifier

    mae_losses_run = np.zeros((len(R_LIST), num_epochs_mae))
    mae_val_losses_run = np.zeros((len(R_LIST), num_epochs_mae))
    class_losses_run = np.zeros((len(R_LIST), num_epochs_classifier))
    class_val_losses_run = np.zeros((len(R_LIST), num_epochs_classifier))
    print(f'for R values: {R_LIST}')

    now = datetime.now()
    run_dir = f'trained_models/RUN_{now.day}_{now.month}_{now.hour}_{now.minute}'
    for fact in fact_list:
        if fact == 'default':
            R_LIST = [0]
        elif fact == 'cp':
            R_LIST = R_LIST
        print(f'for R values: {R_LIST}')
        for i, R in enumerate(R_LIST):
            if R == 100:
                continue
            os.makedirs(run_dir, exist_ok=True)

            print(f'fact: {fact}, R: {R}')
            factorization=fact

            if fact == 'default':
                if args.gpu == 'all':
                    # mae = nn.DataParallel(AutoEncoder56Unet()).to(device)
                    # mae = nn.DataParallel(AutoEncoder56(channels=channels)).to(device)
                    # mae = nn.DataParallel(AutoEncoder11_UN()).to(device)
                    mae = AutoEncoder11_UN().to(device)
                    # mae = nn.DataParallel(UNet()).to(device)
                else:
                    mae = AutoEncoder56().to(device)
            else:
                if args.gpu == 'all':
                    # mae = nn.DataParallel(AutoEncoder56_TD(R=R, in_channels=1, factorization=fact)).to(device)
                    mae = nn.parallel.DistributedDataParallel(AutoEncoder11(R=R, in_channels=1), device_ids=[0, 1, 2]).to(device)
                else:
                   mae = AutoEncoder11(R=R, in_channels=1).to(device)

                

            current_pams = count_parameters(mae)
            print(f'num params: {current_pams}')


            mae, mae_losses, mae_val_losses = train_mae(model=mae, 
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
                            device = device)
            
            mae_losses_run[i,:] = mae_losses
            mae_val_losses_run[i,:] = mae_val_losses

            mses.append(eval_mae(mae, testset_un))
             
            num_classes = 5
            if args.model == 'default':
                classifier = Classifier_UN(autoencoder=mae.module, in_features=2048, out_features=num_classes).to(device)
                # print('hello')

                # classifier = Classifier56Unet(autoencoder=mae.module, in_features=2048, out_features=num_classes).to(device)
                    # classifier = UClassifier(autoencoder=mae.module, out_features=num_classes).to(device)
            else:
                classifier = Classifier_UN(autoencoder=mae, in_features=2048, out_features=num_classes).to(device)
                # classifier = Classifier_self_TD(autoencoder=mae.module, in_features=2048, out_features=num_classes).to(device)

            classifier, class_losses, class_val_losses = train_classifier(classifier=classifier, 
                                        trainset=trainset_sup, 
                                        valset=valset_sup, 
                                        num_epochs=num_epochs_classifier, 
                                        n_warmup_epochs=num_warmup_epochs_classifier, 
                                        learning_rate=args.lr_class,
                                        min_lr = args.min_lr_class,
                                        weight_decay = args.weight_decay_class,
                                        batch_size=args.batch_size_class, 
                                        TRAIN_CLASSIFIER=args.train_class, 
                                        SAVE_MODEL_CLASSIFIER=args.save_class,
                                        R=R,
                                        fact=fact,
                                        run_dir = run_dir)
            
            accuracies.append(eval_classifier(classifier, testset_sup))
            class_losses_run[i,:] = class_losses
            class_val_losses_run[i,:] = class_val_losses

            # print(count_parameters(classifier))

            mae_losses_run[i,:] = mae_losses
            class_losses_run[i,:] = class_losses


    
            mae_save_folder = f'{run_dir}/MAE_losses_{now.day}_{now.month}_{now.hour}_{now.minute}_{fact}_{R}.npy'
            class_save_folder = f'{run_dir}/CLASS_losses_{now.day}_{now.month}_{now.hour}_{now.minute}_{fact}_{R}.npy'
            np.save(mae_save_folder, mae_losses_run)
            np.save(class_save_folder, class_losses_run)
            np.save(f'{run_dir}/accuracies_RUN_{now.day}_{now.month}_{now.hour}_{now.minute}_{fact}_{R}.npy', np.array(accuracies))
            np.save(f'{run_dir}/MSES_RUN_{now.day}_{now.month}_{now.hour}_{now.minute}_{fact}_R.npy', np.array(mses))
            np.savetxt(f'{run_dir}/summary_{fact}_{R}.txt', accuracies, fmt='%f')

