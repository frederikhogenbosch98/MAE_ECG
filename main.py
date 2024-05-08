from torchvision import datasets, transforms
import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import ImageFolder 
from models.model_56x56_TD import AutoEncoder56_CPD, Classifier56_CPD
from models.model_56x56 import AutoEncoder56, Classifier56
import matplotlib.pyplot as plt
import time
import numpy as np
from print_funs import plot_losses, plotimg, plot_single_img, count_parameters
from nn_funcs import CosineAnnealingwithWarmUp, EarlyStopper
from torch.optim.lr_scheduler import StepLR
from ptflops import get_model_complexity_info
import math
from models.resnet50 import ResNet
from datetime import datetime
import platform



def train_mae(model, trainset, valset=None, MASK_RATIO=0.0, num_epochs=50, n_warmup_epochs=5, batch_size=128, learning_rate=5e-4, TRAIN_MAE=True, SAVE_MODEL_MAE=True, R=None, fact=None, p=4):
    # torch.manual_seed(42)
    if TRAIN_MAE:

        criterion = nn.MSELoss() # mean square error loss
        # optimizer = torch.optim.Adam(model.parameters(),
        #                             lr=learning_rate, 
        #                             weight_decay=1e-4)

        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=learning_rate,
                                    weight_decay=1e-4)

        train_loader = torch.utils.data.DataLoader(trainset, 
                                                batch_size=batch_size, 
                                                shuffle=True)#, num_workers=4)
        num_iters = len(train_loader)
        if valset:
            val_loader = torch.utils.data.DataLoader(valset, 
                                        batch_size=batch_size, 
                                        shuffle=False)#, num_workers=2)
            # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=0.0001)
            # scheduler  = StepLR(optimizer, step_size=10, gamma=0.95) 
            scheduler = CosineAnnealingwithWarmUp(optimizer, 
                                                n_warmup_epochs=n_warmup_epochs,
                                                warmup_lr=1e-5,
                                                start_lr=5e-4,
                                                lower_lr=1e-5,
                                                alpha=0.5,
                                                epoch_int=20,
                                                num_epochs=num_epochs)
            # scheduler.print_seq()
            # lambda_lr = lambda epoch: 0.85 ** (epoch / 10)

            # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_lr)

            # early_stopper = EarlyStopper(patience=6)

        outputs = []
        losses = []
        val_losses = []

        print(f"Start MAE training for {n_warmup_epochs} warm-up epochs and {num_epochs-n_warmup_epochs} training epochs")
        t_start = time.time()
        for epoch in range(num_epochs):
            running_loss = 0.0
            t_epoch_start = time.time()
            model.train()
            for i, data in enumerate(train_loader):
                img, _ = data
                img = img.to(device)
                unmasked_img = img
                # plot_single_img(img.cpu(), 7)
                # if MASK_RATIO != 0:
                #     img = mask(img, MASK_RATIO, p)
                img = img.to(device)
                recon = model(img)
                optimizer.zero_grad()
                loss = criterion(recon, unmasked_img)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                # scheduler.step(epoch + i / num_iters)
            scheduler.step()

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

            print('epoch {}: average loss: {:.7f}, val loss: {:.7f}, duration: {:.2f}s, lr: {:.2e}'.format(epoch+1, epoch_loss, validation_loss, t_epoch_end - t_epoch_start, optimizer.param_groups[0]['lr']))
            losses.append(epoch_loss)

           
        t_end = time.time()
        print(f"End of MAE training. Training duration: {np.round((t_end-t_start),2)}s. Training loss: {loss}.")

        now = datetime.now()
        if SAVE_MODEL_MAE:
            save_folder = f'trained_models/MAE_RUN_{fact}_R{R}_{now.day}_{now.month}_{now.hour}_{now.minute}.pth'
            # save_folder = 'trained_models/tranpose_02_05_10am.pth'
            # save_folder = 'data/models_/MAE_TESTRUN.pth'
            torch.save(model.state_dict(), save_folder)
            print(f'mae model saved to {save_folder}')

        # plot_losses(epoch+1, losses)        
        print("\n")
        print("\n")

        return model, losses, val_losses 

    else:
        # model.load_state_dict(torch.load('data/models_mnist/MAE_TESTRUN.pth'))
        model.load_state_dict(torch.load('trained_models/MAE_RUN_cp_R0_8_5_4_38.pth', map_location=torch.device('cpu')))
        # model.load_state_dict(torch.load('trained_models/tranpose_02_05_10am.pth', map_location=torch.device('cpu')))
        print(f'dataset loaded')


        return model



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

    data_list = []
    target_list = []

    for data, _ in testset:
        data_list.append(data.unsqueeze(0))

    test_data_tensor = torch.cat(data_list, dim=0)

    test_data_tensor = test_data_tensor.to(device)


    x = model(test_data_tensor[0:64,:,:,:])
    embedding = model.encoder(x)
    e1 = embedding
    recon = model.decoder(e1)
    # print(recon.shape)
    # print(recon)
    for i in range(10):
        recon_cpu = recon[i,:,:,:]#.detach().numpy()
        recon_cpu = recon_cpu.cpu()
        print(test_data_tensor[i,:,:,:].shape)
        print(recon_cpu.shape)
        plotimg(test_data_tensor[i,:,:,:], recon_cpu)
        

    return average_loss

def train_classifier(classifier, trainset, valset=None, num_epochs=25, n_warmup_epochs=5, learning_rate=1e-4, batch_size=128, TRAIN_CLASSIFIER=True, SAVE_MODEL_CLASSIFIER=True, R=None, fact=None):

    classifier.to(device)
    if TRAIN_CLASSIFIER:
        for param in classifier.encoder.parameters():
            param.requires_grad = False
        

        train_loader = torch.utils.data.DataLoader(trainset, 
                                            batch_size=batch_size, 
                                            shuffle=True, num_workers=2)
        
        num_iters = len(train_loader)

        # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, classifier.parameters()), lr=learning_rate, weight_decay=1e-5)
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, classifier.parameters()),
                            lr=learning_rate,
                            weight_decay=1e-4)
        if valset:
            val_loader = torch.utils.data.DataLoader(valset, 
                                batch_size=batch_size, 
                                shuffle=False, num_workers=2)    


            # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=0.0001)
            # scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
            # scheduler =     CosineAnnealingwithWarmUp(optimizer, 
            #                             n_warmup_epochs=n_warmup_epochs,
            #                             warmup_lr=5e-4,
            #                             start_lr=5e-4,
            #                             lower_lr=8e-6,
            #                             alpha=0.5,
            #                             epoch_int=20,
            #                             num_epochs=num_epochs)
            scheduler = CosineAnnealingwithWarmUp(optimizer, 
                                                  n_warmup_epochs=n_warmup_epochs, 
                                                  warmup_lr=1e-4, 
                                                  start_lr=5e-4, 
                                                  lower_lr=1e-6,
                                                  alpha=0.75, 
                                                  epoch_int=20, 
                                                  num_epochs=num_epochs)

            # scheduler.print_seq()


        # optimizer = torch.optim.Adam(classifier.parameters(), lr=2e-3, weight_decay=0.05)
        # scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

        # lambda_lr = lambda epoch: 0.85 ** (epoch / 10)

        # Create the scheduler
        # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_lr)
        loss_function =  nn.CrossEntropyLoss().to(device)

        early_stopper = EarlyStopper(patience=10, min_delta=0.0001)

        losses = []
        val_losses = []
        print(f"Start CLASSIFIER training for {n_warmup_epochs} warm-up epochs and {num_epochs-n_warmup_epochs} training epochs")        
        t_start = time.time()
        for epoch in range(num_epochs):
            running_loss = 0.0
            classifier.train()
            t_epoch_start = time.time()
            for i, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                # print(labels[0:64])
                # plot_single_img(inputs.cpu(), 10)
                # print(labels[10])
                outputs = classifier(inputs)
                # print(F.softmax(outputs, dim=1))
                # print(outputs.shape)
                # print(labels.shape)
                loss = loss_function(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            scheduler.step()


            if valset:
                classifier.eval()  
                validation_loss = 0.0
                correct = 0
                total = 0
                with torch.no_grad():  
                    for data, target in val_loader:
                        data, target = data.to(device), target.to(device)
                        output = classifier(data)
                        loss = loss_function(output, target)
                        validation_loss += loss.item() * data.size(0)
                        _, predicted = torch.max(output.data, 1)
                        total += target.size(0)
                        correct += (predicted == target).sum().item()

                validation_loss /= len(val_loader.dataset)
                accuracy = correct / total * 100
                # scheduler.step(validation_loss)
                if early_stopper.early_stop(validation_loss):             
                    print(f"EARLY STOPPING AT EPOCH: {epoch}")
                    break
            else:
                validation_loss = 0
            epoch_loss = running_loss / len(train_loader)
            t_epoch_end = time.time()
            # for  param_group in optimizer.param_groups:
            #     lr = param_group['lr']
            t_epoch_end = time.time()
            # print('epoch {}: loss: {:.4f} duration: {:.2f}s'.format(epoch+1, float(loss), float(t_epoch_end-t_epoch_start)))
            epoch_loss = running_loss / len(train_loader)
            print('epoch {}: average loss: {:.7f}, val loss: {:.7f}, accuracy: {:.2f}, duration: {:.2f}s, lr: {:.2e}'.format(epoch+1, epoch_loss, validation_loss, accuracy, t_epoch_end - t_epoch_start, optimizer.param_groups[0]['lr']))
            losses.append(epoch_loss)
            val_losses.append(validation_loss)
            # scheduler.step()
        t_end = time.time()
        print(f"End of CLASSIFIER training. Training duration: {np.round((t_end-t_start),2)}s. final loss: {loss}.")
        now = datetime.now()
        if SAVE_MODEL_CLASSIFIER:
            # save_folder = f'/models_nightrun/RUN_{fact}_R{R}.pth'
            save_folder = f'trained_models/CLASSIFIER_RUN_{fact}_R{R}_{now.day}_{now.month}_{now.hour}_{now.minute}.pth'
            torch.save(classifier.state_dict(), save_folder)
            print(f'classifier model saved to {save_folder}')

        print("\n")
        # print("\n")
    else:
        # classifier.load_state_dict(torch.load('data/models_mnist/CLASSIFIER_MR_02.pth'))
        classifier.load_state_dict(torch.load('trained_models/CLASSIFIER_RUN_cp_R20_6_5_20_31.pth', map_location=torch.device('cpu')))

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
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
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
        transforms.Resize((112,112)), 
        transforms.ToTensor(),         
        ])

    ptbxl_dir = 'data/physionet/ptbxl_full/'
    ptbxl_dataset = datasets.ImageFolder(root=ptbxl_dir, transform=transform)
    print(len(ptbxl_dataset))

    trainset_un, testset_un, valset_un = torch.utils.data.random_split(ptbxl_dataset, [40000, 10000, 2656])    
    # trainset_un, testset_un, valset_un = torch.utils.data.random_split(ptbxl_dataset, [100000, 20000, 10794])    

    # mitbih_ds1_dir = 'data/physionet/mitbih/DS1/'
    # mitbih_ds2_dir = 'data/physionet/mitbih/DS2/'
    # mitbih_dataset_train = datasets.ImageFolder(root=mitbih_ds1_dir, transform=transform)
    # mitbih_dataset_test = datasets.ImageFolder(root=mitbih_ds2_dir, transform=transform) 

    # incartdb = 'data/physionet/incartdb/render/imgs'
    # incartdb_dataset = datasets.ImageFolder(root=incartdb, transform=transform)

    # # print(len(mitbih_dataset_train))
    # # trainset_un, testset_un, valset_un = torch.utils.data.random_split(dataset, [13000, 6000, 2003])
    # # trainset_sup, testset_sup, valset_sup = torch.utils.data.random_split(dataset, [11000, 7002, 3001])
    # combined_dataset_train = torch.utils.data.ConcatDataset([mitbih_dataset_train, incartdb_dataset])
    # print(len(combined_dataset_train))
    # trainset_sup, valset_sup = torch.utils.data.random_split(combined_dataset_train, [47343, 5000])
    # testset_sup = mitbih_dataset_test


    MASK_RATIO = 0
    fact_list = ['cp']#, 'tucker']
    R_LIST = [0]
    mses = []
    accuracies = []

    # MAE
    num_warmup_epochs_mae = 5
    num_epochs_mae = 100 + num_warmup_epochs_mae

    # CLASSIFIER
    num_warmup_epochs_classifier = 0
    num_epochs_classifier = 0 + num_warmup_epochs_classifier

    mae_losses_run = np.zeros((len(R_LIST), num_epochs_mae))
    class_losses_run = np.zeros((len(R_LIST), num_epochs_classifier))

    for fact in fact_list:
        for i, R in enumerate(R_LIST):
            print(f'fact: {fact}, R: {R}')
            factorization=fact
            # encoder = Encoder56_CPD(R, factorization=factorization).to(device)
            # decoder = Decoder56_CPD(R, factorization=factorization).to(device)
            # mae = AutoEncoder56_CPD(R, in_channels=1, channels=[16, 32, 64, 128], depths=[3, 3, 9, 3]).to(device)
            # mae = AutoEncoder56_CPD(R, factorization=fact, in_channels=1).to(device)
            mae = AutoEncoder56().to(device)

            current_pams = count_parameters(mae)
            print(f'num params: {current_pams}')


            mae = train_mae(model=mae, 
                            trainset=trainset_un,
                            valset=valset_un,
                            MASK_RATIO=MASK_RATIO,
                            num_epochs=num_epochs_mae,
                            n_warmup_epochs=num_warmup_epochs_mae,
                            TRAIN_MAE=True,
                            SAVE_MODEL_MAE=True,
                            R=R,
                            fact=fact)

            mses.append(eval_mae(mae, testset_un))
            
            # num_classes = 5
            # # classifier = Classifier56_CPD(autoencoder=mae, in_features=2048, out_features=num_classes).to(device)
            # classifier = Classifier56(autoencoder=mae, in_features=2048, out_features=num_classes).to(device)

            # classifier, class_losses, class_val_losses = train_classifier(classifier=classifier, 
            #                             trainset=trainset_sup, 
            #                             valset=valset_sup, 
            #                             num_epochs=num_epochs_classifier, 
            #                             n_warmup_epochs=num_warmup_epochs_classifier, 
            #                             learning_rate=1e-3,
            #                             batch_size=256, 
            #                             TRAIN_CLASSIFIER=True, 
            #                             SAVE_MODEL_CLASSIFIER=True,
            #                             R=R,
            #                             fact=fact)

            # print(count_parameters(classifier))
            # accuracies.append(eval_classifier(classifier, testset_sup))

            # mae_losses_run[i,:] = mae_losses
            # class_losses_run[i,:] = class_losses


    # now = datetime.now()
    # mae_save_folder = f'trained_models/numpy_arrays/MAE_RUN_{now.day}_{now.month}_{now.hour}_{now.minute}.npy'
    # class_save_folder = f'trained_models/numpy_arrays/CLASS_RUN_{now.day}_{now.month}_{now.hour}_{now.minute}.npy'
    # np.save(mae_save_folder, mae_losses_run)
    # np.save(class_save_folder, class_losses_run)


    # print(mses)
    # print(accuracies)


            



