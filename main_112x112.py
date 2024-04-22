from torchvision import datasets, transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import ImageFolder 
from models.model_112x112 import AutoEncoder112, Classifier112
import matplotlib.pyplot as plt
import time
import numpy as np
from print_funs import plot_losses, plotimg, plot_single_img, count_parameters
from torch.optim.lr_scheduler import StepLR



class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def apply_mask(x, ratio, p):
    x = x.permute(0,5,1,2,3,4)
    rand_mask = torch.rand(x.shape[0], x.shape[1], x.shape[2], x.shape[3]) < ratio
    rand_mask = rand_mask.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, 1, p, p)
    x = torch.where(rand_mask, torch.zeros_like(x), x) 
    return x
 

def patchify(imgs, ratio, p=4):
    """
    imgs: (N, 3, H, W)
    x: (N, L, patch_size**2 *3)
    """
    assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

    h = w = imgs.shape[2] // p
    x = imgs.reshape(shape=(imgs.shape[0], 1, h, p, w, p))
    x = torch.einsum('nchpwq->nhwpqc', x)
    x = apply_mask(x, ratio, p)
    x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 1))
    return x


def unpatchify(x, p=4):
    """
    x: (N, L, patch_size**2 *3)
    imgs: (N, 3, H, W)
    """
    h = w = int(x.shape[1]**.5)
    assert h * w == x.shape[1]
    
    x = x.reshape(shape=(x.shape[0], h, w, p, p, 1))
    x = torch.einsum('nhwpqc->nchpwq', x)
    imgs = x.reshape(shape=(x.shape[0], 1, h * p, h * p))
    return imgs


def mask(batch, ratio, p):
    x = patchify(batch.cpu(), ratio, p)
    imgs = unpatchify(x, p)
    return imgs
    # plt.subplot(2,1,1)
    # plt.imshow(batch[8,0,:,:].cpu().detach().numpy(), cmap="gray")
    # plt.subplot(2,1,2)
    # plt.imshow(imgs[8,0,:,:].cpu().detach().numpy(), cmap="gray")
    # plt.show()


# learning_rate 4e-3
def train_mae(model, trainset, valset=None, MASK_RATIO=0.0, num_epochs=5, batch_size=128, learning_rate=1e-3, TRAIN_MAE=True, SAVE_MODEL_MAE=True, p=4):
    if TRAIN_MAE:

        criterion = nn.MSELoss() # mean square error loss
        optimizer = torch.optim.Adam(model.parameters(),
                                    lr=learning_rate, 
                                    weight_decay=1e-4)
        # scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
        train_loader = torch.utils.data.DataLoader(trainset, 
                                                batch_size=batch_size, 
                                                shuffle=True, num_workers=4)
        if valset:
            val_loader = torch.utils.data.DataLoader(valset, 
                                        batch_size=batch_size, 
                                        shuffle=False, num_workers=2)

        outputs = []
        losses = []

        print(f"Start MAE training for {num_epochs} epochs")
        t_start = time.time()
        for epoch in range(num_epochs):
            running_loss = 0.0
            t_epoch_start = time.time()
            model.train()
            for data in train_loader:
                img, _ = data
                # print(img[0,0,:,:])
                # plot_single_img(img, 10)
                unmasked_img = img.to(device)
                if MASK_RATIO != 0:
                    img = mask(img, MASK_RATIO, p)
                    # print('masking!!!')
                # plot_single_img(img, 10, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                # plot_single_img(img, 10)
                img = img.to(device)
                recon = model(img)
                optimizer.zero_grad()
                loss = criterion(recon, unmasked_img)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            #     break
            # break
            if valset:
                model.eval()
                validation_loss = 0.0

                with torch.no_grad():
                    for data in val_loader:
                        imgs, _ = data
                        imgs = imgs.to(device)
                        outputs = model(imgs)
                        loss = criterion(outputs, imgs)
                        validation_loss += loss.item() * imgs.size(0)

                validation_loss /= len(val_loader.dataset)
            else:
                validation_loss = 0
            epoch_loss = running_loss / len(train_loader)
            t_epoch_end = time.time()
            print('epoch {}: average loss: {:.4f}, val loss: {:.4f}, duration: {:.2f}s'.format(epoch+1, epoch_loss, validation_loss, t_epoch_end - t_epoch_start))
            losses.append(epoch_loss)
            # scheduler.step()
        t_end = time.time()
        print(f"End of MAE training. Training duration: {np.round((t_end-t_start),2)}s. Training loss: {loss}.")

        if SAVE_MODEL_MAE:
            save_folder = 'data/models_imagenette/MAE_CONVNEXT_TESTEPOCHS.pth'
            # save_folder = 'data/models_/MAE_TESTRUN.pth'
            torch.save(model.state_dict(), save_folder)
            print(f'mae model saved to {save_folder}')

        plot_losses(num_epochs, losses)        
        print("\n")
        print("\n")

    else:
        # model.load_state_dict(torch.load('data/models_mnist/MAE_TESTRUN.pth'))
        model.load_state_dict(torch.load('data/models_imagenette/MAE_CONVNEXT_TESTEPOCHS.pth'))


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

    for data, target in testset:
        data_list.append(data.unsqueeze(0))
        target_list.append(target)

    # print(type(data_list))
    test_data_tensor = torch.cat(data_list, dim=0)
    # print(test_data_tensor.shape)
    test_target_tensor = torch.tensor(target_list)
    # print(type(test_data_tensor))
    test_data_tensor = test_data_tensor.to(device)
    # print(test_data_tensor.shape)
    # model = model.cpu()

    x = model(test_data_tensor[0:64,:,:,:])
    embedding = model.encoder(x)
    recon = model.decoder(embedding)
    print(recon.shape)
    for i in range(64):
        recon_cpu = recon[i,:,:,:]#.detach().numpy()
        recon_cpu = recon_cpu.cpu()
        plotimg(test_data_tensor[i,:,:,:], recon_cpu)


def train_classifier(classifier, trainset, valset=None, num_epochs=25, batch_size=128, TRAIN_CLASSIFIER=True, SAVE_MODEL_CLASSIFIER=True):

    classifier.to(device)
    if TRAIN_CLASSIFIER:
        for param in classifier.encoder.parameters():
            param.requires_grad = False
        

        train_loader = torch.utils.data.DataLoader(trainset, 
                                            batch_size=batch_size, 
                                            shuffle=True, num_workers=2)

        if valset:
            val_loader = torch.utils.data.DataLoader(valset, 
                                batch_size=batch_size, 
                                shuffle=False, num_workers=2)    

        # optimizer = torch.optim.Adam(classifier.parameters(), lr=2e-3, weight_decay=0.05)
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, classifier.parameters()), lr=2e-3, weight_decay=0.05)
        # scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
        loss_function =  nn.CrossEntropyLoss().to(device)

        # early_stopper = EarlyStopper(patience=3, min_delta=10)

        losses = []
        print(f"Start CLASSIFIER training for {num_epochs} epochs")
        t_start = time.time()
        for epoch in range(num_epochs):
            running_loss = 0.0
            classifier.train()
            t_epoch_start = time.time()
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                # plot_single_img(inputs.cpu(), 10, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                # print(labels[10])
                outputs = classifier(inputs)
                # print(F.softmax(outputs, dim=1))
                # plot_single_img(outputs.cpu(), 10, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                loss = loss_function(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

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
            else:
                validation_loss = 0
                accuracy = 0
            # if early_stopper.early_stop(validation_loss):             
            #     break

            t_epoch_end = time.time()
            # print('epoch {}: loss: {:.4f} duration: {:.2f}s'.format(epoch+1, float(loss), float(t_epoch_end-t_epoch_start)))
            epoch_loss = running_loss / len(train_loader)
            print('epoch {}: average loss: {:.4f}, val loss: {:.4f}, accuracy: {:.2f}, duration: {:.2f}s'.format(epoch+1, epoch_loss, validation_loss, accuracy, t_epoch_end - t_epoch_start))
            losses.append(epoch_loss)
            # scheduler.step()
        t_end = time.time()
        print(f"End of CLASSIFIER training. Training duration: {np.round((t_end-t_start),2)}s. final loss: {loss}.")

        if SAVE_MODEL_CLASSIFIER:
            save_folder = 'data/models_imagenette/CLASSIFIER_MR_0.pth'
            torch.save(classifier.state_dict(), save_folder)
            print(f'classifier model saved to {save_folder}')

        print("\n")
        # print("\n")
    else:
        classifier.load_state_dict(torch.load('data/models_mnist/CLASSIFIER_MR_02.pth'))

    return classifier


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


    ## IMAGENETTE
    imagenette_classes = [
    'tench',         # Class index 0
    'English springer',  # Class index 1
    'cassette player',   # Class index 2
    'chain saw',      # Class index 3
    'church',         # Class index 4
    'French horn',    # Class index 5
    'garbage truck',  # Class index 6
    'gas pump',       # Class index 7
    'golf ball',      # Class index 8
    'parachute'       # Class index 9
    ]

    for idx, (images, labels) in enumerate(test_loader):
        images, labels = images.to(device), labels.to(device)
        x = model(images)
        _, predicted = torch.max(x.data, 1)
        images = images.cpu()
        plot_single_img(images, 0)
        print(f'prediction: {imagenette_classes[predicted[0].item()]} ----- label: {imagenette_classes[labels[0].item()]}')
        if idx == 10:
            break



if __name__ == "__main__":

    # torch.manuel_seed(42)
    dtype = torch.float32
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f'SELECTED DEVICE: {device}')

    MASK_RATIO = 0

    # normalize = transforms.Normalize(mean=np.mean([0.485, 0.456, 0.406]),
    #                              std=np.mean([0.229, 0.224, 0.225]))

    # normalize = transforms.Normalize(mean=[0.459], std=[0.150])
    # normalize = transforms.Normalize((0.5, ), (0.5, ))
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                #  std=[0.229, 0.224, 0.225])
    transform_train = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((112, 112)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # normalize
    ])

    transform_test = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        # normalize
    ])

    trainset = ImageFolder('data/imagenette2/train', transform=transform_train)
    # print(len(trainset))
    testset = ImageFolder('data/imagenette2/val', transform=transform_test)
    # testset, valset = torch.utils.data.random_split(testset, [int(0.8*len(testset)), int((1-0.8)*len(testset)+1)])

    # mae = AutoEncoder112(in_channels=3, channels=[96, 192, 384, 768], depths=[3, 3, 9, 3]).to(device)
    mae = AutoEncoder112(in_channels=1, channels=[32, 64, 128, 256], depths=[1, 1, 3, 1]).to(device)
    num_epochs_mae = 1
    mae = train_mae(mae, trainset, valset=None, MASK_RATIO=0, num_epochs=num_epochs_mae, TRAIN_MAE=False, SAVE_MODEL_MAE=True, p=8)
    count_parameters(mae)
    # eval_mae(mae, testset)

    num_classes = 10
    classifier = Classifier112(autoencoder=mae, in_features=256, out_features=num_classes).to(device)
    num_epochs_classifier = 1
    classifier = train_classifier(classifier, trainset=trainset, valset=None, num_epochs=num_epochs_classifier,  TRAIN_CLASSIFIER=True, SAVE_MODEL_CLASSIFIER=False)
    count_parameters(classifier)
    eval_classifier(classifier, testset)