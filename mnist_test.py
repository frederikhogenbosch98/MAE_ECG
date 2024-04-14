from torchvision import datasets, transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import ImageFolder 
from models.model_28x28 import AutoEncoder28, Classifier28
from models.model_128x128 import AutoEncoder128, Classifier128
import matplotlib.pyplot as plt
import time
import numpy as np
from print_funs import plot_losses, plotimg, plot_single_img
from torch.optim.lr_scheduler import StepLR


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



def train_mae(model, trainset, MASK_RATIO, num_epochs=5, batch_size=128, learning_rate=1e-4, TRAIN_MAE=True, SAVE_MODEL_MAE=True, p=4):
    torch.manual_seed(42)
    if TRAIN_MAE:
        criterion = nn.MSELoss() # mean square error loss
        optimizer = torch.optim.Adam(model.parameters(),
                                    lr=learning_rate, 
                                    weight_decay=1e-4)
        # scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
        train_loader = torch.utils.data.DataLoader(trainset, 
                                                batch_size=batch_size, 
                                                shuffle=True, num_workers=4)
        outputs = []
        losses = []

        print(f"Start MAE training for {num_epochs} epochs")
        t_start = time.time()
        for epoch in range(num_epochs):
            running_loss = 0.0
            t_epoch_start = time.time()
            for data in train_loader:
                img, _ = data
                # plot_single_img(img, 10)
                if MASK_RATIO != 0:
                    img = mask(img, MASK_RATIO, p)
                    # print('masking!!!')
                # plot_single_img(img, 10)
                img = img.to(device)
                recon = model(img)
                optimizer.zero_grad()
                loss = criterion(recon, img)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            #     break
            # break
            t_epoch_end = time.time()
            # print('epoch {}: loss: {:.4f} duration: {:.2f}s'.format(epoch+1, float(loss), float(t_epoch_end-t_epoch_start)))
            outputs.append((epoch, img, recon),)
            epoch_loss = running_loss / len(train_loader)
            print('epoch {}: average loss: {:.4f} duration: {:.2f}s'.format(epoch+1, epoch_loss, t_epoch_end - t_epoch_start))
            losses.append(epoch_loss)
            # scheduler.step()
        t_end = time.time()
        print(f"End of MAE training. Training duration: {np.round((t_end-t_start),2)}s. final loss: {loss}.")

        # plot_losses(num_epochs, losses)        
        if SAVE_MODEL_MAE:
            save_folder = 'data/models_imagenette/MAE_MR_00.pth'
            torch.save(model.state_dict(), save_folder)
            print(f'mae model saved to {save_folder}')

        print("\n")
        print("\n")

    else:
        model.load_state_dict(torch.load('data/models_imagenette/MAE_MR_00.pth'))

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
        data_list.append(data)
        target_list.append(target)

    test_data_tensor = torch.cat(data_list, dim=0)
    test_target_tensor = torch.tensor(target_list)
    # print(type(test_data_tensor))
    test_data_tensor = test_data_tensor.unsqueeze(0).permute(1,0,2,3).to(device)
    # print(test_data_tensor.shape)
    # model = model.cpu()

    for i in range(10):
        x = model(test_data_tensor[i])
        embedding = model.encoder(x)
        e1 = embedding
        recon = model.decoder(e1).permute(1,2,0).cpu().detach().numpy()
        plotimg(test_data_tensor[i], recon)


def train_classifier(classifier, trainset, num_epochs, batch_size=128, TRAIN_CLASSIFIER=True, SAVE_MODEL_CLASSIFIER=True):

    classifier.to(device)
    if TRAIN_CLASSIFIER:
        for param in classifier.encoder.parameters():
            param.requires_grad = False
        

        train_loader = torch.utils.data.DataLoader(trainset, 
                                            batch_size=batch_size, 
                                            shuffle=True, num_workers=4)

        optimizer = torch.optim.Adam(classifier.classifier.parameters(), lr=0.01)
        # scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
        loss_function =  nn.CrossEntropyLoss().to(device)

        losses = []
        print(f"Start CLASSIFIER training for {num_epochs} epochs")
        t_start = time.time()
        for epoch in range(num_epochs):
            running_loss = 0.0
            t_epoch_start = time.time()
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = classifier(inputs)
                loss = loss_function(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            t_epoch_end = time.time()
            # print('epoch {}: loss: {:.4f} duration: {:.2f}s'.format(epoch+1, float(loss), float(t_epoch_end-t_epoch_start)))
            epoch_loss = running_loss / len(train_loader)
            print('epoch {}: average loss: {:.4f} duration: {:.2f}s'.format(epoch+1, epoch_loss, t_epoch_end - t_epoch_start))
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
        classifier.load_state_dict(torch.load('data/models_imagenette/CLASSIFIER_MR_0.pth'))

    return classifier


def eval_classifier(model, testset, batch_size=128):

    model.to(device)
    model.eval()

    test_loader = torch.utils.data.DataLoader(testset, 
                                            batch_size=batch_size, 
                                            shuffle=True)
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(F.softmax(outputs, dim=1).data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy: {np.round(accuracy,3)}%')

    # imagenette_classes = [
    # 'tench',         # Class index 0
    # 'English springer',  # Class index 1
    # 'cassette player',   # Class index 2
    # 'chain saw',      # Class index 3
    # 'church',         # Class index 4
    # 'French horn',    # Class index 5
    # 'garbage truck',  # Class index 6
    # 'gas pump',       # Class index 7
    # 'golf ball',      # Class index 8
    # 'parachute'       # Class index 9
    # ]

    # for idx, (images, labels) in enumerate(test_loader):
    #     images, labels = images.to(device), labels.to(device)
    #     x = model(images)
    #     _, predicted = torch.max(x.data, 1)
    #     images = images.cpu()
    #     plot_single_img(images, 0)
    #     print(f'prediction: {imagenette_classes[predicted[0].item()]} ----- label: {imagenette_classes[labels[0].item()]}')
    #     if idx == 10:
    #         break



if __name__ == "__main__":

    dtype = torch.float32
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # transform = transforms.Compose([
    # # transforms.Resize((128, 128)),  
    # transforms.ToTensor(),
    # transforms.Normalize((0.5,), (0.5,))  
    # ])

    # mnist_data = datasets.MNIST('data', train=True, download=True, transform=transform)
    # trainset, testset = torch.utils.data.random_split(mnist_data, [50000, 10000])

    MASK_RATIO = 0

    # mae = AutoEncoder28().to(device)
    # num_epochs_mae = 20
    # mae = train_mae(mae, trainset, MASK_RATIO, num_epochs=num_epochs_mae, TRAIN_MAE=True, SAVE_MODEL_MAE=False)
    # # eval_mae(mae, testset)


    # num_classes = 10
    # classifier = Classifier28(autoencoder=mae, num_classes=num_classes).to(device)
    # num_epochs_classifier = 10
    # classifier = train_classifier(classifier, trainset=trainset, num_epochs=num_epochs_classifier, batch_size=64, TRAIN_CLASSIFIER=True, SAVE_MODEL_CLASSIFIER=False)
    # eval_classifier(classifier, testset)


    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)) 
    ])

    trainset = ImageFolder('data/imagenette2/train', transform=transform)
    testset = ImageFolder('data/imagenette2/val', transform=transform)

    mae = AutoEncoder128().to(device)
    num_epochs_mae = 100
    mae = train_mae(mae, trainset, MASK_RATIO, num_epochs=num_epochs_mae, TRAIN_MAE=True, SAVE_MODEL_MAE=True, p=8)

    num_classes = 10
    classifier = Classifier128(autoencoder=mae, num_classes=num_classes).to(device)
    num_epochs_classifier = 50
    classifier = train_classifier(classifier, trainset=trainset, num_epochs=num_epochs_classifier, batch_size=64, TRAIN_CLASSIFIER=True, SAVE_MODEL_CLASSIFIER=True)
    eval_classifier(classifier, testset)

    # imagenette_loader = torch.utils.data.DataLoader(imagenette_data, batch_size=64, shuffle=True, num_workers=4)


    
    # for images, labels in imagenette_loader:
    #     plot_single_img(images, 1)
    #     print(imagenette_classes[labels[1].item()])
    #     break
    # imagenette_data = datasets.Imagenette('data', split='train', size='full', download=False, transform=transform)
    # print(imagenette_data.shape)
    # trainset, testset = torch.utils.data.random_split(imagenette_data, [])

    # imgs = outputs[num_epochs-1][1].cpu()
    # imgs = imgs.detach().numpy()
        