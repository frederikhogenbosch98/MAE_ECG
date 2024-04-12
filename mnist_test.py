from torchvision import datasets, transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model_28x28 import AutoEncoder, Classifier
import matplotlib.pyplot as plt
import time
import numpy as np
from print_funs import plot_losses, plotimg, plot_single_img


def apply_mask(x, ratio):
    x = x.permute(0,5,1,2,3,4)
    rand_mask = torch.rand(x.shape[0], x.shape[1], x.shape[2], x.shape[3]) < ratio
    rand_mask = rand_mask.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, 1, 4, 4)
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
    x = apply_mask(x, ratio)
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

def mask(batch, ratio):
    x = patchify(batch.cpu(), ratio)
    imgs = unpatchify(x)
    return imgs
    # plt.subplot(2,1,1)
    # plt.imshow(batch[8,0,:,:].cpu().detach().numpy(), cmap="gray")
    # plt.subplot(2,1,2)
    # plt.imshow(imgs[8,0,:,:].cpu().detach().numpy(), cmap="gray")
    # plt.show()



def train_mae(model, trainset, num_epochs=5, batch_size=64, learning_rate=1e-3, TRAIN=True, SAVE_MODEL=True):
    torch.manual_seed(42)
    if TRAIN:
        criterion = nn.MSELoss() # mean square error loss
        optimizer = torch.optim.Adam(model.parameters(),
                                    lr=learning_rate, 
                                    weight_decay=1e-5) # <--
        train_loader = torch.utils.data.DataLoader(trainset, 
                                                batch_size=batch_size, 
                                                shuffle=True)
        outputs = []
        losses = []
        print(f"Start training for {num_epochs} epochs")
        t_start = time.time()
        for epoch in range(num_epochs):
            running_loss = 0.0
            t_epoch_start = time.time()
            for data in train_loader:
                img, _ = data
                # plot_single_img(img, 10)
                img = mask(img, 0.7)
                # plot_single_img(img, 10)
                img = img.to(device)
                recon = model(img)
                loss = criterion(recon, img)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                running_loss += loss.item()
            #     break
            # # break
            t_epoch_end = time.time()
            print('epoch {}: loss: {:.4f} duration: {:.2f}s'.format(epoch+1, float(loss), float(t_epoch_end-t_epoch_start)))
            outputs.append((epoch, img, recon),)
            epoch_loss = running_loss / len(train_loader)
            losses.append(epoch_loss)
        t_end = time.time()
        print(f"End of training. Training duration: {np.round((t_end-t_start),2)}s. final loss: {loss}.")

        plot_losses(num_epochs, losses)        

        if SAVE_MODEL:
            save_folder = 'data/models_mnist/MAE.pth'
            torch.save(model.state_dict(), save_folder)
            print(f'model saved to {save_folder}')

    else:
        model.load_state_dict(torch.load('data/models_mnist/MAE.pth'))

    return model



def eval_mae(model, testset, batch_size=64):
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


def train_classifier(classifier, trainset, num_epochs):

    for param in classifier.encoder.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(classifier.classifier.parameters(), lr=0.001)
    loss_function =  nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = classifier(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()



def eval_classifier(model, testset):
    pass



if __name__ == "__main__":

    dtype = torch.float32
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    mnist_data = datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor())
    trainset, testset = torch.utils.data.random_split(mnist_data, [50000, 10000])


    mae = AutoEncoder().to(device)
    num_epochs_mae = 6
    mae = train_mae(mae, trainset, num_epochs=num_epochs_mae, TRAIN_MAE=True, SAVE_MODEL_MAE=False)
    eval_mae(mae, testset)


    num_classes = 10
    classifier = Classifier(autoencoder=mae, num_classes=num_classes).to(device)
    num_epochs_classifier = 10
    classifier = train_classifier(classifier, autoencoder=mae, num_epochs=num_epochs_classifier)
    eval_classifier(classifier, testset)

    # imgs = outputs[num_epochs-1][1].cpu()
    # imgs = imgs.detach().numpy()
        