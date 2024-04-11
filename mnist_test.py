from torchvision import datasets, transforms
import torch
import torch.nn as nn
from model_scratch import AutoEncoder
import matplotlib.pyplot as plt
import time
import numpy as np
from print_funs import plot_losses

# mnist_data = torch.utils.data.Subset(mnist_data, range(4096))
# mnist_data = list(mnist_data)[:4096]
# print(type(mnist_data))

def train(model, trainset, num_epochs=5, batch_size=64, learning_rate=1e-3, TRAIN=True, SAVE_MODEL=True):
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
                img = img.to(device)
                recon = model(img)
                loss = criterion(recon, img)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                running_loss += loss.item()
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



def eval(model, testset, batch_size=64):
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
    print(type(test_data_tensor))

    for i in range(2):
        x = model(test_data_tensor[i])
        embedding = model.encoder(x)
        e1 = embedding
        recon = model.decoder(e1).permute(1,2,0).cpu().detach().numpy()
        # recon = recon.detach().numpy()
        plt.subplot(2, 2, 1)
        plt.imshow(testset[i][0])
        plt.subplot(2, 2, 2)
        plt.imshow(recon)
        plt.show()


if __name__ == "__main__":

    dtype = torch.float32
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    mnist_data = datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor())
    trainset, testset = torch.utils.data.random_split(mnist_data, [50000, 10000])

    model = AutoEncoder().to(device)
    num_epochs = 20
    model = train(model, trainset, num_epochs=num_epochs, TRAIN=False, SAVE_MODEL=True)
    eval(model, testset)
    # imgs = outputs[num_epochs-1][1].cpu()
    # imgs = imgs.detach().numpy()
        