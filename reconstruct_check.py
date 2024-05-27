import torch
from torchvision import datasets, transforms
from models.model_56x56_TD import AutoEncoder56_TD, Classifier56_TD
from models.model_56x56 import AutoEncoder56, Classifier56
from print_funs import plot_losses, plotimg, plot_single_img, count_parameters
import torch.nn as nn
# from models._11am import AutoEncoder11
from models._11am_back import AutoEncoder11
from models._11am_un import AutoEncoder11_UN
from models.convnext import ConvNext
from models.UNet import UNet
from PIL import Image
import numpy as np

def save_image(tensor, filename):
    # Convert the tensor to a NumPy array and move channels to the last dimension
    image_array = tensor.permute(1, 2, 0).detach().cpu().numpy()
    
    # Normalize the image array to 0-255 range
    image_array = (image_array * 255).astype(np.uint8)
    
    # Handle single-channel grayscale images
    if image_array.shape[2] == 1:
        image_array = image_array.squeeze(axis=2)
    
    # Create an image from the array
    image = Image.fromarray(image_array)
    
    # Save the image
    image.save(filename)


def eval_mae(model, testset, batch_size=128):
    device = torch.device("cuda")

    model.to(device)
    model.eval()

    test_loader = torch.utils.data.DataLoader(testset, 
                                            batch_size=batch_size, 
                                            shuffle=True)



    data_list = []

    for data, _ in testset:
        data_list.append(data.unsqueeze(0))

    test_data_tensor = torch.cat(data_list, dim=0)
    test_data_tensor = test_data_tensor.to(device)


    recon = model(test_data_tensor[0:64,:,:,:])
    # embedding = model.encoder(x)
    # embedding = model.module.encoder(x)
    # e1 = embedding
    # recon = model.decoder(e1)
    # recon = model.module.decoder(e1)
    # print(recon.shape)
    # print(recon)
    # recon = model(x)
    for i in range(10):
        recon_cpu = recon[i,:,:,:]#.detach().numpy()
        recon_cpu = recon_cpu.cpu()
        # print(test_data_tensor[i,:,:,:].shape)
        print(recon_cpu.shape)
        plotimg(test_data_tensor[i,:,:,:], recon_cpu)
        save_image(test_data_tensor[i,:,:,:],f'imgs/original_image_{i}.png')
        save_image(recon_cpu, f'imgs/reconstructed_image_{i}.png')
        


if __name__ == "__main__":
    device = torch.device("cuda:0")
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((128,128)), 
        transforms.ToTensor(),         
    ])
    device_ids = [0, 2, 3]

    ptbxl_dir = 'data/physionet/ptbxl_full_224/'
    testset = datasets.ImageFolder(root=ptbxl_dir, transform=transform)

    # model = AutoEncoder56().to(device)
    # model = nn.DataParallel(AutoEncoder11_UN(channels=[32, 64, 128, 256])).to(device)
    # model = AutoEncoder11(R=100, in_channels=1).to(device)
    model = nn.DataParallel(AutoEncoder11_UN(), device_ids=device_ids).to(device)
    # model = nn.DataParallel(ConvNext())
    # model = nn.DataParallel(AutoEncoder56())
    # model = AutoEncoder56_TD(R=20, in_channels=1, channels=[16, 32, 64]).to(device) 
    # model.load_state_dict(torch.load('trained_models/RUN_8_5_13_42/MAE_RUN_default_R0_8_5_13_42_epoch_40.pth', map_location=torch.device('cpu')))
    model.load_state_dict(torch.load('trained_models/model_comparison/RUN_26_5_23_0_exprun/MAE_RUN_basic_R0_27_5_9_3_epoch_30.pth'))
    # model.load_state_dict(torch.load('trained_models/250_epoch_01_05_11am.pth', map_location=torch.device('cpu')))
    # model.load_state_dict(torch.load('trained_models/250_epoch_01_05_11am.pth', map_location=torch.device('cpu')))

    eval_mae(model, testset)