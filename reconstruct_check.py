import torch
from torchvision import datasets, transforms
from models.model_56x56_TD import AutoEncoder56_TD, Classifier56_TD
from models.model_56x56 import AutoEncoder56, Classifier56
from print_funs import plot_losses, plotimg, plot_single_img, count_parameters
import torch.nn as nn
from models._11am import AutoEncoder11



def eval_mae(model, testset, batch_size=128):
    device = torch.device("mps")

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


    x = model(test_data_tensor[0:64,:,:,:])
    embedding = model.encoder(x)
    # embedding = model.module.encoder(x)
    e1 = embedding
    recon = model.decoder(e1)
    # recon = model.module.decoder(e1)
    # print(recon.shape)
    # print(recon)
    for i in range(10):
        recon_cpu = recon[i,:,:,:]#.detach().numpy()
        recon_cpu = recon_cpu.cpu()
        # print(test_data_tensor[i,:,:,:].shape)
        print(recon_cpu.shape)
        plotimg(test_data_tensor[i,:,:,:], recon_cpu)
        


if __name__ == "__main__":
    device = torch.device("mps")
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((112,112)), 
        transforms.ToTensor(),         
    ])

    ptbxl_dir = 'data/physionet/ptbxl_full_224/'
    testset = datasets.ImageFolder(root=ptbxl_dir, transform=transform)

    # model = AutoEncoder56().to(device)
    model = AutoEncoder11()
    # model = nn.DataParallel(AutoEncoder56())
    # model = AutoEncoder56_TD(R=20, in_channels=1, channels=[16, 32, 64]).to(device) 
    # model.load_state_dict(torch.load('trained_models/RUN_8_5_13_42/MAE_RUN_default_R0_8_5_13_42_epoch_40.pth', map_location=torch.device('cpu')))
    # model.load_state_dict(torch.load('trained_models/dropbox/MAE_RUN_default_R0_16_5_22_32.pth', map_location=torch.device('cpu')))
    model.load_state_dict(torch.load('trained_models/250_epoch_01_05_11am.pth', map_location=torch.device('cpu')))
    # model.load_state_dict(torch.load('trained_models/250_epoch_01_05_11am.pth', map_location=torch.device('cpu')))

    eval_mae(model, testset)