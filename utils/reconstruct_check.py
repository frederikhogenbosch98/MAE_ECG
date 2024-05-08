import torch



def eval_mae(model, testset, batch_size=128):
    model.to(device)
    model.eval()

    test_loader = torch.utils.data.DataLoader(testset, 
                                            batch_size=batch_size, 
                                            shuffle=True)



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