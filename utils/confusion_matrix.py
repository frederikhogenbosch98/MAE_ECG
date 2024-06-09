from nn_funcs import conf_matrix, MITBIHImageWithFeatureDataset
from torchvision import transforms
import torch
import numpy as np
from models._11am_un import AutoEncoder11_UN

if __name__ == "__main__":

    device = torch.device("cpu")

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((128, 128)), 
        transforms.ToTensor(),         
    ])

    # LABELED
    mitbih_ds2_dir = 'data/physionet/mitbih_rr/DS2/'
    mitbih_dataset_test = MITBIHImageWithFeatureDataset(root_dir=mitbih_ds2_dir, transform=transform) 
    testset_sup = mitbih_dataset_test

    model = AutoEncoder11_UN()

    model.load_state_dict(torch.load('trained_models/last/last_run.pth')) 

    model.to(device)
    model.eval()

    test_loader = torch.utils.data.DataLoader(testset_sup, 
                                            batch_size=128, 
                                            shuffle=True)
    y_pred = []
    y_true = []
    correct = 0
    total = 0
    test_accuracy = []
    with torch.no_grad():
        for images, features, labels in test_loader:
            images, features, labels = images.to(device), features.to(device), labels.to(device)
            output = model(images, features)

            _, predicted = torch.max(output.data, 1)
            for i in range(len(labels)):
                labels_cpu = labels.cpu()
                preds_cpu = predicted.cpu()
                y_true.append(labels_cpu[i].item())
                y_pred.append(preds_cpu[i].item())
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            test_accuracy.append((predicted == labels).sum().item() / predicted.size(0))
        
    print(f'acc: {np.mean(test_accuracy)}')

    accuracy = 100 * correct / total
    print(f'Accuracy: {np.round(accuracy,3)}%')
    print(conf_matrix(y_true, y_pred))
