from nn_funcs import conf_matrix, MITBIHImageWithFeatureDataset
from torchvision import transforms
import torch
import numpy as np
from models._11am_un import AutoEncoder11_UN, Classifier_UN
from models._11am_back import AutoEncoder11
import torch.nn as nn

if __name__ == "__main__":

    R_0_classifiers = ['CLASSIFIER_RUN_cp_R0_8_6_11_27.pth', 'CLASSIFIER_RUN_cp_R0_8_6_11_49.pth', 'CLASSIFIER_RUN_cp_R0_8_6_12_12.pth',\
                       'CLASSIFIER_RUN_cp_R0_8_6_12_37.pth', 'CLASSIFIER_RUN_cp_R0_8_6_13_2.pth', 'CLASSIFIER_RUN_cp_R0_8_6_13_28.pth', \
                        'CLASSIFIER_RUN_cp_R0_8_6_13_59.pth', 'CLASSIFIER_RUN_cp_R0_8_6_14_30.pth', 'CLASSIFIER_RUN_cp_R0_8_6_15_2.pth',\
                            'CLASSIFIER_RUN_cp_R0_8_6_15_44.pth', 'CLASSIFIER_RUN_cp_R0_8_6_16_27.pth', 'CLASSIFIER_RUN_cp_R0_8_6_17_11.pth']


    R_100_classifier = ["CLASSIFIER_RUN_cp_R100_8_6_17_37.pth",
    "CLASSIFIER_RUN_cp_R100_8_6_18_1.pth",
    "CLASSIFIER_RUN_cp_R100_8_6_18_25.pth",
    "CLASSIFIER_RUN_cp_R100_8_6_18_52.pth",
    "CLASSIFIER_RUN_cp_R100_8_6_19_20.pth",
    "CLASSIFIER_RUN_cp_R100_8_6_19_47.pth",
    "CLASSIFIER_RUN_cp_R100_8_6_20_21.pth",
    "CLASSIFIER_RUN_cp_R100_8_6_20_55.pth",
    "CLASSIFIER_RUN_cp_R100_8_6_21_29.pth",
    "CLASSIFIER_RUN_cp_R100_8_6_22_15.pth",
    "CLASSIFIER_RUN_cp_R100_8_6_23_2.pth",
    "CLASSIFIER_RUN_cp_R100_8_6_23_50.pth"]
    
    device = torch.device("cuda:0")

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((128, 128)), 
        transforms.ToTensor(),         
    ])

    # LABELED
    mitbih_ds2_dir = 'data/physionet/mitbih_rr/DS2/'
    mitbih_dataset_test = MITBIHImageWithFeatureDataset(root_dir=mitbih_ds2_dir, transform=transform) 
    testset_sup = mitbih_dataset_test

    # mae = AutoEncoder11(R=100, factorization='cp')
    for i in R_100_classifier:
        print(i)
        mae = AutoEncoder11(R=100)
        classifier = Classifier_UN(autoencoder=mae,in_features=256, out_features=5)
        classifier = nn.DataParallel(classifier, device_ids=[0,2,3]).to(device)

        classifier.load_state_dict(torch.load(f'trained_models/compressed/RUN_8_6_11_15_full_training_runn/{i}'))

        classifier.to(device)
        classifier.eval()

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
                output = classifier(images, features)

                _, predicted = torch.max(output.data, 1)
                for i in range(len(labels)):
                    y_true.append(labels[i].item())
                    y_pred.append(predicted[i].item())
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                test_accuracy.append((predicted == labels).sum().item() / predicted.size(0))
            
        print(f'acc: {np.mean(test_accuracy)}')

        accuracy = 100 * correct / total
        print(f'Accuracy: {np.round(accuracy,3)}%')
        # print(conf_matrix(y_true, y_pred))
