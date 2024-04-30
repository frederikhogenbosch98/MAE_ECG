import torch


class SupervisedDataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
        loaded_data = torch.load(data_path)
        print("Type of loaded_data['data']: ", type(loaded_data['data']))
        print("Type of loaded_data['labels']: ", type(loaded_data['labels']))
        self.data = loaded_data['data']
        self.labels = loaded_data['labels'] 

        print(len(self.data))
        print(len(self.labels))

        assert len(self.data) == len(self.labels), "Data and labels must have the same length"
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]


dataset = SupervisedDataset('data/datasets/supervised_dataset_22k.pt')
print(len(dataset))
loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

for data, labels in loader:
    print(data[0])
    print(labels[0])
    break