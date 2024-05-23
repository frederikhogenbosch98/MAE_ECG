import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.parallel import DistributedDataParallel as DDP



class YourModel(nn.Module):
    def __init__(self):
        super(YourModel, self).__init__()
        self.fc = nn.Linear(10, 10)

    def forward(self, x):
        return self.fc(x)

def main():
    device_ids = [0, 2, 3]
    main_device = device_ids[0]
    device = torch.device(f'cuda:{main_device}' if torch.cuda.is_available() else 'cpu')
    # Initialize dataset and dataloader
    dataset = TensorDataset(torch.randn(1000, 10), torch.randn(1000, 10))
    dataloader = DataLoader(dataset, batch_size=32, num_workers=2)

    # Initialize model
    model = YourModel().to(device)
    model = nn.DataParallel(model, device_ids=device_ids)

    # Define loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Training loop
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    print(f"Rank {device} training completed.")

if __name__ == "__main__":
    main()
