import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.parallel import DistributedDataParallel as DDP

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

class YourModel(nn.Module):
    def __init__(self):
        super(YourModel, self).__init__()
        self.fc = nn.Linear(10, 10)

    def forward(self, x):
        return self.fc(x)

def main(rank, world_size):
    setup(rank, world_size)

    # Initialize dataset and dataloader
    dataset = TensorDataset(torch.randn(100, 10), torch.randn(100, 10))
    dataloader = DataLoader(dataset, batch_size=32, num_workers=2)

    # Initialize model
    model = YourModel().to(rank)
    model = DDP(model, device_ids=[rank])

    # Define loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Training loop
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(rank), targets.to(rank)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    print(f"Rank {rank} training completed.")
    cleanup()

if __name__ == "__main__":
    world_size = 4  # Number of GPUs available
    torch.multiprocessing.spawn(main, args=(world_size,), nprocs=world_size, join=True)
