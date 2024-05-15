import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class MITBIHImageWithFeatureDataset(Dataset):
    def __init__(self, root_dir, transform=None, scale_method='normalize'):
        self.root_dir = root_dir
        self.transform = transform
        self.scale_method = scale_method

        # Gather all image and feature paths
        self.image_paths = []
        self.feature_paths = []
        self.features = []

        # Traverse the directory structure
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith('.png'):
                    image_path = os.path.join(root, file)
                    feature_path = os.path.splitext(image_path)[0] + 'std.txt'
                    
                    if os.path.exists(feature_path):
                        self.image_paths.append(image_path)
                        self.feature_paths.append(feature_path)
                        
                        # Load feature to collect all features for scaling
                        with open(feature_path, 'r') as f:
                            feature = float(f.read().strip())
                            self.features.append(feature)

        # Print collected paths for debugging
        print(f"Collected {len(self.image_paths)} image paths and {len(self.feature_paths)} feature paths")

        # Convert features to tensor
        self.features = torch.tensor(self.features, dtype=torch.float32)

        # Calculate scaling parameters
        if scale_method == 'normalize':
            self.feature_min = self.features.min().item()
            self.feature_max = self.features.max().item()
            print(self.feature_min)
            print(self.feature_max)
        elif scale_method == 'standardize':
            self.feature_mean = self.features.mean().item()
            self.feature_std = self.features.std().item()
        else:
            raise ValueError("scale_method should be either 'normalize' or 'standardize'")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        feature_path = self.feature_paths[index]



        # Load image
        image = Image.open(image_path).convert('L')  # 'L' mode for grayscale
        if self.transform:
            image = self.transform(image)

        # Load feature
        with open(feature_path, 'r') as f:
            feature = float(f.read().strip())

        # Scale feature
        if self.scale_method == 'normalize':
            feature = (feature - self.feature_min) / (self.feature_max - self.feature_min)
        elif self.scale_method == 'standardize':
            feature = (feature - self.feature_mean) / self.feature_std

        # Convert feature to tensor
        feature = torch.tensor([feature], dtype=torch.float32)

        # Extract label from the path (assuming the structure is root/class_name/filename)
        label = os.path.basename(os.path.dirname(image_path))

        return image, feature, label



# Usage
transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
])

dataset = MITBIHImageWithFeatureDataset(root_dir='data/physionet/mitbih/DS12', transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

# Iterate over the DataLoader
for images, features, labels in dataloader:
    print(images.shape, features.shape, labels)
    print(features)
    
