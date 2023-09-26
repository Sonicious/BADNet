# platform
from importlib_metadata import version
import platform
print('The Python version is {}.'.format(platform.python_version()))

import numpy as np
print('The numpy version is {}.'.format(version('numpy')))

import rasterio
print('The rasterio version is {}.'.format(version('rasterio')))

import torch
print('The torch version is {}.'.format(version('torch')))
from torch.nn import functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split

import pytorch_lightning as pl
print('The pytorch_lightning version is {}.'.format(version('pytorch_lightning')))
import matplotlib.pyplot as plt
print('The matplotlib version is {}.'.format(version('matplotlib')))

# additional basic packages
import random
import time

# data paths
feature_file_path = 'data/FeatureFile_230628.tif'
label_file_path = 'data/LabelFile_230628.tif'

class BadnetDataset(Dataset):
    def __init__(self, feature_file_path, label_file_path, chip_size=64, overlap=0):
        self.feature_file_path = feature_file_path
        self.label_file_path = label_file_path
        
        # Open the files temporarily just to get the dimensions
        with rasterio.open(self.feature_file_path) as feature_file, \
             rasterio.open(self.label_file_path) as label_file:
            
            self.width, self.height = feature_file.width, feature_file.height
        
        # Define the chip size, overlap, and padding value
        self.chip_size = chip_size
        self.overlap = overlap
        
        # Calculate the number of chips in each dimension
        self.step_size = self.chip_size - self.overlap
        self.chips_x = (self.width - self.overlap) // self.step_size
        self.chips_y = (self.height - self.overlap) // self.step_size
        
    def __len__(self):
        # Total number of chips in the dataset
        return self.chips_x * self.chips_y
    
    def __getitem__(self, idx):
        # Calculate the chip coordinates based on the index
        chip_y, chip_x = divmod(idx, self.chips_x)
        col = chip_x * self.step_size
        row = chip_y * self.step_size
        
        
        # Define the window size (with overlap)
        window_size = self.chip_size + self.overlap
        
        # Read a window from each file
        with rasterio.open(self.feature_file_path, mode='r') as feature_file, \
             rasterio.open(self.label_file_path, mode='r') as label_file:
            # Read a window from each file
            feature_window = feature_file.read(window=rasterio.windows.Window(col, row, window_size, window_size))
            label_window = label_file.read(window=rasterio.windows.Window(col, row, window_size, window_size))
        
        # Convert the numpy arrays to a supported data type
        feature_window = feature_window.astype(np.float32)
        label_window = label_window.astype(np.int32)
        
        # Convert the windows to PyTorch tensors
        feature_tensor = torch.tensor(feature_window)
        label_tensor = torch.tensor(label_window)
        
        label_tensor = label_tensor.squeeze(0)  # Remove the channel dimension if it exists
        return feature_tensor, label_tensor

# Define the CNN model
class BADNet(nn.Module):
    def __init__(self):
        super(BADNet, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, stride=1, padding=1)  # Output size: 16x64x64
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # Output size: 16x32x32
        
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)  # Output size: 32x32x32
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # Output size: 32x16x16
        
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)  # Output size: 64x16x16
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # Output size: 64x8x8
        
        # Fully connected layers
        self.fc1 = nn.Linear(in_features=64 * 8 * 8, out_features=256)  # Output size: 256
        self.fc2 = nn.Linear(in_features=256, out_features=128)  # Output size: 128
        self.fc3 = nn.Linear(in_features=128, out_features=1)  # Output size: 1
        
        # Dropout layer
        self.dropout = nn.Dropout(p=0.25)
        
    def forward(self, x):
        # Apply convolutional layers with ReLU activation and max pooling
        x = self.pool1(F.relu(self.conv1(x)))  # Output size: 16x32x32
        x = self.pool2(F.relu(self.conv2(x)))  # Output size: 32x16x16
        x = self.pool3(F.relu(self.conv3(x)))  # Output size: 64x8x8
        
        # Flatten the output for the fully connected layers
        x = x.view(-1, 64 * 8 * 8)  # Output size: 64 * 8 * 8 = 4096
        
        # Apply fully connected layers with ReLU activation and dropout
        x = self.dropout(F.relu(self.fc1(x)))  # Output size: 256
        x = self.dropout(F.relu(self.fc2(x)))  # Output size: 128
        
        # Final layer with sigmoid activation
        x = torch.sigmoid(self.fc3(x))  # Output size: 1
        
        return x

class LitModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = BADNet()
        self.criterion = nn.MSELoss()
        
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        features, labels = batch
        outputs = self(features)
        loss = self.criterion(outputs, self.burntindex(labels))
        return loss

    def validation_step(self, batch, batch_idx):
        features, labels = batch
        outputs = self(features)
        loss = self.criterion(outputs, self.burntindex(labels))
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)
    
    def burntindex(self, label):
        bai = sum(label.flatten()) / (label.shape[0] * label.shape[1])
        return bai
    
# Load the dataset
dataset = BadnetDataset(feature_file_path, label_file_path, chip_size=64, overlap=0)

# Split the dataset into training, validation, and testing sets
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False, num_workers=0)

# Train the model
lit_model = LitModel()
trainer = pl.Trainer(max_epochs=10, accelerator='auto')
trainer.fit(lit_model, train_loader, val_loader)


