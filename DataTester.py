import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class DummyDataset(Dataset):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        try:
            # Generate random data for features and labels
            # Features: shape (6, 64, 64), random float values
            # Labels: shape (64, 64), random binary values
            features = torch.randn(6, 64, 64)
            labels = torch.randint(0, 2, (64, 64))
            return features, labels
        except Exception as e:
            print(f'Error at index {idx}: {e}')
            raise

# Usage:
num_samples = 1000
dummy_dataset = DummyDataset(num_samples)

# Create a DataLoader
dummy_dataloader = DataLoader(dummy_dataset, batch_size=32, shuffle=True, num_workers=1)

# Try to fetch a batch of data
data_batch, labels_batch = next(iter(dummy_dataloader))
print(data_batch.shape, labels_batch.shape)