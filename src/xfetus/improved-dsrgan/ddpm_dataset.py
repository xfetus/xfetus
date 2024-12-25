import os

from torch.utils.data import Dataset
import torch
import numpy as np

class PrecomputedFetalPlaneDataset(Dataset):
    """Fetal Plane dataset."""

    def __init__(self, root_dir, filenames):
        # Load numpy arrays into memory
        self.data = [np.load(os.path.join(root_dir, f)).astype(np.float32) for f in filenames]
        self.cumulative_lengths = [data.shape[0] for data in self.data]
        self.cumulative_lengths = np.cumsum(self.cumulative_lengths)

    def __len__(self):
        return self.cumulative_lengths[-1]

    def __getitem__(self, idx):

        # Find the numpy array and the index within that array
        for i, length in enumerate(self.cumulative_lengths):
            if idx < length:
                if i == 0:
                    array_idx = idx
                else:
                    array_idx = idx - self.cumulative_lengths[i - 1]
                return torch.from_numpy(self.data[i][array_idx]), i
        
        raise IndexError("Index out of range")

