import torch
from torch.utils.data import Dataset
import numpy as np


class ModuloAdditionDataset(Dataset):
    def __init__(self, p=113, train_ratio=0.7, seed=42, mode='train'):
        self.p = p
        self.train_ratio = train_ratio
        np.random.seed(seed)

        # Generate all possible pairs
        pairs = np.array([(a, b) for a in range(p) for b in range(p)])

        # Shuffle and split into train and test
        np.random.shuffle(pairs)
        split_point = int(len(pairs) * train_ratio)

        if mode == 'train':
            self.pairs = pairs[:split_point]
        else:
            self.pairs = pairs[split_point:]

    def __len__(self):
        # Depending on whether we are in train or test mode, the total length will change
        return len(self.pairs)

    def __getitem__(self, idx):
        a, b = self.pairs[idx]
        x = torch.tensor([a, b, self.p], dtype=torch.int32)

        # Perform the addition operation
        c = (a + b) % self.p
        y = torch.tensor(c, dtype=torch.int32)

        return x, y
