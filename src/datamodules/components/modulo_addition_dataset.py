import torch
from torch.utils.data import Dataset
import numpy as np


class ModuloAdditionDataset(Dataset):
    def __init__(self, p=113, train_ratio=0.7, seed=42):
        self.p = p
        self.train_ratio = train_ratio
        np.random.seed(seed)

        # Generate all possible pairs
        pairs = np.array([(a, b) for a in range(p) for b in range(p)])

        # Shuffle and split into train and test
        np.random.shuffle(pairs)
        split_point = int(len(pairs) * train_ratio)
        self.train_pairs = pairs[:split_point]
        self.test_pairs = pairs[split_point:]

    def __len__(self):
        # Depending on whether we are in train or test mode, the total length will change
        if self.training:
            return len(self.train_pairs)
        else:
            return len(self.test_pairs)

    def set_mode(self, mode):
        if mode == "train":
            self.training = True
        elif mode == "test":
            self.training = False
        else:
            raise ValueError("Mode can be either 'train' or 'test'")

    def __getitem__(self, idx):
        # Depending on the mode we are in, we either choose from the training pairs or test pairs
        if self.training:
            pair = self.train_pairs[idx]
        else:
            pair = self.test_pairs[idx]

        a, b = pair

        # Create one-hot encodings for a and b
        a_one_hot = torch.zeros(self.p)
        a_one_hot[a] = 1
        b_one_hot = torch.zeros(self.p)
        b_one_hot[b] = 1

        # Perform the addition operation
        c = (a + b) % self.p

        # Return the one-hot encoded inputs and the result
        return a_one_hot, b_one_hot, c
