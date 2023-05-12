import torch
from src.datamodules.components.modulo_addition_dataset import ModuloAdditionDataset


def test_modulo_addition_dataset():
    p = 113
    dataset = ModuloAdditionDataset(p=p, train_ratio=0.7, seed=42, mode='train')
    assert len(dataset) == int(p * p * 0.7)
    x, y = dataset[0]
    assert x.shape == (3,)
    assert y.shape == ()
    assert x.dtype == torch.int32
    assert y.dtype == torch.int32
    assert (x[0] + x[1]) % p == y
