import torch
from src.datamodules.modulo_addition_datamodule import ModuloAdditionDataModule


def test_modulo_addition_datamodule():
    p = 113
    batch_size = 64
    dm = ModuloAdditionDataModule(
        p=p,
        train_ratio=0.7,
        seed=42,
        batch_size=batch_size
    )

    assert not dm.data_train and not dm.data_val and not dm.data_test

    dm.setup()
    assert dm.data_train and dm.data_val and dm.data_test
    assert dm.train_dataloader() and dm.val_dataloader() and dm.test_dataloader()

    batch = next(iter(dm.train_dataloader()))
    x, y = batch
    assert x.shape == (batch_size, 3)
    assert x.dtype == torch.int32

    assert len(y) == batch_size
    assert y.shape == (batch_size,)
    assert y.dtype == torch.int32
