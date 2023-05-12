from typing import Any, Dict, Optional, Tuple

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from src.datamodules.components.modulo_addition_dataset import ModuloAdditionDataset


class ModuloAdditionDataModule(LightningDataModule):
    def __init__(
        self,
        p: int = 113,
        train_ratio: float = 0.7,
        seed: int = 42,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def prepare_data(self):
        # Data is generated on the fly, so nothing to do here
        pass

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_test:
            self.data_train = ModuloAdditionDataset(
                p=self.hparams.p,
                train_ratio=self.hparams.train_ratio,
                seed=self.hparams.seed,
                mode='train'
            )
            self.data_val = ModuloAdditionDataset(
                p=self.hparams.p,
                train_ratio=self.hparams.train_ratio,
                seed=self.hparams.seed,
                mode='test'
            )
            self.data_test = ModuloAdditionDataset(
                p=self.hparams.p,
                train_ratio=self.hparams.train_ratio,
                seed=self.hparams.seed,
                mode='test'
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "datamodule" / "mnist.yaml")
    cfg.data_dir = str(root / "data")
    _ = hydra.utils.instantiate(cfg)
