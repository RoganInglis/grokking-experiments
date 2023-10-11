from typing import Any, List

import os
import torch
import wandb
import plotly.express as px
import pandas as pd
import torchlens as tl
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy


class TransformerLitModule(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler = None,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["net"])

        self.net = net

        # loss function
        self.criterion = torch.nn.CrossEntropyLoss()

        # metric objects for calculating and averaging accuracy across batches
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()

        self.image_output_dir = None

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        self.val_acc_best.reset()

    def step(self, batch: Any, verbose=False):
        x, y = batch
        logits = self.forward(x)
        logits = logits[:, -1, :]
        loss = self.criterion(logits.to(torch.float64), y.long())

        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch, verbose=False)

        # update and log metrics
        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": targets}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def save_fourier_embedding_image(self, batch_idx: int):
        if self.image_output_dir is None:
            self.image_output_dir = os.path.join(self.logger.save_dir, 'images')
            os.makedirs(self.image_output_dir, exist_ok=True)

        embedding_img_dir = os.path.join(self.image_output_dir, 'embedding')
        os.makedirs(embedding_img_dir, exist_ok=True)

        emb_weights = self.net.emb.detach().cpu()
        emb_fourier_norm = torch.fft.fft(emb_weights, dim=0).norm(dim=1)

        df = pd.DataFrame({'Frequency k': range(len(emb_fourier_norm[:emb_fourier_norm.shape[0] // 2])),
                           'Norm of Fourier Component': emb_fourier_norm[:emb_fourier_norm.shape[0] // 2]})
        fig = px.bar(df, x='Frequency k', y='Norm of Fourier Component', title='Fourier Components of Embedding Matrix', range_y=[0, 50])
        file_path = os.path.join(embedding_img_dir, f'fourier_embedding_{"{:06}".format(batch_idx)}.png')
        fig.write_image(file_path)
        wandb.log({'Fourier Components of Embedding Matrix': wandb.Image(file_path)})

    def compute_model_history(self):
        p = 113   # TODO - make this a parameter
        x = torch.stack([
            torch.arange(0, p).unsqueeze(-1).expand(-1, p).reshape(-1),
            torch.arange(0, p).repeat(p),
            torch.ones(p * p) * p
        ]).permute(1, 0).long()
        model_history = tl.log_forward_pass(self.net, x, layers_to_save='all')
        return model_history

    def save_attention_score_for_head_image(self, batch_idx: int, model_history: dict,  head: int = 0):
        if self.image_output_dir is None:
            self.image_output_dir = os.path.join(self.logger.save_dir, 'images')
            os.makedirs(self.image_output_dir, exist_ok=True)

        attention_img_dir = os.path.join(self.image_output_dir, 'attention')
        os.makedirs(attention_img_dir, exist_ok=True)

        # Get attention score for relevant head
        p = 113  # TODO - make this a parameter
        attention_score_for_head = model_history['transformer.layers.0.attn.fn.softmax'].tensor_contents[:, head, -1, 0].reshape(p, p).detach().cpu().numpy()

        # Plot as heatmap
        df = pd.DataFrame(attention_score_for_head)
        fig = px.imshow(df, title=f'Attention Score for Head {head}')
        file_path = os.path.join(attention_img_dir, f'attention_score_for_head_{head}_{"{:06}".format(batch_idx)}.png')
        fig.write_image(file_path)
        wandb.log({f'Attention Score for Head {head}': wandb.Image(file_path)})

    def save_activation_for_neuron_image(self, batch_idx: int, model_history: dict, neuron: int = 0):
        if self.image_output_dir is None:
            self.image_output_dir = os.path.join(self.logger.save_dir, 'images')
            os.makedirs(self.image_output_dir, exist_ok=True)

        activation_img_dir = os.path.join(self.image_output_dir, 'activation')
        os.makedirs(activation_img_dir, exist_ok=True)

        # Get activation for relevant neuron
        p = 113
        activation_for_neuron = model_history['transformer.layers.0.ff.fn.net.1'].tensor_contents[:, -1, neuron].reshape(p, p).detach().cpu().numpy()

        # Plot as heatmap
        df = pd.DataFrame(activation_for_neuron)
        fig = px.imshow(df, title=f'Activation for Neuron {neuron}')
        file_path = os.path.join(activation_img_dir, f'activation_for_neuron_{neuron}_{"{:06}".format(batch_idx)}.png')
        fig.write_image(file_path)
        wandb.log({f'Activation for Neuron {neuron}': wandb.Image(file_path)})

    def save_norm_of_logits_in_2d_fourier_basis_image(self, batch_idx: int, model_history: dict):
        if self.image_output_dir is None:
            self.image_output_dir = os.path.join(self.logger.save_dir, 'images')
            os.makedirs(self.image_output_dir, exist_ok=True)

        logits_img_dir = os.path.join(self.image_output_dir, 'logits')
        os.makedirs(logits_img_dir, exist_ok=True)

        # Get logits
        p = 113
        logits = model_history['transformer.layers.0.ff.fn.net.2'].tensor_contents[:, -1, 0].reshape(p, p).detach().cpu().numpy()

        # Get norm of logits in 2d fourier basis
        logits_fourier_norm = torch.fft.fft(torch.tensor(logits), dim=0).norm(dim=1).detach().cpu().numpy()

        # Plot as heatmap
        df = pd.DataFrame(logits_fourier_norm)
        fig = px.imshow(df, title=f'Norm of Logits in 2d Fourier Basis')
        file_path = os.path.join(logits_img_dir, f'logits_fourier_norm_{"{:06}".format(batch_idx)}.png')
        fig.write_image(file_path)
        wandb.log({f'Norm of Logits in 2d Fourier Basis': wandb.Image(file_path)})

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_epoch_end(self, outputs: List[Any]):
        acc = self.val_acc.compute()  # get current val acc
        self.val_acc_best(acc)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/acc_best", self.val_acc_best.compute(), prog_bar=True)

        model_history = self.compute_model_history()
        self.save_attention_score_for_head_image(self.current_epoch, model_history)
        self.save_activation_for_neuron_image(self.current_epoch, model_history)

        self.save_fourier_embedding_image(self.current_epoch)

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.test_acc(preds, targets)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "mnist.yaml")
    _ = hydra.utils.instantiate(cfg)
