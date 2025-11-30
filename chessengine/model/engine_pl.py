import pytorch_lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch
from chessengine.model.engine import ChessEngine
from chessengine.model.loss import Loss
from chessengine.model.loss_rl import Loss as RLLoss    
from chessengine.model.utils import flatten_dict
from chessengine.model.metrics import MoveMetrics

class ChessLightningModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(flatten_dict(config))
        self.model = ChessEngine(config["model"])
        self.rl = config.get("rl", False)
        if self.rl:
            self.loss_module = RLLoss(config)
        else:
            self.loss_module = Loss(config)
        self.loss_config = config["train"]
        self.lr = self.loss_config["lr"]
        
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, labels = batch  # Assuming batch = (input_tensor, label_dict)
        x_out, move_pred = self.model(x)
        total_loss, loss_dict, logit_dict = self.loss_module(x_out, move_pred, labels)

        self.log("train/loss_total", total_loss, prog_bar=True)
        for name, val in loss_dict.items():
            self.log(f"train/loss_{name}", val, prog_bar=(name == "move"))

        metrics = MoveMetrics.compute_all_metrics(logit_dict, labels, TRAINING=True)

        for k, val in metrics.items():
            self.log(f"train/{k}", val, prog_bar=(k == "move_accuracy" or k == "true_prob"))

        return total_loss

    def validation_step(self, batch, batch_idx):
        x, labels = batch
        x_out, move_pred = self.model(x)
        total_loss, loss_dict, logit_dict = self.loss_module(x_out, move_pred, labels)

        self.log("val/loss_total", total_loss, prog_bar=True)
        for name, val in loss_dict.items():
            self.log(f"val/loss_{name}", val, prog_bar=(name == "move"))

        metrics = MoveMetrics.compute_all_metrics(logit_dict, labels)

        for k, val in metrics.items():
            self.log(f"val/{k}", val, prog_bar=(k == "move_accuracy" or k == "true_prob"))

    def test_step(self, batch, batch_idx):
        x, labels = batch
        x_out, move_pred = self.model(x)
        total_loss, loss_dict, logit_dict = self.loss_module(x_out, move_pred, labels)

        self.log("test/loss_total", total_loss, prog_bar=True)
        for name, val in loss_dict.items():
            self.log(f"test/loss_{name}", val, prog_bar=(name == "move"))

        metrics = MoveMetrics.compute_all_metrics(logit_dict, labels)

        for k, val in metrics.items():
            self.log(f"test/{k}", val, prog_bar=(k == "move_accuracy" or k == "true_prob"))

    def configure_optimizers(self):

        ############### Unpack loss parameters ################
        lr = self.loss_config["lr"]        
        weight_decay = self.loss_config["weight_decay"]
        T_max = self.loss_config["T_max"]

        ############### Optimizer and Scheduler ###############
        optimizer = AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=T_max)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}