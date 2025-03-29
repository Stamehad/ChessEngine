import pytorch_lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch
from chessengine.engine import ChessEngine
from chessengine.loss import Loss

class ChessLightningModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        self.model = ChessEngine(config["model"])
        self.loss_module = Loss(config)
        self.loss_config = config["train"]
        
    def forward(self, x):
        return self.model(x)

    def compute_move_accuracy(self, move_pred, move_target):
        """
        Computes the accuracy of the move prediction.
        Args:
            move_pred (Tensor): The predicted moves from the model. (B, 64, 7)
            move_target (Tensor): The ground truth moves. (B, 64) with -100 masking
        Returns:
            accuracy (float): The accuracy of the move prediction between 0 and 1.
        """
        B, S, C = move_pred.shape
        pred_labels = move_pred.argmax(dim=-1)  # (B, 64)
        mask = (move_target != -100)
        correct = (pred_labels == move_target) & mask
        accuracy = correct.sum().float() / mask.sum().clamp(min=1)
        return accuracy.item()

    def training_step(self, batch, batch_idx):
        x, labels = batch  # Assuming batch = (input_tensor, label_dict)
        x_out, move_pred = self.model(x)
        total_loss, loss_dict = self.loss_module(x_out, move_pred, labels)

        self.log("train/loss_total", total_loss, prog_bar=True)
        for name, val in loss_dict.items():
            self.log(f"train/loss_{name}", val, prog_bar=(name == "move"))

        acc = self.compute_move_accuracy(move_pred, labels["move_target"])
        self.log("train/move_accuracy", acc, prog_bar=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        x, labels = batch
        x_out, move_pred = self.model(x)
        total_loss, loss_dict = self.loss_module(x_out, move_pred, labels)

        self.log("val/loss_total", total_loss, prog_bar=True)
        for name, val in loss_dict.items():
            self.log(f"val/loss_{name}", val, prog_bar=(name == "move"))

        acc = self.compute_move_accuracy(move_pred, labels["move_target"])
        self.log("val/move_accuracy", acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, labels = batch
        x_out, move_pred = self.model(x)
        total_loss, loss_dict = self.loss_module(x_out, move_pred, labels)

        self.log("test/loss_total", total_loss, prog_bar=True)
        for name, val in loss_dict.items():
            self.log(f"test/loss_{name}", val, prog_bar=(name == "move"))

        acc = self.compute_move_accuracy(move_pred, labels["move_target"])
        self.log("test/move_accuracy", acc, prog_bar=True)

    def configure_optimizers(self):

        ############### Unpack loss parameters ################
        lr = self.loss_config["lr"]        
        weight_decay = self.loss_config["weight_decay"]
        T_max = self.loss_config["T_max"]

        ############### Optimizer and Scheduler ###############
        optimizer = AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=T_max)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}