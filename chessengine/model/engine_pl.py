import pytorch_lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch
from chessengine.model.engine import ChessEngine
from chessengine.model.loss import Loss
from chessengine.model.utils import flatten_dict

class ChessLightningModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(flatten_dict(config))
        self.model = ChessEngine(config["model"])
        self.loss_module = Loss(config)
        self.loss_config = config["train"]
        self.lr = self.loss_config["lr"]
        
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, labels = batch  # Assuming batch = (input_tensor, label_dict)
        x_out, move_pred = self.model(x)
        total_loss, loss_dict, move_logits = self.loss_module(x_out, move_pred, labels)

        self.log("train/loss_total", total_loss, prog_bar=True)
        for name, val in loss_dict.items():
            self.log(f"train/loss_{name}", val, prog_bar=(name == "move"))

        acc = self.compute_move_accuracy(move_logits, labels["true_index"])
        self.log("train/move_accuracy", acc, prog_bar=True)

        avg_prob = self.compute_true_move_prob(move_logits, labels["true_index"])
        self.log("train/move_true_prob", avg_prob, prog_bar=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        x, labels = batch
        x_out, move_pred = self.model(x)
        total_loss, loss_dict, move_logits = self.loss_module(x_out, move_pred, labels)

        self.log("val/loss_total", total_loss, prog_bar=True)
        for name, val in loss_dict.items():
            self.log(f"val/loss_{name}", val, prog_bar=(name == "move"))

        acc = self.compute_move_accuracy(move_logits, labels["true_index"])
        self.log("val/move_accuracy", acc, prog_bar=True)

        avg_prob = self.compute_true_move_prob(move_logits, labels["true_index"])
        self.log("train/move_true_prob", avg_prob, prog_bar=False)

    def test_step(self, batch, batch_idx):
        x, labels = batch
        x_out, move_pred = self.model(x)
        total_loss, loss_dict, move_logits = self.loss_module(x_out, move_pred, labels)

        self.log("test/loss_total", total_loss, prog_bar=True)
        for name, val in loss_dict.items():
            self.log(f"test/loss_{name}", val, prog_bar=(name == "move"))

        acc = self.compute_move_accuracy(move_logits, labels["true_index"])
        self.log("test/move_accuracy", acc, prog_bar=True)

        avg_prob = self.compute_true_move_prob(move_logits, labels["true_index"])
        self.log("train/move_true_prob", avg_prob, prog_bar=False)

    def configure_optimizers(self):

        ############### Unpack loss parameters ################
        lr = self.loss_config["lr"]        
        weight_decay = self.loss_config["weight_decay"]
        T_max = self.loss_config["T_max"]

        ############### Optimizer and Scheduler ###############
        optimizer = AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=T_max)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}
    
    def compute_move_accuracy(self, move_logits, true_index):
        """
        Computes the accuracy of the move prediction.
        Args:
            move_logits (Tensor): logits for all legal moves (B, L)
            true_index (Tensor): index of ground truth move (B,)
        Returns:
            accuracy (float): The accuracy of the move prediction in percentage.
        """
        pred_idx = move_logits.argmax(dim=-1)  # (B,)
        correct = (pred_idx == true_index)  # (B,)
        accuracy = 100 * correct.sum().float() / (true_index != -1).sum().float() 
        return accuracy.item()
    
    def compute_true_move_prob(self, move_logits, true_index):
        """
        Computes the average softmax probability assigned to the correct move.
        Args:
            move_logits (Tensor): logits for all legal moves (B, L)
            true_index (Tensor): index of ground truth move (B,)
        Returns:
            avg_prob (float): Average predicted probability assigned to the correct move.
        """
        mask = (true_index != -1)                               # (B,)
        move_logits = move_logits[mask]                         # (B', L)
        true_index = true_index[mask]                           # (B',)

        probs = torch.softmax(move_logits, dim=-1)              # (B', L)
        true_probs = probs.gather(1, true_index.unsqueeze(1))   # (B', 1)
        avg_prob = true_probs.mean().item()                     # Scalar
        return avg_prob
