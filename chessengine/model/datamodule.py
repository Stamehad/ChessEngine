import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from chessengine.model.dataclass import ChessPositionDataset  # updated stacked format

class ChessDataModule(pl.LightningDataModule):
    """
    Batch format:
    - x: (B, 8, 8, 21) tensor representing the chessboard and piece types
    - labels: dict containing:
        | Key              | Shape      | Description                                                  |
        |------------------|------------|--------------------------------------------------------------|
        | `eval`           | (B, 1)     | Scalar win target (2=white win, 1= draw, 0=black win)        |
        | `move_target`    | (B, 64)    | Labels for changed squares (0=empty, 1–6=piece type)         |
        | `king_square`    | (B, 1)     | Index [0–63] of opponent king square                         |
        | `check`          | (B, 1)     | Whether opponent king is in check (0 or 1)                   |
        | `threat_target`  | (B, 64)    | Labels for newly threatened opponent pieces (-100 or 0 or 1) |
        | `terminal_flag`  | (B, 1)     | Game state (0=active, 1=stalemate, 2=checkmate)              |
        | `legal_moves`    | (B, 64, L) | Like move_target (=ground truth) but for all L legal moves   |
        | `true_index`     | (B,)       | Index of the ground truth move in legal moves                |
    """

    def __init__(self, config, data_paths, OVERFIT=False):
        super().__init__()
        self.data_paths = data_paths
        self.save_hyperparameters()  # captures __init__ args: batch_size, num_workers, etc.
        self.dataloader_config = config["dataloader"]
        self.data_split = config.get("data_split", 0.1)
        self.seed = config.get("seed", 42)
        self.OVERFIT = OVERFIT

    def setup(self, stage=None):
        # Load the full dataset and split
        if stage == "fit" or stage is None:
            full_dataset = ChessPositionDataset(self.data_paths)
            val_size = int(len(full_dataset) * self.data_split)
            train_size = len(full_dataset) - val_size

            if self.OVERFIT:
                _, self.val_dataset = random_split(full_dataset, [train_size, val_size], 
                                                                    generator=torch.Generator().manual_seed(self.seed))
                self.train_dataset = self.val_dataset
            else:
                self.train_dataset, self.val_dataset = random_split(full_dataset, [train_size, val_size], 
                                                                    generator=torch.Generator().manual_seed(self.seed))

            print(f"Train set size: {len(self.train_dataset)}")
            print(f"Validation set size: {len(self.val_dataset)}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            **self.dataloader_config, 
            collate_fn=self._collate_fn, 
            shuffle=True
            )
        
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            **self.dataloader_config, 
            collate_fn=self._collate_fn
            )
    
    def _collate_fn(self, batch):
        xs, labels = zip(*batch)
        xs = torch.stack(xs)  # (B, 8, 8, 21)

        batched_labels = {}
        for k in labels[0]:
            values = [lbl[k] for lbl in labels]
            if k == "legal_moves":
                stacked = torch.stack(values)  # (B, 64, L)
                # Compute mask of which columns are all -100
                is_pad = (stacked == -100)     # (B, 64, L)
                all_pad = is_pad.all(dim=(0, 1))  # (L,)
                last_valid_index = all_pad.logical_not().nonzero(as_tuple=False).max().item()
                trimmed = stacked[:, :, :last_valid_index + 1]
                batched_labels[k] = trimmed
            else:
                batched_labels[k] = torch.stack(values)
                if k == "true_index":
                    batched_labels[k] = batched_labels[k].squeeze(-1)

        return xs, batched_labels