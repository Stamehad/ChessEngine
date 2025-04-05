import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from chessengine.model.dataclass import ChessPositionDataset
from chessengine.model.dataloader import collate_fn

class ChessDataModule(pl.LightningDataModule):
    def __init__(self, config, data_paths):
        super().__init__()
        self.data_paths = data_paths
        self.save_hyperparameters()  # captures __init__ args: batch_size, num_workers, etc.
        self.dataloader_config = config["dataloader"]
        self.data_split = config.get("data_split", 0.1)
        self.seed = config.get("seed", 42)

    def setup(self, stage=None):
        # Load the full dataset and split
        full_dataset = ChessPositionDataset(self.data_paths)
        val_size = int(len(full_dataset) * self.data_split)
        train_size = len(full_dataset) - val_size
        self.train_dataset, self.val_dataset = random_split(full_dataset, [train_size, val_size], 
                                                            generator=torch.Generator().manual_seed(self.seed))

        print(f"Train set size: {len(self.train_dataset)}")
        print(f"Validation set size: {len(self.val_dataset)}")
        # print(f"Batch size: {self.batch_size}")
        # print(f'Number of steps per epoch: {len(self.train_dataset) // self.batch_size}')
        # print(f'Number of validation steps: {len(self.val_dataset) // self.batch_size}')
        # print(f'Preparing dataloaders with {self.num_workers} workers...')

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
        return collate_fn(batch)