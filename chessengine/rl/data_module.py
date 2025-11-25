import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, random_split

class SelfPlayDataModule(pl.LightningDataModule):
    def __init__(self, dataset: Dataset, config):
        super().__init__()
        self.dataset = dataset
        self.batch_size = config.get("batch_size", 32)
        self.val_split = config.get("val_split", 0.1)
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage=None):
        if self.val_split <= 0 or len(self.dataset) < 2:
            self.train_dataset = self.dataset
            self.val_dataset = self.dataset
        else:
            val_size = max(1, int(len(self.dataset) * self.val_split))
            train_size = len(self.dataset) - val_size
            self.train_dataset, self.val_dataset = random_split(
                self.dataset,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(42),
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self._collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            collate_fn=self._collate_fn,
        )

    def _collate_fn(self, batch):
        xs, labels = zip(*batch)
        xs = torch.stack(xs)

        batched_labels = {}
        for k in labels[0]:
            values = [lbl[k] for lbl in labels]
            if k == "legal_moves":
                stacked = torch.stack(values)
                is_pad = stacked == -100
                all_pad = is_pad.all(dim=(0, 1))
                nonzero = all_pad.logical_not().nonzero(as_tuple=False)
                if nonzero.numel() == 0:
                    trimmed = stacked[:, :, :0]
                else:
                    trimmed = stacked[:, :, : nonzero.max().item() + 1]
                batched_labels[k] = trimmed
            else:
                batched_labels[k] = torch.stack(values)
                if k == "true_index":
                    batched_labels[k] = batched_labels[k].squeeze(-1)

        return xs, batched_labels
