import torch
from torch.utils.data import random_split, DataLoader
from chessengine.dataclass import ChessPositionDataset

def get_dataloaders(data_paths, config):

    #################### Unpack config ###################
    seed = config.get('seed', 42)
    B = config['batch_size']
    num_workers = config.get('num_workers', 4)
    val_fraction = config.get('val_split', 0.1)

    ################### Create dataset ###################
    full_dataset = ChessPositionDataset(data_paths)

    ################# Split dataset into train and validation sets ###################
    val_size = int(len(full_dataset) * val_fraction)
    train_size = len(full_dataset) - val_size

    train_set, val_set = random_split(full_dataset, [train_size, val_size],
                                       generator=torch.Generator().manual_seed(seed))

    ################### Create dataloaders ###################
    train_loader = DataLoader(train_set, batch_size=B, shuffle=True, num_workers=num_workers, persistent_workers=True)
    val_loader = DataLoader(val_set, batch_size=B, shuffle=False, num_workers=num_workers, persistent_workers=True)

    return train_loader, val_loader