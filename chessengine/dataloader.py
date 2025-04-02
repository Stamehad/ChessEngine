import torch
from torch.utils.data import random_split, DataLoader
from chessengine.dataclass import ChessPositionDataset
from chessengine.sampler import ShardSampler

def collate_fn(batch):
    boards = []
    evals, checks, king_squares, threat_targets, move_targets, terminal_flags = [], [], [], [], [], []
    legal_moves_list = []
    true_indices = []

    max_L = max(item[1]['legal_moves'].shape[1] for item in batch)

    for x, labels in batch:
        boards.append(x)

        # Pad legal_moves
        legal_moves = labels['legal_moves']
        L = legal_moves.shape[1]
        if L < max_L:
            pad = torch.full((64, max_L - L), fill_value=-100, dtype=legal_moves.dtype)
            legal_moves = torch.cat([legal_moves, pad], dim=1)
        legal_moves_list.append(legal_moves)

        evals.append(labels['eval'])
        checks.append(labels['check'])
        king_squares.append(labels['king_square'])
        threat_targets.append(labels['threat_target'])
        # move_targets.append(labels['move_target'])
        terminal_flags.append(labels['terminal_flag'])
        true_indices.append(labels['true_index'])

    return (
        torch.stack(boards),
        {
            'eval': torch.stack(evals),
            'check': torch.stack(checks),
            'king_square': torch.stack(king_squares),
            'threat_target': torch.stack(threat_targets),
            # 'move_target': torch.stack(move_targets),
            'terminal_flag': torch.stack(terminal_flags),
            'legal_moves': torch.stack(legal_moves_list),
            'true_index': torch.stack(true_indices).squeeze(-1)  # shape (B,)
        }
    )

def get_dataloaders(data_paths, config):

    #################### Unpack config ###################
    seed = config.get('seed', 42)
    batch_size = config['batch_size']
    num_workers = config.get('num_workers', 4)
    val_fraction = config.get('val_split', 0.1)
    persistent_workers = config.get('persistent_workers', True)

    ################### Create dataset ###################
    print(f"Loading dataset from: {data_paths}")
    full_dataset = ChessPositionDataset(data_paths)

    ################# Split dataset into train and validation sets ###################
    val_size = int(len(full_dataset) * val_fraction)
    train_size = len(full_dataset) - val_size

    train_set, val_set = random_split(full_dataset, [train_size, val_size],
                                       generator=torch.Generator().manual_seed(seed))

    ################### Create dataloaders ###################
    print(f"Train set size: {len(train_set)}")
    print(f"Validation set size: {len(val_set)}")
    print(f"Batch size: {batch_size}")
    
    #print(f"Preparing sampler...")
    #train_sampler = ShardSampler(train_set, shuffle_within_shard=True, shuffle_shards=True)

    print(f'Preparing dataloaders with {num_workers} workers...')
    train_loader = DataLoader(train_set, 
                            batch_size=batch_size, 
                            shuffle=True,
                            #sampler=train_sampler, 
                            collate_fn=collate_fn, 
                            num_workers=num_workers, 
                            persistent_workers=persistent_workers
    )

    #val_sampler = ShardSampler(val_set, shuffle_within_shard=False, shuffle_shards=False)
    val_loader = DataLoader(val_set, 
                            batch_size=batch_size, 
                            shuffle=False, 
                            #sampler=val_sampler,
                            collate_fn=collate_fn, 
                            num_workers=num_workers, 
                            persistent_workers=persistent_workers
    )

    return train_loader, val_loader