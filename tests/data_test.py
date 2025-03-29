import pytest
import torch
from chessengine.dataclass import ChessPositionDataset

def test_chess_position_dataset():
    ## This test checks the ChessPositionDataset class
    ## It checks that the dataset loads correctly and that the sample format is as expected

    ########## Load dataset ########
    dataset = ChessPositionDataset(["../chess_engine/data/positions_short3.pt"])

    ########## Choose random sample ##########
    N = len(dataset)
    idx = torch.randint(0, N, (1,)).item()
    batch = dataset[idx]

    ########## Unpack batch ##########
    x, labels = batch
    
    ########## Check the board tensor ##########
    assert isinstance(x, torch.Tensor)
    assert x.shape == (8, 8, 20)  # Updated shape for a chess board
    

    ########## Check the label dictionary ##########
    label_keys = ['eval', 'check', 'king_square', 'threat_target', 'move_target', 'terminal_flag']
    for key in label_keys:
        assert key in labels

    assert isinstance(labels['eval'], torch.Tensor)
    assert labels['eval'].shape == (1,)  # Updated shape for eval
    assert isinstance(labels['check'], torch.Tensor)
    assert labels['check'].shape == (1,)  # Updated shape for check
    assert isinstance(labels['king_square'], torch.Tensor)
    assert labels['king_square'].shape == (1,)  # Updated shape for king_square
    assert isinstance(labels['threat_target'], torch.Tensor)
    assert labels['threat_target'].shape == (64,)  # Updated shape for threat_target
    assert isinstance(labels['move_target'], torch.Tensor)
    assert labels['move_target'].shape == (64,)  # Updated shape for move_target
    assert isinstance(labels['terminal_flag'], torch.Tensor)
    assert labels['terminal_flag'].shape == (1,)  # Updated shape for terminal_flag