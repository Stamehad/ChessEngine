import pytest
import torch
import yaml
from chessengine.model.dataloader import get_dataloaders

from dotenv import load_dotenv
import os
load_dotenv()
TEST_DATASET = os.getenv("TEST_DATASET")

def test_dataloader():
    ########## Get config from YAML ##########
    with open("engine_config.yaml", "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    B = config['dataloader']['batch_size']
    F = config['model']['feature_dim']

    config["dataloader"]["num_workers"] = 0
    config["dataloader"]["persistent_workers"] = False


    ########## Initialize dataset and model ##########
    train_loader, val_loader = get_dataloaders([TEST_DATASET], config)

    # Test train loader
    for batch in train_loader:
        x, labels = batch

        assert isinstance(x, torch.Tensor)
        keys = ['eval', 'king_square', 'check', 'threat_target', 'terminal_flag', 'legal_moves', 'true_index']
        for key in keys:
            assert key in labels, f"Key {key} not found in labels"
            
        assert x.shape == (B, 8, 8, F)  # Replace with expected shape
        assert labels['eval'].shape == (B, 1)
        #assert labels['move_target'].shape == (B, 64)
        assert labels['king_square'].shape == (B, 1)
        assert labels['check'].shape == (B, 1)
        assert labels['threat_target'].shape == (B, 64)
        assert labels['terminal_flag'].shape == (B, 1)
        assert labels['legal_moves'].shape[0] == B
        assert labels['legal_moves'].shape[1] == 64
        assert labels['true_index'].shape == (B,)

        break  # Remove this line if you want to test all batches

    # Test validation loader
    for batch in val_loader:
        x, labels = batch
        
        assert isinstance(x, torch.Tensor)
        keys = ['eval', 'king_square', 'check', 'threat_target', 'terminal_flag', 'legal_moves', 'true_index']
        for key in keys:
            assert key in labels, f"Key {key} not found in labels"
            
        assert x.shape == (B, 8, 8, F)
        assert labels['eval'].shape == (B, 1)
        # assert labels['move_target'].shape == (B, 64)
        assert labels['king_square'].shape == (B, 1)  
        assert labels['check'].shape == (B, 1)
        assert labels['threat_target'].shape == (B, 64)
        assert labels['terminal_flag'].shape == (B, 1)
        assert labels['legal_moves'].shape[0] == B
        assert labels['legal_moves'].shape[1] == 64
        assert labels['true_index'].shape == (B,)
        
        break