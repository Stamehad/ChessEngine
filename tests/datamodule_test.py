from chessengine.model.datamodule import ChessDataModule
import yaml
from dotenv import load_dotenv
import os
load_dotenv()
TEST_DATASET = os.getenv("TEST_DATASET")

def test_chess_data_module():
    with open("engine_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    dm = ChessDataModule(config, [TEST_DATASET])
    dm.setup()

    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()

    batch = next(iter(train_loader))
    x, labels = batch

    assert x.shape[0] <= config["dataloader"]["batch_size"]
    assert isinstance(labels, dict)
    assert "eval" in labels and "legal_moves" in labels

    batch = next(iter(val_loader))
    x, labels = batch

    assert x.shape[0] <= config["dataloader"]["batch_size"]
    assert isinstance(labels, dict)
    assert "eval" in labels and "legal_moves" in labels