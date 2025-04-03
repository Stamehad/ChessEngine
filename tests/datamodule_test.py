from chessengine.datamodule import ChessDataModule
import yaml

def test_chess_data_module():
    with open("engine_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    data_paths = ["data/test_positions.pt"]
    
    dm = ChessDataModule(config, data_paths)
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