import yaml
from chessengine.engine_pl import ChessLightningModule
from chessengine.dataclass import ChessPositionDataset
from chessengine.dataloader import get_dataloaders
import pytorch_lightning as pl


def test_lightning_train_loop():

    ########## Get config from YAML ##########
    with open("engine_config.yaml", "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    ########## Initialize dataset and model ##########
    #dataset = ChessPositionDataset(["../chess_engine/data/test_positions.pt"])
    train_loader, val_loader = get_dataloaders(["../chess_engine/data/test_positions.pt"], config)
    model = ChessLightningModule(config)
    
    trainer = pl.Trainer(fast_dev_run=True)
    trainer.fit(model, train_loader, val_loader)