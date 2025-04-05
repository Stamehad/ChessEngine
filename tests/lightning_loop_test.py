import yaml
from chessengine.model.engine_pl import ChessLightningModule
from chessengine.model.dataclass import ChessPositionDataset
from chessengine.model.dataloader import get_dataloaders
import pytorch_lightning as pl

from dotenv import load_dotenv
import os
load_dotenv()
TEST_DATASET = os.getenv("TEST_DATASET")

def test_lightning_train_loop():

    ########## Get config from YAML ##########
    with open("engine_config.yaml", "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    ########## Initialize dataset and model ##########
    #dataset = ChessPositionDataset(["../chess_engine/data/test_positions.pt"])
    train_loader, val_loader = get_dataloaders([TEST_DATASET], config)
    model = ChessLightningModule(config)
    
    trainer = pl.Trainer(fast_dev_run=True)
    trainer.fit(model, train_loader, val_loader)