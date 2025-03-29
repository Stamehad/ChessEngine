import yaml
from chessengine.engine_pl import ChessLightningModule
from chessengine.dataclass import ChessPositionDataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl


def test_lightning_train_loop():

    ########## Get config from YAML ##########
    with open("engine_config.yaml", "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    ########## Initialize dataset and model ##########
    dataset = ChessPositionDataset(["../chess_engine/data/positions_short3.pt"])
    model = ChessLightningModule(config)
    
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4, persistent_workers=True)
    val_dataloader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=4, persistent_workers=True)
    trainer = pl.Trainer(fast_dev_run=True)
    trainer.fit(model, dataloader, val_dataloader)