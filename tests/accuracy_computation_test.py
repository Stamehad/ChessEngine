import yaml
from chessengine.engine_pl import ChessLightningModule
from chessengine.dataclass import ChessPositionDataset
import torch

def test_move_accuracy_computation():

    ########## Get config from YAML ##########
    with open("engine_config.yaml", "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    ########## Initialize dataset and model ##########
    model = ChessLightningModule(config)

    ########## Dummy data ##########
    move_pred = torch.tensor([[[0.1]*7]*64])  # shape (1, 64, 7)
    move_pred[0, 10, 3] = 5.0  # highest logit → class 3
    move_target = torch.full((1, 64), -100)
    move_target[0, 10] = 3

    acc = model.compute_move_accuracy(move_pred, move_target)
    assert acc == 1.0