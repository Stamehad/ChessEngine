import yaml
from chessengine.model.engine_pl import ChessLightningModule
from chessengine.model.dataclass import ChessPositionDataset
import torch

def test_move_accuracy_computation():
    with open("engine_config.yaml", "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    model = ChessLightningModule(config)

    B = 4  # batch size
    L = 40  # number of legal moves
    true_index = torch.randint(0, L, (B,))
    move_logits = torch.full((B, L), -10.0)
    for i in range(B):
        move_logits[i, true_index[i]] = 10.0

    true_index[0] = -1
    acc = model.compute_move_accuracy(move_logits, true_index)
    assert acc == 100.0