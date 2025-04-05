import torch
import yaml
from chessengine.model.engine_pl import ChessLightningModule

def test_true_move_prob_metric():
    with open("engine_config.yaml", "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    model = ChessLightningModule(config)

    B = 4  # batch size
    L = 10  # number of legal moves
    true_index = torch.randint(0, L, (B,))
    move_logits = torch.full((B, L), -10.0)
    for i in range(B):
        move_logits[i, true_index[i]] = 10.0

    avg_prob = model.compute_true_move_prob(move_logits, true_index)
    assert abs(avg_prob - 1.0) < 1e-5
