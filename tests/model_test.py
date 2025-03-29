import torch
from chessengine.engine import ChessEngine
import yaml

def test_chess_engine_output_shapes():
    config = {
        "embed_dim": 128,
        "num_heads": 8,
        "num_layers": 12,
        "dropout": 0.1,
        "mlp_ratio": 4,
        "eval_hidden_dim": 128,
        "n_recycles": 4,
    }

    ########## Get config from YAML ##########
    with open("engine_config.yaml", "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    ########## Initialize model ##########
    model = ChessEngine(config["model"])

    ########## Dummy input ##########
    B = torch.randint(1, 11, (1,)) # batch size
    F = config["model"]["feature_dim"]
    H = config["model"]["embed_dim"]
    dummy_input = torch.randn(B, 8, 8, F) 

    x, move_pred = model(dummy_input)

    # Check shapes
    assert x.shape == (B, 65, H), f"Expected transformer output shape ({B}, 65, {H}), got {x.shape}"
    assert move_pred.shape == (B, 64, 7), f"Expected move prediction shape ({B}, 64, 7), got {move_pred.shape}"