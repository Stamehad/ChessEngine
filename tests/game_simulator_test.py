from chessengine.model.engine_pl import ChessLightningModule
from chessengine.chess_rl.self_play.game_simulator import GameSimulator
import chess
import chess.pgn
import torch
import yaml
import io
import json
import os
from dotenv import load_dotenv
load_dotenv()
CHECKPOINT_PATH = os.getenv("BASE_MODEL")


# load configs
with open("engine_config.yaml", "r") as f:
    config = yaml.safe_load(f)

rl_config = {
    'mcts':{"num_simulations": 5, "cpuct": 1.0, "temperature_start": 1.0,}
}

# load model and mcts
model = ChessLightningModule.load_from_checkpoint(CHECKPOINT_PATH, config=config)
sim = GameSimulator(model, rl_config)

# start_fens = None  
# n_games = 2 # Start from the initial position
# x, pi, lm, z = sim.play_games(start_fens=start_fens, n_games=n_games)

# print("x shape:", x.shape, "x dtype:", x.dtype)
# print("pi shape:", pi.shape, "pi dtype:", pi.dtype)
# print("lm shape:", lm.shape, "lm dtype:", lm.dtype)
# print("z shape:", z.shape, "z dtype:", z.dtype)

# print(z)

# no_p = x.shape[0]   
# L_max = pi.shape[-1]

# # Check the shapes of the outputs
# assert x.shape == (no_p, 8, 8, 21)
# assert pi.shape == (no_p, L_max)
# assert lm.shape == (no_p, 64, L_max)
# assert z.shape == (no_p,)

# # Check the data types of the outputs
# assert x.dtype == torch.uint8
# assert pi.dtype == torch.float32
# assert lm.dtype == torch.int8
# assert z.dtype == torch.int8

def test_game_simulator():
    """
    Test the GameSimulator class.
    """
    # Test 1: Play a game and check the output
    start_fens = None  
    n_games = 2 # Start from the initial position
    x, pi, lm, z = sim.play_games(start_fens=start_fens, n_games=n_games)

    no_p = x.shape[0]   
    L_max = pi.shape[-1]

    # Check the shapes of the outputs
    assert x.shape == (no_p, 8, 8, 21)
    assert pi.shape == (no_p, L_max)
    assert lm.shape == (no_p, 64, L_max)
    assert z.shape == (no_p,)

    # Check the data types of the outputs
    assert x.dtype == torch.uint8
    assert pi.dtype == torch.float32
    assert lm.dtype == torch.int8
    assert z.dtype == torch.int8