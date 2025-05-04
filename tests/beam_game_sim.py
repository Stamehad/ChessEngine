# run_beam_selfplay.py

import os
import yaml
import torch
from dotenv import load_dotenv

from chessengine.model.engine_pl import ChessLightningModule
from chessengine.chess_rl.beam_search.beam_tree import BeamTree
from chessengine.chess_rl.self_play.beam_game_simulator import BeamGameSimulator

##################################
# Load Config and Model
##################################
load_dotenv()
CHECKPOINT_PATH = os.getenv("BASE_MODEL")

with open("engine_config.yaml", 'r') as f:
    config = yaml.safe_load(f)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
config['train']['device'] = device

print(f"Loading model from checkpoint: {CHECKPOINT_PATH}")
model = ChessLightningModule.load_from_checkpoint(CHECKPOINT_PATH, config=config)
model.eval()
model.to(device)

##################################
# Set Beam Search Parameters
##################################

config_rl = {'beam': {
  'width': 5,                  # Beam width for beam search
  'depth': 10,                  # Maximum depth for beam search
  'topk_schedule': {0: 3, 1: 2, 2: 1, 3: 1},
  #'topk_schedule': {0: 8, 1: 5, 2: 3, 3: 2, 4: 1, 5: 1, 6: 1} 
}
}

# Number of games to simulate
num_games = 4

##################################
# Initialize Beam Game Simulator
##################################

beam_simulator = BeamGameSimulator(
    model=model,
    config=config_rl,
)

##################################
# Play Games
##################################
print(f"Starting {num_games} beam search games...")
data = beam_simulator.play_games(n_games=num_games)
print(data[0].shape)  # (B, 8, 8, 21)
print(data[1].shape)  # (B, 64, L_max)
print(data[2].shape)  # (B,)
print(data[3].shape)  # (B,)
print(data[2])
print(data[3])

##################################
# Print Results
##################################
# for idx, game in enumerate(games):
#     print(f"Game {idx + 1}: Result = {game.outcome}, {len(game.history)} moves played.")

# Optionally: Save games, collect positions, etc.