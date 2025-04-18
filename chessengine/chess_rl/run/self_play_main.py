# chess_rl/run/self_play_main.py
import yaml
import os
import torch
import random
import chess
from importlib import import_module
from chess_rl.self_play.game_simulator import GameSimulator
from chess_rl.self_play.dataset_buffer import ReplayBuffer

def load_config(config_path='chess_rl/config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_plain_model(config):
    """Loads the nn.Module directly for inference."""
    model_module_path = config['model']['model_path'].replace('/', '.').replace('.py', '')
    # Hacky way to find class name (same logic as in LightningModule init)
    # Ideally, refactor _find_model_class_name into a shared util
    import inspect
    import sys
    model_dir = os.path.dirname(config['model']['model_path'])
    if model_dir not in sys.path:
         sys.path.insert(0, model_dir)
    module = import_module(model_module_path)
    model_class_name = None
    for name, obj in inspect.getmembers(module):
        if inspect.isclass(obj) and issubclass(obj, torch.nn.Module) and obj is not torch.nn.Module:
            model_class_name = name
            break
    if model_dir in sys.path and sys.path[0] == model_dir:
        sys.path.pop(0)
    if not model_class_name:
         raise NameError(f"Could not find nn.Module class in {config['model']['model_path']}")

    ModelClass = getattr(module, model_class_name)
    model = ModelClass() # Adjust instantiation if needed

    # Load the latest weights saved by the trainer
    checkpoint_path = os.path.join(config['model']['checkpoint_dir'], config['model']['latest_checkpoint_name'])
    if os.path.exists(checkpoint_path):
        print(f"Loading model weights for self-play from: {checkpoint_path}")
        try:
            # Load state dict, ensure it's mapped to the correct device (CPU usually fine for setup)
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            # Adjust keys if needed (e.g., if saved with 'model.' prefix)
            if list(state_dict.keys())[0].startswith('model.'):
                 state_dict = {k.partition('model.')[2]: v for k, v in state_dict.items()}
            model.load_state_dict(state_dict)
        except Exception as e:
            print(f"Warning: Failed to load weights from {checkpoint_path}. Using initial model weights. Error: {e}")
    else:
        print(f"No checkpoint found at {checkpoint_path}. Using initial model weights for self-play.")

    model.eval() # Set to evaluation mode
    return model

def load_start_positions(fen_files):
    """Loads starting FENs from specified files."""
    start_positions = [None] # Always include standard start
    if fen_files:
        for filename in fen_files:
            try:
                with open(filename, 'r') as f:
                    for line in f:
                        fen = line.strip()
                        if fen:
                            # Basic validation
                            try:
                                chess.Board(fen)
                                start_positions.append(fen)
                            except ValueError:
                                print(f"Warning: Invalid FEN skipped in {filename}: {fen}")
            except FileNotFoundError:
                print(f"Warning: Start position FEN file not found: {filename}")
    print(f"Loaded {len(start_positions) - 1} custom starting positions.")
    return start_positions

def run_self_play_cycle(config):
    print("\n--- Starting Self-Play Cycle ---")
    # Determine device (use GPU if available for faster model inference)
    device = torch.device("cuda" if torch.cuda.is_available() and config['training']['accelerator'] == 'gpu' else "cpu")
    print(f"Using device: {device} for model inference during self-play.")

    # 1. Load the latest model weights (plain nn.Module)
    model = load_plain_model(config)
    model.to(device) # Move model to the chosen device

    # 2. Initialize components
    simulator = GameSimulator(model, config)
    replay_buffer = ReplayBuffer(config)
    start_positions = load_start_positions(config['self_play'].get('start_pos_fen_files', None))

    # 3. Run N self-play games
    num_games = config['self_play']['num_games_per_cycle']
    print(f"Starting {num_games} self-play games...")
    games_played = 0
    while games_played < num_games:
        # Select a starting position (includes standard start)
        start_fen = random.choice(start_positions)
        print(f"\nStarting Game {games_played + 1}/{num_games}...")
        if start_fen:
            print(f"Using Start FEN: {start_fen}")

        game_data = simulator.play_game(start_fen=start_fen)

        # 4. Save game trajectory to buffer
        if game_data:
             replay_buffer.add_game_data(game_data)
             games_played += 1 # Only count if game produced data
        else:
             print("Warning: Game produced no data. Skipping save.")
             # Optionally retry or handle differently

    print(f"--- Self-Play Cycle Completed ({games_played} games saved) ---")

if __name__ == "__main__":
    config = load_config()
    run_self_play_cycle(config)