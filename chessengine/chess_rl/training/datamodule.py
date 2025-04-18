# chess_rl/training/datamodule.py
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import pytorch_lightning as pl
import pickle
import numpy as np
import os
import random
import chess # Needed for move mapping if done here

class ReplayDataset(Dataset):
    def __init__(self, data_files, config):
        self.data_files = data_files
        self.config = config
        self.all_samples = self._load_all_samples()
        # You need a fixed mapping from move UCI to policy index for the target
        self.move_to_index = self._create_move_index_map() # Implement this based on your model!
        self.num_actions = len(self.move_to_index)

    def _load_all_samples(self):
        """Loads all (state, policy_dict, value) tuples from game files."""
        all_samples = []
        print(f"Loading data from {len(self.data_files)} files...")
        loaded_count = 0
        for file_path in self.data_files:
            try:
                with open(file_path, 'rb') as f:
                    game_data = pickle.load(f)
                    all_samples.extend(game_data)
                    loaded_count += 1
            except Exception as e:
                print(f"Warning: Could not load or read file {file_path}: {e}")
        print(f"Loaded {len(all_samples)} samples from {loaded_count} files.")
        return all_samples

    def _create_move_index_map(self):
        """Creates a mapping from move UCI string to model output index.
           *** This MUST match your model's output layer structure ***
        """
        # Example: Replace with your actual move mapping logic
        # Should match the inverse of what's used in MCTS._map_logits_to_legal_moves
        all_moves_list = self._get_all_possible_moves_from_config_or_model() # Get the list used by the model
        move_map = {move: i for i, move in enumerate(all_moves_list)}
        if not move_map:
             raise ValueError("Move index map could not be created. Check model configuration.")
        print(f"Created move map with {len(move_map)} entries.")
        return move_map

    def _get_all_possible_moves_from_config_or_model(self):
        # Placeholder: How do you know the full list of moves your model predicts?
        # It might be in a config file, or derived from python-chess potentially,
        # but MUST be consistent.
        # raise NotImplementedError("Please implement _get_all_possible_moves_from_config_or_model")
        # Example (requires a fixed definition):
        try:
            # Assume you have a function in your model file or utils that provides this
            from chessengine.model.transformer_model import get_move_list
            return get_move_list()
        except ImportError:
             raise ImportError("Could not import get_move_list from your model file. Please define how to get the full move list.")


    def __len__(self):
        return len(self.all_samples)

    def __getitem__(self, idx):
        state_repr, policy_dict, value_target = self.all_samples[idx]

        # Convert policy dict {move_uci: prob} to target tensor [num_actions]
        policy_target_tensor = torch.zeros(self.num_actions)
        for move_uci, prob in policy_dict.items():
            if move_uci in self.move_to_index:
                index = self.move_to_index[move_uci]
                policy_target_tensor[index] = prob
            else:
                 # This might happen if MCTS produces a move UCI not in the fixed map
                 # (e.g., unexpected promotion) - needs careful handling/logging
                 print(f"Warning: Move '{move_uci}' from buffer not in move_to_index map. Skipping.")


        # Ensure targets have correct types
        state_tensor = torch.from_numpy(state_repr).float() # Ensure float
        value_target_tensor = torch.tensor(value_target).float() # Ensure float

        return state_tensor, policy_target_tensor, value_target_tensor


class ReplayDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.buffer_dir = config['self_play']['replay_buffer_dir']
        self.batch_size = config['training']['batch_size']
        self.sample_fraction = config['training']['buffer_sample_fraction']
        self.all_game_files = []

    def setup(self, stage=None):
        # Find all game files in the buffer directory
        self.all_game_files = sorted([os.path.join(self.buffer_dir, f) for f in os.listdir(self.buffer_dir) if f.endswith('.pkl')])
        if not self.all_game_files:
            print("Warning: No data files found in replay buffer directory.")
            self.dataset = None # Handle empty buffer case
        else:
            # Sample a fraction of files to load for this training epoch/cycle
            num_files_to_sample = int(len(self.all_game_files) * self.sample_fraction)
            num_files_to_sample = max(1, min(num_files_to_sample, len(self.all_game_files))) # Ensure at least 1, max all
            files_to_load = random.sample(self.all_game_files, num_files_to_sample)

            self.dataset = ReplayDataset(files_to_load, self.config)


    def train_dataloader(self):
        if self.dataset is None or len(self.dataset) == 0:
             print("No data available for training.")
             # Return an empty loader to avoid crashing Trainer.fit
             return DataLoader([])

        # Create a DataLoader only if dataset is valid
        # No need for SubsetRandomSampler here if ReplayDataset loads a subset of files
        # Shuffle=True is important for training stability
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=os.cpu_count()//2, pin_memory=True)

    # Optional: Implement val_dataloader, test_dataloader if needed