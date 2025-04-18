# chess_rl/self_play/dataset_buffer.py
import os
import pickle
import random
import numpy as np
from datetime import datetime

class ReplayBuffer:
    def __init__(self, config):
        self.buffer_dir = config['self_play']['replay_buffer_dir']
        # Max buffer size in terms of *games* (can be adjusted)
        # self.max_size_games = config['self_play'].get('buffer_max_games', 1000)
        os.makedirs(self.buffer_dir, exist_ok=True)
        self.all_game_files = self._load_existing_files()
        print(f"Initialized Replay Buffer. Found {len(self.all_game_files)} existing game files.")


    def _load_existing_files(self):
        """Finds all .pkl game files in the buffer directory."""
        return sorted([os.path.join(self.buffer_dir, f) for f in os.listdir(self.buffer_dir) if f.endswith('.pkl')])

    def add_game_data(self, game_data):
        """Saves a single game's trajectory data to a file."""
        if not game_data:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = os.path.join(self.buffer_dir, f"game_{timestamp}.pkl")
        try:
            with open(filename, 'wb') as f:
                pickle.dump(game_data, f)
            self.all_game_files.append(filename)
            # Optional: Implement buffer size limit (FIFO by deleting oldest files)
            # self._enforce_buffer_limit()
            # print(f"Saved game data to {filename}")
        except Exception as e:
            print(f"Error saving game data to {filename}: {e}")


    # Optional: If buffer gets too big
    # def _enforce_buffer_limit(self):
    #     while len(self.all_game_files) > self.max_size_games:
    #         file_to_remove = self.all_game_files.pop(0) # Remove oldest
    #         try:
    #             os.remove(file_to_remove)
    #             print(f"Removed old game file: {file_to_remove}")
    #         except OSError as e:
    #             print(f"Error removing old game file {file_to_remove}: {e}")

    def get_all_data_paths(self):
        """Returns paths to all currently stored game files."""
        return self.all_game_files

    # Sampling logic will be handled by the PyTorch Dataset/DataLoader
    # We just provide the list of files here.