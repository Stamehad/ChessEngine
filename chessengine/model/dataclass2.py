import torch

class ChessPositionDataset(torch.utils.data.Dataset):
    def __init__(self, data_paths):
        """
        Args:
            data_paths: list of .pt files (e.g., ["positions_0.pt", "positions_1.pt"])
        """
        self.data_paths = data_paths
        self.index_map = []
        self.shard_sizes = []

        for i, path in enumerate(data_paths):
            data = torch.load(path, map_location='cpu')
            self.shard_sizes.append(len(data))
            self.index_map.extend([(i, j) for j in range(len(data))])

    def __len__(self):
        return len(self.index_map)
    
    def __getitem__(self, idx):
        shard_idx, local_idx = self.index_map[idx]
        path = self.data_paths[shard_idx]

        # 🔁 One-shard cache (persistent across calls)
        if not hasattr(self, "_shard_cache"):
            self._shard_cache = {}

        if shard_idx not in self._shard_cache:
            print(f"[DEBUG] Loading shard {shard_idx} from {path}")
            self._shard_cache = {shard_idx: torch.load(path)}

        data = self._shard_cache[shard_idx]
        item = data[local_idx]

        if idx % 1000 == 0:
            print(f"[DEBUG] Accessing sample {idx} from shard {shard_idx}, local {local_idx}")

        return self.format_item(item)
    
    # def __getitem__(self, idx):
    #     shard_idx, local_idx = self.index_map[idx]
    #     path = self.data_paths[shard_idx]
    #     data = torch.load(path)
    #     item = data[local_idx]

    #     if idx % 1000 == 0:
    #         shard_idx, local_idx = self.index_map[idx]
    #         print(f"[DEBUG] Loading sample {idx} from shard {shard_idx}, local idx {local_idx}")
    #     return self.format_item(item)

    def format_item(self, item):
        x = item['board'].float()  # (8, 8, 21)
        labels = {
            'eval': item['eval'].float() / 2,                   # (1,)
            'check': item['check'].float(),                     # (1,)
            'king_square': item['king_square'].long(),          # (1,)
            'threat_target': item['threat_target'].long(),      # (64,)
            'terminal_flag': item['terminal_flag'].long(),      # (1,)
            'legal_moves': item['legal_moves'].long(),          # (64, L)
            'true_index': item['true_index'].long(),            # (1,)
        }
        return x, labels