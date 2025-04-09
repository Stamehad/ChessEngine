import torch
import time

class ChessPositionDataset(torch.utils.data.Dataset):
    def __init__(self, data_paths):
        """
        Args:
            data_paths: list of .pt files (each a dict of stacked tensors)
        """
        self.data = None

        for i, path in enumerate(data_paths):
            start = time.time()
            shard = torch.load(path, map_location="cpu")
            print(f"Loaded {path} in {time.time() - start:.2f}s")

            if self.data is None:
                self.data = {k: [v] for k, v in shard.items()}
            else:
                for k, v in shard.items():
                    self.data[k].append(v)

        # Compute global max L for legal_moves
        global_max_L = max(t.shape[-1] for shard in self.data["legal_moves"] for t in shard)
        print(f"Global max legal_moves width: {global_max_L}")

        # Pad each legal_moves tensor to global_max_L
        padded_legal_moves = []
        for t in self.data["legal_moves"]:
            L = t.shape[-1]
            B = t.shape[0]
            if L < global_max_L:
                pad = torch.full((B, 64, global_max_L - L), fill_value=-100, dtype=t.dtype)
                t = torch.cat([t, pad], dim=-1)
            padded_legal_moves.append(t)
        self.data["legal_moves"] = padded_legal_moves

        # Concatenate tensors across all shards
        self.data = {k: torch.cat(v, dim=0) for k, v in self.data.items()}

    def __len__(self):
        return self.data["eval"].shape[0]

    def __getitem__(self, idx):
        return self.format_item(idx)

    def format_item(self, idx):
        x = self.data['board'][idx].float()  # (8, 8, 21)
        labels = {
            'eval': self.data['eval'][idx].float() / 2,                   # (1,)
            'check': self.data['check'][idx].float(),                     # (1,)
            'king_square': self.data['king_square'][idx].long(),          # (1,)
            'threat_target': self.data['threat_target'][idx].long(),      # (64,)
            'terminal_flag': self.data['terminal_flag'][idx].long(),      # (1,)
            'legal_moves': self.data['legal_moves'][idx].long(),          # (64, L)
            'true_index': self.data['true_index'][idx].long(),            # (1,)
        }
        return x, labels