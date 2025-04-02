import torch

class ChessPositionDataset(torch.utils.data.Dataset):
    def __init__(self, data_paths):
        """
        Args:
            data_paths: list of .pt files (e.g., ["positions_0.pt", "positions_1.pt"])
        """
        self.samples = []
        for path in data_paths:
            data = torch.load(path, weights_only=True)
            self.samples.extend(data)

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        item = self.samples[idx]
        return self.format_item(item)
    
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