import torch
from torch.utils.data import Dataset

class SelfPlayDataset(Dataset):
    def __init__(self, buffer, device='cpu'):
        mask = (buffer.values != 255) & (buffer.move_idx >= 0)
        if mask.sum() == 0:
            raise ValueError("No valid self-play samples were generated.")

        features = buffer.features[mask].detach().to(device).float()
        self.features = features.view(-1, 8, 8, 21).to(device)
        self.eval = buffer.values[mask].detach().to(device).long().unsqueeze(-1)
        self.sq_changes = buffer.sq_changes[mask].detach().to(device).long()
        self.label_changes = buffer.label_changes[mask].detach().to(device).long()
        self.true_index = buffer.move_idx[mask].detach().to(device).long()

        n = self.features.size(0)
        self.check = torch.zeros(n, 1, dtype=torch.float32, device=device)
        self.king_square = torch.zeros(n, 1, dtype=torch.long, device=device)
        self.threat_target = torch.full((n, 64), -100, dtype=torch.long, device=device)
        self.terminal_flag = torch.zeros(n, 1, dtype=torch.long, device=device)

    def __len__(self):
        return self.features.size(0)

    def __getitem__(self, idx):
        labels = {
            "eval": self.eval[idx],
            "check": self.check[idx],
            "king_square": self.king_square[idx],
            "threat_target": self.threat_target[idx],
            "terminal_flag": self.terminal_flag[idx],
            "sq_changes": self.sq_changes[idx],
            "label_changes": self.label_changes[idx],
            "true_index": self.true_index[idx],
        }
        return self.features[idx], labels
