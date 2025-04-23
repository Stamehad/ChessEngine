import torch
from typing import Tuple, List

def masked_one_hot(tensor, num_classes, mask_value=-100):
    """
    Converts tensor of shape (...), with valid values in [0, num_classes-1] and masked entries as mask_value,
    into one-hot of shape (..., num_classes), with zero vectors for masked entries.
    """
    mask = (tensor != mask_value)
    safe_tensor = tensor.clone()
    safe_tensor[~mask] = 0  # dummy class to avoid indexing error
    one_hot = torch.nn.functional.one_hot(safe_tensor, num_classes=num_classes)  # (..., num_classes)
    one_hot[~mask] = 0  # set masked entries to all-zero vector
    return one_hot

def flatten_dict(d):
    """Flatten a nested dictionary."""
    flat_dict = {}
    for key, value in d.items():
        if isinstance(value, dict):
            for sub_key, sub_value in flatten_dict(value).items():
                flat_dict[f"{sub_key}"] = sub_value
        else:
            flat_dict[key] = value
    return flat_dict

def batch_legal_moves(legal_moves_list: List[torch.Tensor], NO_BATCH_DIM=True) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Batches a list of legal move tensors with shape (64, L_i) into a single padded tensor of shape (B, 64, L_max).
    Also returns a mask indicating which entries are real (not padding).
    Args:
        legal_moves_list: List of Tensors, each of shape (64, L_i), where L_i can vary.
        NO_BATCH_DIM: If False, we assume input tensors have shape (B_i, 64, L_i) and produces (B, 64, L_max) where B = sum(B_i).
    Returns:
        A single Tensor of shape (B, 64, L_max), where entries are padded with -100 if L_i < L_max,
        and a mask of shape (B, L_max) where mask[b, l] = True if any square (among 64) has a valid move index at that location.
    """
    if NO_BATCH_DIM:
        # Add a batch dimension to each tensor
        legal_moves_list = [l.unsqueeze(0) for l in legal_moves_list] # (64, L_i) -> (1, 64, L_i)

    L_max = max(legal_moves.shape[-1] for legal_moves in legal_moves_list)
    # Pad each tensor to the maximum length L_max with -100
    padded_legal_moves = [] # list of (B_i, 64, L_max)
    for l in legal_moves_list:
        B, L = l.shape[0], l.shape[-1]
        if l.shape[-1] < L_max:
            pad = torch.full((B, 64, L_max - L), fill_value=-100, dtype=l.dtype)
            l = torch.cat([l, pad], dim=-1)
        padded_legal_moves.append(l)
    
    # Stack the padded tensors into a single tensor
    padded_legal_moves = torch.cat(padded_legal_moves, dim=0)  # shape: (B, 64, L_max)

    # mask: True if any of the 64 squares has a valid move at position l
    mask = (padded_legal_moves != -100).any(dim=1)  # shape: (B, L_max)
    
    return padded_legal_moves, mask  # (B, 64, L_max), (B, L_max)
