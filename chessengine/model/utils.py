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

def pad_and_stack(tensor_list, BATCH_DIM=False, pad_value=0.0) -> torch.Tensor:
    """
    Args:
        tensor_list: list of tensors of shape (..., L_i) 
        BATCH_DIM: if True, tensors are of shape (B_i, ..., L_i)
        pad_value: value to use for padding (default is 0.0)
    Returns: 
        a padded tensor of shape (B, ..., L_max) where L_max = max(L_i)
        if BATCH_DIM is True, B = sum(B_i)
    """
    if not BATCH_DIM:
        tensor_list = [t.unsqueeze(0) for t in tensor_list]
    
    L_max = max(t.shape[-1] for t in tensor_list)

    padded = []
    for t in tensor_list:
        l = t.shape[-1]
        if l < L_max:
            pad_shape = t.shape[:-1] + (L_max - l,)
            pad = torch.full(pad_shape, fill_value=pad_value, dtype=t.dtype, device=t.device)
            t = torch.cat([t, pad], dim=-1) # (B_i, ..., L_max)
        padded.append(t)
    
    return torch.cat(padded, dim=0)  # (B_max, ..., L_max)

import time
import logging

logger = logging.getLogger(__name__)

class Timer:
    def __init__(self, name):
        self.name = name
        self.start = None

    def __enter__(self):
        self.start = time.perf_counter()

    def __exit__(self, exc_type, exc_value, traceback):
        duration = time.perf_counter() - self.start
        print(f"[Timer] {self.name}: {duration:.4f} seconds")
        

def compute_topk_coverage(move_probs: torch.Tensor, max_k: int = 10) -> torch.Tensor:
    """
    Compute cumulative coverage of top-k moves from a batched move probability distribution.

    Args:
        move_probs: Tensor of shape (B, L) with move probabilities (sum to 1 per batch).
        max_k: Maximum number of top moves to consider (default: 10).

    Returns:
        Tensor of shape (B, max_k) where each element is the cumulative probability
        of the top-k moves per batch. Saturates at 1.0 if fewer than max_k moves exist.
    """
    B, L = move_probs.shape
    k = min(max_k, L)

    # Sort move probabilities in descending order along last dimension
    sorted_probs, _ = torch.sort(move_probs, descending=True, dim=1)  # (B, L)

    # Compute cumulative sum along last dimension
    cumsum = torch.cumsum(sorted_probs, dim=1)  # (B, L)

    if k < max_k:
        # Pad cumsum with last valid value to reach max_k length
        last_vals = cumsum[:, k-1].unsqueeze(1)  # (B, 1)
        pad_size = max_k - k
        pad = last_vals.expand(B, pad_size)  # (B, pad_size)
        coverage = torch.cat([cumsum[:, :k], pad], dim=1)  # (B, max_k)
    else:
        coverage = cumsum[:, :max_k]  # (B, max_k)

    return coverage