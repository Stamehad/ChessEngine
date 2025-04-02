import torch

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