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
