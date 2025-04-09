import torch
import os
from tqdm import tqdm

def convert_shard_format(input_path, output_path):
    os.makedirs(output_path, exist_ok=True)
    pt_files = sorted(f for f in os.listdir(input_path) if f.endswith(".pt"))

    for filename in tqdm(pt_files, desc="Converting shards"):
        in_file = os.path.join(input_path, filename)
        out_file = os.path.join(output_path, filename)

        data = torch.load(in_file, map_location="cpu")

        keys = data[0].keys()
        batch = {key: [] for key in keys}

        # Collect values
        for sample in data:
            for key in keys:
                batch[key].append(sample[key])

        # Special handling for legal_moves
        legal_moves_list = batch['legal_moves']
        max_L = max(t.shape[1] for t in legal_moves_list)
        padded_legal_moves = []

        for t in legal_moves_list:
            L = t.shape[1]
            if L < max_L:
                pad = torch.full((64, max_L - L), fill_value=-100, dtype=t.dtype)
                t = torch.cat([t, pad], dim=1)
            padded_legal_moves.append(t)

        # Replace legal_moves with padded and stacked
        stacked = {}
        stacked['legal_moves'] = torch.stack(padded_legal_moves)

        # Stack all other fields
        for key in keys:
            if key == 'legal_moves':
                continue
            values = batch[key]
            if isinstance(values[0], torch.Tensor):
                stacked[key] = torch.stack(values)
            else:
                stacked[key] = torch.tensor(values)

        # Save
        torch.save(stacked, out_file)

    print(f"✅ Done! Converted shards saved to: {output_path}")