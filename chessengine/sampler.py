import random
from torch.utils.data import Sampler

class ShardSampler(Sampler):
    def __init__(self, dataset, shuffle_within_shard=True, shuffle_shards=False):
        """
        Args:
            dataset: your ChessPositionDataset instance; it must have an attribute 'index_map'
                     where each entry is a tuple (shard_idx, local_idx).
            shuffle_within_shard (bool): if True, shuffle the indices within each shard.
            shuffle_shards (bool): if True, shuffle the order of shards between epochs.
        """
        self.shuffle_within_shard = shuffle_within_shard
        self.shuffle_shards = shuffle_shards

        if hasattr(dataset, 'dataset') and hasattr(dataset, 'indices'):
            self.base_dataset = dataset.dataset
            self.subset_indices = dataset.indices
        else:
            self.base_dataset = dataset
            self.subset_indices = None

        # Build a mapping from shard index to list of global indices
        self.shard_to_indices = {}
        if self.subset_indices is None:
            for global_idx, (shard_idx, local_idx) in enumerate(self.base_dataset.index_map):
                self.shard_to_indices.setdefault(shard_idx, []).append(global_idx)
        else:
            for i, global_idx in enumerate(self.subset_indices):
                shard_idx, local_idx = self.base_dataset.index_map[global_idx]
                self.shard_to_indices.setdefault(shard_idx, []).append(i)

        self.shard_indices = list(self.shard_to_indices.keys())

    def __iter__(self):
        # Optionally shuffle the order of shards
        if self.shuffle_shards:
            random.shuffle(self.shard_indices)
        else:
            self.shard_indices.sort()

        # For each shard, yield its indices (optionally shuffled)
        for shard in self.shard_indices:
            indices = self.shard_to_indices[shard]
            if self.shuffle_within_shard:
                random.shuffle(indices)
            for idx in indices:
                yield idx

    def __len__(self):
        return len(self.dataset)