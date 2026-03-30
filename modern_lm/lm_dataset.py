import glob
import os

import numpy as np
import torch
from torch.utils.data import IterableDataset


class LMDataset(IterableDataset):  # pyright: ignore[reportMissingTypeArgument]
    """A simple dataset for training models with language modeling objective.

    Dataset files should contain token ids, possibly divided into shards (.npy files),
    each shard then will be loaded lazily.
    """

    def __init__(
        self,
        dataset_dir: str,
        batch_size: int,
        seq_length: int,
        num_replicas: int = 1,
        rank: int = 0,
    ) -> None:
        shard_files = glob.glob(os.path.join(dataset_dir, "*.npy"))
        if len(shard_files) == 0:
            raise ValueError(f"Could not find any .npy file in {dataset_dir}")

        self.shard_files = sorted(shard_files)

        self.batch_size = batch_size
        self.seq_length = seq_length
        self.num_replicas = num_replicas
        self.rank = rank

        self.token_per_batch = self.batch_size * self.seq_length

        # validate data
        self.total_tokens = None
        self._validate_shards()

        self.reset()

    def reset(self) -> None:
        """Reset the dataset to the beginning."""
        self.shard_idx = -1
        self.ptr = self.rank * self.token_per_batch
        self._load_next_shard()

    def _validate_shards(self) -> None:
        # only run validation on master process
        if self.rank != 0:
            return

        total_tokens = 0
        for shard_file in self.shard_files:
            shard = np.load(shard_file, mmap_mode="r")
            num_shard_tokens = shard.shape[0]
            if num_shard_tokens < self.token_per_batch + 1:
                raise ValueError(
                    f"Shard {shard_file} contains {num_shard_tokens} tokens, which is less than "
                    f"the required {self.token_per_batch + 1} tokens for one batch (on a single replica)."
                )
            total_tokens += num_shard_tokens

        self.total_tokens = total_tokens

    def __iter__(self):
        self.reset()

        while self.shard_idx < len(self.shard_files):
            input_ids = np.empty((self.token_per_batch,), dtype=np.uint16)
            labels = np.empty((self.token_per_batch,), dtype=np.uint16)
            num_tokens_to_fill = self.token_per_batch
            if self.ptr + (num_tokens_to_fill + 1) - 1 >= self.shard.shape[0]:
                # if we do not have enough tokens, take the remaining tokens and load the next shard
                num_remain_tokens = self.shard.shape[0] - self.ptr - 1
                if num_remain_tokens > 0:
                    input_ids[:num_remain_tokens] = self.shard[self.ptr : -1]
                    labels[:num_remain_tokens] = self.shard[self.ptr + 1 :]
                    num_tokens_to_fill -= num_remain_tokens
                self.ptr = 0
                if not self._load_next_shard():
                    break

            # assume each shard contains no less than `num_tokens_to_fill + 1` tokens
            # TODO: handle this assumption
            assert num_tokens_to_fill + 1 <= self.shard.shape[0]
            input_ids[-num_tokens_to_fill:] = self.shard[self.ptr : self.ptr + num_tokens_to_fill]
            labels[-num_tokens_to_fill:] = self.shard[
                self.ptr + 1 : self.ptr + num_tokens_to_fill + 1
            ]
            self.ptr = (
                self.ptr + num_tokens_to_fill + self.token_per_batch * (self.num_replicas - 1)
            )
            self._normalize_ptr()

            input_ids = (
                torch
                .from_numpy(input_ids.astype(np.int64))
                .view(self.batch_size, self.seq_length)
                .pin_memory()
            )
            labels = (
                torch
                .from_numpy(labels.astype(np.int64))
                .view(self.batch_size, self.seq_length)
                .pin_memory()
            )
            yield input_ids, labels

    def _load_next_shard(self) -> bool:
        self.shard_idx = self.shard_idx + 1
        if self.shard_idx >= len(self.shard_files):
            return False
        self.shard = np.load(self.shard_files[self.shard_idx], mmap_mode="r")
        return True

    def _normalize_ptr(self) -> None:
        """self.ptr may exceed the length of the shard, so we need to take care of that."""
        assert self.shard is not None
        # as we ignore the last token in each shard when filling input_ids,
        # so we need to subtract 1 from the length
        while self.ptr >= self.shard.shape[0] - 1:
            self.ptr -= self.shard.shape[0] - 1
            self._load_next_shard()
