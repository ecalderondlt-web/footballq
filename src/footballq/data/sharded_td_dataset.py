"""Lazy, shard-grouped access to finalized TD-JEPA tensor manifests."""

from __future__ import annotations

import bisect
import hashlib
import json
import math
from collections import OrderedDict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset, Sampler

from footballq.data.td_jepa_dataset import TDJEPAData, TDJEPADataset, load_td_jepa_data


def _file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


class ShardedTDJEPADataset(Dataset):
    """Map-style dataset that loads finalized tensor shards on demand."""

    def __init__(
        self,
        manifest_path: str | Path,
        split: str,
        *,
        cache_size: int = 1,
        verify_hashes_on_load: bool = False,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if not self.manifest.get("tensor_hashes_complete"):
            raise ValueError("Sharded TD-JEPA manifest has not finalized tensor hashes.")
        self.split = split
        self.cache_size = max(1, int(cache_size))
        self.verify_hashes_on_load = bool(verify_hashes_on_load)
        self.root = self.manifest_path.parent.parent
        self.shards = [item for item in self.manifest["shards"] if item["split"] == split]
        if not self.shards:
            raise ValueError(f"Sharded TD-JEPA manifest contains no {split!r} examples.")
        self.shard_starts: list[int] = []
        self.shard_ends: list[int] = []
        offset = 0
        for shard in self.shards:
            self.shard_starts.append(offset)
            offset += int(shard["example_count"])
            self.shard_ends.append(offset)
        self.example_count = offset
        self._cache: OrderedDict[int, TDJEPAData] = OrderedDict()
        self.prototype = self._load_shard(0)
        self.prototype.metadata = {
            **(self.prototype.metadata or {}),
            "sharded_dataset_manifest_path": str(self.manifest_path),
            "sharded_dataset_manifest_sha256": self.manifest["manifest_payload_sha256"],
            "tensor_hashes_complete": True,
        }

    def __len__(self) -> int:
        return self.example_count

    def _load_shard(self, shard_index: int) -> TDJEPAData:
        cached = self._cache.get(shard_index)
        if cached is not None:
            self._cache.move_to_end(shard_index)
            return cached
        entry = self.shards[shard_index]
        path = self.root / entry["path"]
        if self.verify_hashes_on_load and _file_sha256(path) != entry["tensor_sha256"]:
            raise ValueError(f"TD-JEPA tensor hash mismatch: {path}")
        data = load_td_jepa_data(path)
        if len(data.match_id) != int(entry["example_count"]):
            raise ValueError(f"TD-JEPA tensor count does not match manifest: {path}")
        self._cache[shard_index] = data
        self._cache.move_to_end(shard_index)
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return data

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        shard_index = bisect.bisect_right(self.shard_ends, index)
        local_index = index - self.shard_starts[shard_index]
        data = self._load_shard(shard_index)
        return TDJEPADataset(data)[local_index]

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_cache"] = OrderedDict()
        return state


class ShardGroupedSampler(Sampler[int]):
    """Shuffle shards and their examples while retaining shard-local reads."""

    def __init__(self, dataset: ShardedTDJEPADataset, *, shuffle: bool, seed: int = 0) -> None:
        self.dataset = dataset
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.dataset)

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        self.epoch += 1
        shard_order = list(range(len(self.dataset.shards)))
        if self.shuffle:
            shard_order = torch.randperm(len(shard_order), generator=generator).tolist()
        for shard_index in shard_order:
            start = self.dataset.shard_starts[shard_index]
            end = self.dataset.shard_ends[shard_index]
            if self.shuffle:
                local_order = torch.randperm(end - start, generator=generator).tolist()
                yield from (start + local_index for local_index in local_order)
            else:
                yield from range(start, end)


def temperature_shard_allocations(
    example_counts: list[int],
    *,
    num_samples: int,
    temperature: float,
) -> list[int]:
    """Allocate a fixed sample budget with shard mass proportional to count**temperature."""

    if not example_counts or any(int(count) <= 0 for count in example_counts):
        raise ValueError("Shard example counts must be a non-empty list of positive integers.")
    if int(num_samples) <= 0:
        raise ValueError("num_samples must be positive.")
    if not 0.0 <= float(temperature) <= 1.0:
        raise ValueError("temperature must lie in [0, 1].")

    weights = [float(count) ** float(temperature) for count in example_counts]
    weight_total = sum(weights)
    raw = [float(num_samples) * weight / weight_total for weight in weights]
    allocations = [math.floor(value) for value in raw]
    remainder = int(num_samples) - sum(allocations)
    priority = sorted(
        range(len(raw)),
        key=lambda index: (-(raw[index] - allocations[index]), index),
    )
    for index in priority[:remainder]:
        allocations[index] += 1
    return allocations


class ShardTemperatureSampler(Sampler[int]):
    """Sample a fixed budget by shard temperature while retaining shard-local reads."""

    def __init__(
        self,
        dataset: ShardedTDJEPADataset,
        *,
        num_samples: int,
        temperature: float = 0.5,
        seed: int = 0,
    ) -> None:
        self.dataset = dataset
        self.num_samples = int(num_samples)
        self.temperature = float(temperature)
        self.seed = int(seed)
        self.epoch = 0
        self.allocations = temperature_shard_allocations(
            [int(shard["example_count"]) for shard in dataset.shards],
            num_samples=self.num_samples,
            temperature=self.temperature,
        )

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        self.epoch += 1
        shard_order = torch.randperm(len(self.dataset.shards), generator=generator).tolist()
        for shard_index in shard_order:
            start = self.dataset.shard_starts[shard_index]
            shard_size = self.dataset.shard_ends[shard_index] - start
            remaining = self.allocations[shard_index]
            while remaining > 0:
                local_order = torch.randperm(shard_size, generator=generator).tolist()
                take = min(remaining, shard_size)
                yield from (start + local_index for local_index in local_order[:take])
                remaining -= take
