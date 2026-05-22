from __future__ import annotations

import random
from collections import defaultdict
from typing import Any, Iterable

import torch


def detach_sample(sample: dict[str, Any], keep_x: bool = True) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in sample.items():
        if k == "x" and not keep_x:
            continue
        if torch.is_tensor(v):
            out[k] = v.detach().cpu().clone()
        elif isinstance(v, (int, float, str)):
            out[k] = v
        else:
            out[k] = v
    return out


def split_batch(batch: dict[str, Any]) -> list[dict[str, Any]]:
    n = None
    for v in batch.values():
        if torch.is_tensor(v):
            n = int(v.shape[0])
            break
        if isinstance(v, list):
            n = len(v)
            break
    if n is None:
        return []
    rows: list[dict[str, Any]] = []
    for i in range(n):
        row: dict[str, Any] = {}
        for k, v in batch.items():
            if torch.is_tensor(v):
                row[k] = v[i]
            elif isinstance(v, list):
                row[k] = v[i]
            else:
                row[k] = v
        rows.append(row)
    return rows


def collate_samples(samples: list[dict[str, Any]], device: torch.device | str | None = None) -> dict[str, Any]:
    if not samples:
        return {}
    out: dict[str, Any] = {}
    keys = samples[0].keys()
    for k in keys:
        vals = [s[k] for s in samples if k in s]
        if not vals:
            continue
        if torch.is_tensor(vals[0]):
            try:
                t = torch.stack(vals, dim=0)
            except RuntimeError:
                t = torch.as_tensor(vals)
            out[k] = t.to(device) if device is not None else t
        elif isinstance(vals[0], (int, float)):
            t = torch.tensor(vals)
            out[k] = t.to(device) if device is not None else t
        else:
            out[k] = vals
    return out


class ReplayBuffer:
    """Small exemplar memory with optional balanced replacement.

    Samples are regular CAIDBench batch rows. The buffer deliberately avoids
    dataset-specific assumptions so it can be used for CDDB, DFIL, E3, HGR,
    SUR-LID, and generic rehearsal baselines.
    """

    def __init__(self, capacity: int = 0, balanced: bool = True, seed: int = 0, group_key: str = "label") -> None:
        self.capacity = int(capacity)
        self.balanced = bool(balanced)
        self.group_key = group_key
        self.rng = random.Random(seed)
        self.samples: list[dict[str, Any]] = []

    def __len__(self) -> int:
        return len(self.samples)

    def clear(self) -> None:
        self.samples.clear()

    def state_dict(self) -> dict[str, Any]:
        return {"capacity": self.capacity, "balanced": self.balanced, "group_key": self.group_key, "samples": self.samples}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.capacity = int(state.get("capacity", self.capacity))
        self.balanced = bool(state.get("balanced", self.balanced))
        self.group_key = str(state.get("group_key", self.group_key))
        self.samples = list(state.get("samples", []))

    def add_batch(self, batch: dict[str, Any], keep_x: bool = True) -> None:
        self.add_samples(split_batch(batch), keep_x=keep_x)

    def add_samples(self, samples: Iterable[dict[str, Any]], keep_x: bool = True) -> None:
        if self.capacity <= 0:
            return
        self.samples.extend(detach_sample(s, keep_x=keep_x) for s in samples)
        self._trim()

    def replace(self, samples: Iterable[dict[str, Any]], keep_x: bool = True) -> None:
        self.samples = [detach_sample(s, keep_x=keep_x) for s in samples]
        self._trim()

    def _sample_group(self, sample: dict[str, Any]) -> str:
        if self.group_key == "label":
            key = "y"
        else:
            key = self.group_key
        v = sample.get(key, "unknown")
        if torch.is_tensor(v):
            return str(int(v.item())) if v.numel() == 1 else str(v.tolist())
        return str(v)

    def _trim(self) -> None:
        if self.capacity <= 0 or len(self.samples) <= self.capacity:
            return
        if not self.balanced:
            self.rng.shuffle(self.samples)
            self.samples = self.samples[: self.capacity]
            return
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for s in self.samples:
            groups[self._sample_group(s)].append(s)
        keys = sorted(groups)
        quota = max(self.capacity // max(len(keys), 1), 1)
        selected: list[dict[str, Any]] = []
        leftovers: list[dict[str, Any]] = []
        for k in keys:
            group = groups[k]
            self.rng.shuffle(group)
            selected.extend(group[:quota])
            leftovers.extend(group[quota:])
        remaining = self.capacity - len(selected)
        if remaining > 0:
            self.rng.shuffle(leftovers)
            selected.extend(leftovers[:remaining])
        self.rng.shuffle(selected)
        self.samples = selected[: self.capacity]

    def sample(self, n: int, device: torch.device | str | None = None) -> dict[str, Any]:
        if len(self.samples) == 0 or n <= 0:
            return {}
        if n >= len(self.samples):
            chosen = list(self.samples)
        else:
            chosen = self.rng.sample(self.samples, n)
        return collate_samples(chosen, device=device)

    def all(self, device: torch.device | str | None = None) -> dict[str, Any]:
        return collate_samples(list(self.samples), device=device)
