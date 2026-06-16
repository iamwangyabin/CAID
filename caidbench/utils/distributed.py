from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Iterator, Sequence

import torch
import torch.distributed as dist
from torch.utils.data import Sampler


@dataclass(frozen=True)
class DistributedState:
    enabled: bool
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    backend: str | None = None

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0


def env_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1") or 1)


def env_rank() -> int:
    return int(os.environ.get("RANK", "0") or 0)


def env_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0") or 0)


def is_distributed_ready() -> bool:
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def is_main_process() -> bool:
    return not is_distributed_ready() or dist.get_rank() == 0


def init_distributed(backend: str | None = None) -> DistributedState:
    world_size = env_world_size()
    if world_size <= 1:
        return DistributedState(enabled=False)
    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available, but WORLD_SIZE > 1.")
    selected_backend = backend or ("nccl" if torch.cuda.is_available() else "gloo")
    if selected_backend == "nccl" and not torch.cuda.is_available():
        raise RuntimeError("distributed.backend=nccl requires CUDA.")
    if not dist.is_initialized():
        dist.init_process_group(backend=selected_backend, init_method="env://")
    return DistributedState(
        enabled=True,
        rank=dist.get_rank(),
        local_rank=env_local_rank(),
        world_size=dist.get_world_size(),
        backend=selected_backend,
    )


def destroy_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def choose_device(requested: str, state: DistributedState) -> torch.device:
    requested = str(requested)
    if not state.enabled:
        return torch.device(requested if requested != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    if requested == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(state.local_rank)
        return torch.device(f"cuda:{state.local_rank}")
    if requested not in {"auto", "cpu"} and requested.startswith("cuda"):
        raise RuntimeError("A CUDA device was requested for distributed training, but CUDA is unavailable.")
    return torch.device("cpu")


def broadcast_object(value: Any, src: int = 0) -> Any:
    if not is_distributed_ready():
        return value
    payload = [value if dist.get_rank() == src else None]
    dist.broadcast_object_list(payload, src=src)
    return payload[0]


def all_gather_object(value: Any) -> list[Any]:
    if not is_distributed_ready():
        return [value]
    gathered: list[Any] = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, value)
    return gathered


def broadcast_tensor(tensor: torch.Tensor, src: int = 0, device: torch.device | str | None = None) -> torch.Tensor:
    if is_distributed_ready():
        value = broadcast_object(tensor.detach().cpu() if dist.get_rank() == src else None, src=src)
    else:
        value = tensor.detach().cpu()
    if not torch.is_tensor(value):
        raise RuntimeError("broadcast_tensor expected a tensor payload.")
    return value.to(device) if device is not None else value


def broadcast_module_state(module: torch.nn.Module, src: int = 0) -> None:
    if not is_distributed_ready():
        return
    with torch.no_grad():
        for tensor in module.state_dict().values():
            if torch.is_tensor(tensor):
                dist.broadcast(tensor, src=src)


def barrier() -> None:
    if is_distributed_ready():
        dist.barrier()


def mean_scalars(values: dict[str, float], device: torch.device | str) -> dict[str, float]:
    if not values or not is_distributed_ready():
        return values
    keys = list(values)
    tensor = torch.tensor([float(values[key]) for key in keys], dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= float(dist.get_world_size())
    return {key: float(value) for key, value in zip(keys, tensor.detach().cpu().tolist())}


class EvalShardSampler(Sampler[int]):
    """Shard evaluation indices without padding or duplicate samples."""

    def __init__(self, data_source: Sequence[Any] | Any, *, rank: int, world_size: int) -> None:
        self.data_source = data_source
        self.rank = int(rank)
        self.world_size = int(world_size)

    def __iter__(self) -> Iterator[int]:
        return iter(range(self.rank, len(self.data_source), self.world_size))

    def __len__(self) -> int:
        n = len(self.data_source)
        if self.rank >= n:
            return 0
        return (n - 1 - self.rank) // self.world_size + 1
