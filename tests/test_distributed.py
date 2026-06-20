from __future__ import annotations

import socket
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from caidbench.utils.distributed import broadcast_module_state


def _loopback_ifname() -> str | None:
    names = {name for _idx, name in socket.if_nameindex()}
    for candidate in ("lo", "lo0"):
        if candidate in names:
            return candidate
    return None


class _ModuleWithNonContiguousBuffer(nn.Module):
    def __init__(self, rank: int) -> None:
        super().__init__()
        base = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        if rank != 0:
            base.fill_(-1)
        self.register_buffer("noncontiguous", base[:, 1:])


def _broadcast_noncontiguous_state_worker(rank: int, init_method: str) -> None:
    dist.init_process_group(
        "gloo",
        init_method=init_method,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        module = _ModuleWithNonContiguousBuffer(rank)
        assert not module.noncontiguous.is_contiguous()

        broadcast_module_state(module, src=0)

        expected = torch.tensor(
            [
                [1.0, 2.0, 3.0],
                [5.0, 6.0, 7.0],
                [9.0, 10.0, 11.0],
            ]
        )
        torch.testing.assert_close(module.noncontiguous, expected)
        assert not module.noncontiguous.is_contiguous()
    finally:
        dist.destroy_process_group()


def test_broadcast_module_state_handles_noncontiguous_buffers(tmp_path, monkeypatch) -> None:
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("torch.distributed gloo backend is unavailable")
    ifname = _loopback_ifname()
    if ifname is not None:
        monkeypatch.setenv("GLOO_SOCKET_IFNAME", ifname)

    init_method = f"file://{tmp_path / 'shared_init'}"
    mp.spawn(_broadcast_noncontiguous_state_worker, args=(init_method,), nprocs=2, join=True)
