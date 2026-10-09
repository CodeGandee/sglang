from __future__ import annotations

import json
import logging
import os
import time
from collections import defaultdict

import torch

from sglang.srt.mem_cache.storage.mmap import alloc_mmap

logger = logging.getLogger(__name__)

_CUDA_HOST_REGISTERED_RANGES_ATTR = "_sglang_cuda_host_registered_ranges"


class HostTensorAllocator:
    def __init__(self):
        """Initialize the HostTensorAllocator."""
        self.dtype = None
        self.dims = None

    def allocate(self, dims: tuple, dtype: torch.dtype, device: str) -> torch.Tensor:
        assert (
            device == "cpu"
        ), f"HostTensorAllocator only supports CPU allocations; got device={device!r}"
        self.dtype = dtype
        self.dims = dims
        return alloc_mmap(dims, dtype)


class ShmHostTensorAllocator(HostTensorAllocator):
    def __init__(self):
        super().__init__()
        self.fds = []
        self.mms = []

    @property
    def fd(self):
        return self.fds[0] if self.fds else None

    @property
    def mm(self):
        return self.mms[0] if self.mms else None

    def allocate(self, dims: tuple, dtype: torch.dtype, device: str) -> torch.Tensor:
        assert (
            device == "cpu"
        ), f"ShmHostTensorAllocator only supports CPU allocations; got device={device!r}"
        self.dtype = dtype
        self.dims = dims
        from sglang.srt.mem_cache.storage.mmap import alloc_shm

        tensor, fd, mm = alloc_shm(dims, dtype)
        self.fds.append(fd)
        self.mms.append(mm)
        return tensor

    def __del__(self):
        for fd in getattr(self, "fds", []):
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
        self.fds = []


def get_allocator_from_storage(allocator_type):
    if allocator_type == "mooncake":
        try:
            from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import (
                MooncakeHostTensorAllocator,
            )

            return MooncakeHostTensorAllocator()
        except ImportError:
            logger.warning(
                "Mooncake's tensor allocator requires mooncake >= 0.3.8.post1. "
                "Please upgrade Mooncake by 'pip install mooncake-transfer-engine --upgrade'. "
                "Fallback to use default allocator."
            )
            return HostTensorAllocator()
    elif allocator_type == "mori":
        try:
            from sglang.srt.mem_cache.storage.umbp.umbp_host_allocator import (
                UMBPHostTensorAllocator,
            )

            return UMBPHostTensorAllocator()
        except (ImportError, RuntimeError) as exc:
            logger.warning(
                "UMBPHostTensorAllocator unavailable (%s). "
                "Falling back to torch.empty-based allocator.",
                exc,
            )
            return HostTensorAllocator()
    elif allocator_type == "shm":
        return ShmHostTensorAllocator()
    else:
        return HostTensorAllocator()


def get_allocator_type(server_args) -> str:
    backend = getattr(server_args, "hicache_storage_backend", None)
    if backend == "shm":
        return "shm"
    if backend == "dynamic":
        extra_config_str = getattr(
            server_args, "hicache_storage_backend_extra_config", None
        )
        if extra_config_str:
            try:
                config = json.loads(extra_config_str)
                if config.get("allocator") == "shm":
                    return "shm"
            except Exception:
                pass
    return backend or "default"


def _cuda_error_desc(cudart, rc) -> str:
    try:
        return str(cudart.cudaGetErrorString(torch._C._cudart.cudaError(int(rc))))
    except Exception:
        return "<error string unavailable>"


def _drain_cuda_last_error() -> None:
    """Clear latched CUDA runtime errors left by failed cudart calls.

    cudaHostRegister failures latch the error code in the calling thread's
    cudart TLS, and the next torch CUDA op (which checks the latch) then raises
    AcceleratorError for an error that was already handled. torch's cudart
    binding exposes no cudaGetLastError, so drain via ctypes on every libcudart
    already loaded in this process — each instance keeps its own TLS.
    """
    import ctypes
    import re

    try:
        with open("/proc/self/maps") as maps_file:
            libs = sorted(
                set(re.findall(r"/\S*libcudart\S*\.so\S*", maps_file.read()))
            )
    except OSError:
        return
    for lib in libs:
        try:
            handle = ctypes.CDLL(lib)
            get_last_error = handle.cudaGetLastError
            get_last_error.restype = ctypes.c_int
            get_last_error()
        except Exception:
            continue


def _cuda_host_register(buffer: torch.Tensor) -> None:
    lock_path = os.environ.get("PREDKV_HOST_REGISTER_LOCK")
    if not lock_path:
        _cuda_host_register_impl(buffer)
        return
    import fcntl

    # Serialize big registrations across ranks: concurrent cudaHostRegister
    # calls on shared hosts race inside the driver and fail intermittently.
    with open(lock_path, "w") as lock_f:
        fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
        _cuda_host_register_impl(buffer)


def _cuda_host_register_impl(buffer: torch.Tensor) -> None:
    """Pin `buffer` with cudaHostRegister, chunked and retried.

    Big single-shot registrations fail on fragmented hosts: the driver's gup
    pages array needs a high-order contiguous kmalloc (order-10 for a 4 GiB
    chunk). PREDKV_HOST_REGISTER_CHUNK_GB (float, GiB; default off = single
    call) bounds the chunk; 16 MiB floor bounds the ioctl count. Failed
    attempts are retried 12×10s; each failure's latched CUDA error is drained
    so the next torch op does not trip over it.
    """
    cudart = torch.cuda.cudart()
    base = buffer.data_ptr()
    total = buffer.numel() * buffer.element_size()
    chunk_bytes = total
    env_chunk_gb = float(os.environ.get("PREDKV_HOST_REGISTER_CHUNK_GB", "0"))
    if env_chunk_gb > 0:
        chunk_bytes = min(total, max(int(env_chunk_gb * 1024**3), 16 * 1024**2))
    registered_ranges: list[tuple[int, int]] = []
    try:
        offset = 0
        while offset < total:
            size = min(chunk_bytes, total - offset)
            ptr = base + offset
            rc = -1
            for _attempt in range(12):
                rc = int(cudart.cudaHostRegister(ptr, size, 0))
                if rc == 0:
                    break
                logger.warning(
                    "cudaHostRegister attempt %d failed (rc=%d, %s) "
                    "at offset=%d size=%d; retrying in 10s",
                    _attempt,
                    rc,
                    _cuda_error_desc(cudart, rc),
                    offset,
                    size,
                )
                _drain_cuda_last_error()
                time.sleep(10)
            if rc != 0:
                raise RuntimeError(
                    f"cudaHostRegister failed (rc={rc}, "
                    f"{_cuda_error_desc(cudart, rc)}) at offset={offset} size={size} "
                    f"(total={total}, chunk_limit={chunk_bytes}); host buffer is not "
                    f"pinned and device transfers may silently return stale data."
                )
            registered_ranges.append((ptr, size))
            offset += size
        # Keep the exact registration bases alive with the tensor. CUDA requires
        # cudaHostUnregister to receive each base pointer, not just the tensor's
        # original base once after several independent registrations.
        setattr(buffer, _CUDA_HOST_REGISTERED_RANGES_ATTR, registered_ranges)
    except Exception:
        remaining = _cuda_host_unregister_ranges(cudart, registered_ranges)
        if remaining:
            setattr(buffer, _CUDA_HOST_REGISTERED_RANGES_ATTR, remaining)
        raise


def _cuda_host_unregister_ranges(
    cudart, registered_ranges: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    """Best-effort unregister of chunked ranges; returns the ones that failed."""
    failed = []
    for ptr, _size in registered_ranges:
        rc = int(cudart.cudaHostUnregister(ptr))
        if rc != 0:
            logger.warning(
                "cudaHostUnregister failed (rc=%d, %s) for ptr=%#x",
                rc,
                _cuda_error_desc(cudart, rc),
                ptr,
            )
            failed.append((ptr, _size))
    return failed


def _cuda_host_unregister(buffer: torch.Tensor) -> None:
    cudart = torch.cuda.cudart()
    ranges = getattr(buffer, _CUDA_HOST_REGISTERED_RANGES_ATTR, None)
    if ranges:
        _cuda_host_unregister_ranges(cudart, ranges)
        return
    rc = cudart.cudaHostUnregister(buffer.data_ptr())
    if int(rc) != 0:
        # Best-effort on shutdown: warn, don't raise -- a leak is reclaimed at exit.
        logger.warning(
            "cudaHostUnregister failed (rc=%d, %s) for ptr=%#x",
            int(rc),
            _cuda_error_desc(cudart, rc),
            buffer.data_ptr(),
        )


def alloc_with_host_register(
    dims: tuple,
    dtype: torch.dtype,
    device: str,
    pin_memory: bool,
    allocator: HostTensorAllocator,
) -> torch.Tensor:
    """
    Allocate tensor and register host memory with cudaHostRegister.
    CudaHostRegister only applies when pin_memory=True.
    """
    buffer = allocator.allocate(dims, dtype=dtype, device=device)
    if pin_memory:
        _cuda_host_register(buffer)
    return buffer


def alloc_with_pin_memory(
    dims: tuple,
    dtype: torch.dtype,
    device: str,
    pin_memory: bool,
    allocator: None,
) -> torch.Tensor:
    """
    Allocate tensor using PyTorch's built-in pin_memory flag.
    """
    buffer = torch.empty(dims, dtype=dtype, device=device, pin_memory=pin_memory)
    return buffer


ALLOC_MEMORY_FUNCS = defaultdict(
    lambda: alloc_with_host_register,
    {
        "npu": alloc_with_pin_memory,
        "musa": alloc_with_pin_memory,
    },
)
