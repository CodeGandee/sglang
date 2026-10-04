# mapping on device memory, host memory and memory allocator

import logging
from typing import Optional

import torch

from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.kv_vmm_backing import KvVmmBufferOwner
from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool, KvBufferDesc
from sglang.srt.utils import is_cuda, is_hip

logger = logging.getLogger(__name__)

# sgl_kernel.kvcacheio is only available in CUDA/ROCm sgl-kernel builds (not XPU/MPS/NPU/CPU).
_is_cuda = is_cuda()
_is_hip = is_hip()
if _is_cuda or _is_hip:
    from sgl_kernel.kvcacheio import transfer_kv_all_layer_mla
else:

    def transfer_kv_all_layer_mla(*args, **kwargs):
        raise RuntimeError(
            "HiSparse device KV transfer requires sgl_kernel.kvcacheio (CUDA/ROCm). "
            "It is not available on this backend."
        )


class HiSparseDSATokenToKVPool(DSATokenToKVPool):
    def __init__(
        self,
        size: int,
        page_size: int,
        kv_lora_rank: int,
        dtype: torch.dtype,
        qk_rope_head_dim: int,
        layer_num: int,
        device: str,
        index_head_dim: int,
        enable_memory_saver: bool,
        kv_cache_dim: int,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
        host_to_device_ratio: int = 2,
        live_growth_initial_tokens: Optional[int] = None,
    ):
        if live_growth_initial_tokens is not None and (
            not device.startswith("cuda")
            or enable_memory_saver
            or live_growth_initial_tokens < page_size
            or live_growth_initial_tokens > size
            or live_growth_initial_tokens % page_size
            or size % page_size
        ):
            raise ValueError(
                "live HiSparse backing requires bounded page-aligned CUDA capacity"
            )
        self._live_growth_initial_tokens = live_growth_initial_tokens
        self._live_growth_owner: Optional[KvVmmBufferOwner] = None
        try:
            super().__init__(
                size=size,
                page_size=page_size,
                kv_lora_rank=kv_lora_rank,
                dtype=dtype,
                qk_rope_head_dim=qk_rope_head_dim,
                layer_num=layer_num,
                device=device,
                index_head_dim=index_head_dim,
                enable_memory_saver=enable_memory_saver,
                kv_cache_dim=kv_cache_dim,
                start_layer=start_layer,
                end_layer=end_layer,
                index_buf_size=size * host_to_device_ratio,
            )
        except BaseException:
            if self._live_growth_owner is not None:
                self._live_growth_owner.close()
            raise
        self.bytes_per_token = self.kv_cache_dim * self.dtype.itemsize

    def _create_buffers(self):
        if self._live_growth_initial_tokens is None:
            return super()._create_buffers()
        if self.custom_mem_pool:
            raise ValueError("live HiSparse backing cannot share a custom memory pool")
        dtype = self.store_dtype
        descriptors = [
            KvBufferDesc(
                f"hisparse-layer-{self.start_layer + index}",
                (self.size + self.page_size, 1, self.kv_cache_dim),
                row_bytes=self.kv_cache_dim * dtype.itemsize,
                tokens_per_row=1,
            )
            for index in range(self.layer_num)
        ]
        device_id = torch.device(self.device).index
        if device_id is None:
            device_id = torch.cuda.current_device()
        self._live_growth_owner = KvVmmBufferOwner(
            device=self.device,
            device_id=device_id,
            store_dtype=dtype,
            page_size=self.page_size,
            reserved_num_tokens=self.size,
            buffer_descs=descriptors,
        )
        self._live_growth_owner.grow_prefix(self._live_growth_initial_tokens)
        self.kv_buffer = self._live_growth_owner.tensors
        # Native dummy writes target the backed sink page. Do not initialize the
        # reserved, unbacked tail as the fixed-pool torch.zeros path does.
        for buffer in self.kv_buffer:
            buffer[: self.page_size].zero_()

    @property
    def backed_device_tokens(self) -> int:
        """Return backed usable slots independently of final logical capacity."""
        if self._live_growth_owner is None:
            return self.size
        return self._live_growth_owner.usable_num_tokens

    @property
    def live_growth_owner(self) -> Optional[KvVmmBufferOwner]:
        """Return the native backing owner for growth-epoch coordination."""
        return self._live_growth_owner

    def prepare_device_growth(self, num_tokens: int) -> None:
        """Map complete worker-local buffers before agreed page publication.

        Parameters
        ----------
        num_tokens : int
            Monotonic page-aligned usable capacity within the final ceiling.
        """
        if self._live_growth_owner is None:
            raise RuntimeError("HiSparse live growth was not enabled at construction")
        self._live_growth_owner.grow_prefix(num_tokens)

    def get_kv_size_bytes(self):
        if self._live_growth_owner is None:
            return super().get_kv_size_bytes()
        return self._live_growth_owner.backed_bytes + sum(
            buffer.nbytes for buffer in self.index_k_with_scale_buffer
        )

    def _clear_buffers(self):
        super()._clear_buffers()
        if self._live_growth_owner is not None:
            self._live_growth_owner.close()
            self._live_growth_owner = None

    def register_mapping(self, full_to_hisparse_device_index_mapping: torch.Tensor):
        self.full_to_hisparse_device_index_mapping = (
            full_to_hisparse_device_index_mapping
        )

    def translate_loc_to_hisparse_device(self, compressed_indices: torch.Tensor):
        return self.full_to_hisparse_device_index_mapping[compressed_indices]

    def _translate_loc_to_hisparse_device(self, compressed_indices: torch.Tensor):
        return self.full_to_hisparse_device_index_mapping[compressed_indices]

    def translate_loc_from_full_to_hisparse_device(self, full_indices: torch.Tensor):
        return self._translate_loc_to_hisparse_device(full_indices)

    def translate_loc_from_full_to_compressed(self, full_indices: torch.Tensor):
        return full_indices

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ):
        loc = self.translate_loc_to_hisparse_device(loc)
        super().set_kv_buffer(layer, loc, cache_k, cache_v)

    def set_mla_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k_nope: torch.Tensor,
        cache_k_rope: torch.Tensor,
    ):
        loc = self.translate_loc_to_hisparse_device(loc)
        super().set_mla_kv_buffer(layer, loc, cache_k_nope, cache_k_rope)

    def get_mla_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        dst_dtype: Optional[torch.dtype] = None,
    ):
        loc = self.translate_loc_to_hisparse_device(loc)
        return super().get_mla_kv_buffer(layer, loc, dst_dtype)

    def transfer_values_on_device(self, dst_indices, src_indices):
        transfer_kv_all_layer_mla(
            src_layers=self.data_ptrs,
            dst_layers=self.data_ptrs,
            src_indices=src_indices,
            dst_indices=dst_indices,
            item_size=self.bytes_per_token,
            num_layers=self.layer_num,
        )

    def get_cpu_copy(self, indices, mamba_indices=None):
        raise NotImplementedError("HiSparseDevicePool does not support get_cpu_copy")

    def load_cpu_copy(self, kv_cache_cpu, indices, mamba_indices=None):
        raise NotImplementedError("HiSparseDevicePool does not support load_cpu_copy")
