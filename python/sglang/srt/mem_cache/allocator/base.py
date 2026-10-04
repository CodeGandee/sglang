"""
Copyright 2025 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool import KVCache


@dataclass(frozen=True)
class PreparedPageGrowth:
    """Preallocated additive page publication for one serialized allocator.

    Attributes
    ----------
    old_size, new_size : int
        Capacity transition excluding the padded page.
    previous_free, previous_release : torch.Tensor
        Exact free-list identities used to refuse a stale prepared transition.
    free_pages : torch.Tensor
        Already allocated final page list; commit performs no device allocation.
    """

    old_size: int
    new_size: int
    previous_free: torch.Tensor
    previous_release: torch.Tensor
    free_pages: torch.Tensor
    revision: int


class BaseTokenToKVPoolAllocator(abc.ABC):
    @abc.abstractmethod
    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        device: str,
        kvcache: KVCache,
        need_sort: bool,
    ):
        self.size = size
        self.page_size = page_size
        self.dtype = dtype
        self.device = device
        self._kvcache = kvcache
        self.need_sort = need_sort

        self.free_pages = None
        self.release_pages = None
        self.is_not_in_free_group = True
        self.free_group = []
        self._growth_revision = 0

    @property
    def size_full(self):
        return self.size

    def debug_print(self) -> str:
        return ""

    def available_size(self):
        return (len(self.free_pages) + len(self.release_pages)) * self.page_size

    def get_kvcache(self):
        return self._kvcache

    def free_group_begin(self):
        self.is_not_in_free_group = False
        self.free_group = []

    def free_group_end(self):
        self.is_not_in_free_group = True
        if self.free_group:
            self.free(torch.cat(self.free_group))

    @staticmethod
    def _copy_for_free_group(free_index: torch.Tensor) -> torch.Tensor:
        """Take ownership before a caller can mutate a deferred tensor view."""
        return free_index.clone()

    def merge_and_sort_free(self):
        if len(self.release_pages) > 0:
            self.free_pages = torch.cat((self.free_pages, self.release_pages))
            self.free_pages, _ = torch.sort(self.free_pages)
            self.release_pages = torch.empty(
                (0,), dtype=self.release_pages.dtype, device=self.device
            )

    def translate_kv_indices_for_transfer(
        self, kv_indices: torch.Tensor
    ) -> torch.Tensor:
        """Token ids as the PD-disaggregation transfer engine addresses them.

        Identity here: a static pool's token ids index its registered buffers
        directly. Virtual-id pools must override.
        """
        return kv_indices

    def get_cpu_copy(self, indices, mamba_indices=None):
        # FIXME: reuse the get_cpu_copy after paged allocator is implemented
        raise NotImplementedError()

    def load_cpu_copy(self, kv_cache_cpu, indices, mamba_indices=None):
        # FIXME: reuse the load_cpu_copy after paged allocator is implemented
        raise NotImplementedError()

    def alloc_extend(self, *args, **kwargs):
        raise NotImplementedError("alloc_extend is only for paged allocator")

    def alloc_decode(self, *args, **kwargs):
        raise NotImplementedError("alloc_decode is only for paged allocator")

    def resize(self, config) -> None:
        self.size = config.max_total_num_tokens
        if self.page_size > 1:
            self.num_pages = config.max_total_num_tokens // self.page_size
        self.clear()

    def _prepare_free_page_growth(self, new_size: int) -> PreparedPageGrowth:
        """Append newly backed pages without resetting live allocation state.

        Parameters
        ----------
        new_size : int
            Page-aligned capacity whose physical backing the caller has prepared.

        Notes
        -----
        Concrete simple allocators opt into this operation. Hybrid/logical
        allocators must coordinate their own dependent buffers before publishing.
        The scheduler must serialize this operation with allocation and freeing.
        """
        if new_size < self.size or new_size % self.page_size:
            raise ValueError("growth requires monotonic page-aligned capacity")
        if new_size == self.size:
            return PreparedPageGrowth(
                self.size,
                new_size,
                self.free_pages,
                self.release_pages,
                self.free_pages,
                self._growth_revision,
            )
        old_pages = self.size // self.page_size
        new_pages = new_size // self.page_size
        added = torch.arange(
            old_pages + 1,
            new_pages + 1,
            dtype=self.free_pages.dtype,
            device=self.device,
        )
        # Allocate before publishing either the new size or page list. Deferred
        # frees, active free groups and every live page retain their identities.
        free_pages = torch.cat((self.free_pages, added))
        return PreparedPageGrowth(
            self.size,
            new_size,
            self.free_pages,
            self.release_pages,
            free_pages,
            self._growth_revision,
        )

    def _publish_prepared_page_growth(self, prepared: PreparedPageGrowth) -> None:
        """Publish preallocated free pages after all participants agree ready."""
        if (
            prepared.old_size != self.size
            or prepared.previous_free is not self.free_pages
            or prepared.previous_release is not self.release_pages
            or prepared.revision != self._growth_revision
        ):
            raise ValueError("prepared allocator growth is stale")
        self.free_pages = prepared.free_pages
        self.size = prepared.new_size
        self._growth_revision += 1

    @abc.abstractmethod
    def clear(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def alloc(self, need_size: int):
        raise NotImplementedError()

    @abc.abstractmethod
    def free(self, free_index: torch.Tensor):
        raise NotImplementedError()

    def free_segment(self, free_index: torch.Tensor, *, start_pos: int):
        """Free ``kv_row[start_pos : start_pos + n]`` of one request (or a
        page-aligned copy); subclasses may use ``start_pos`` to skip the
        data-dependent dedup. Default: plain free()."""
        self.free(free_index)

    def free_segments(self, segments):
        """Free disjoint ascending ``(free_index, start_pos)`` segments of one
        request's kv row; a boundary page shared by consecutive segments is
        emitted once (the later segment's head is trimmed)."""
        ps = self.page_size
        prev_end = None
        for free_index, start_pos in segments:
            n = free_index.numel()
            if n == 0:
                continue
            seg_end = start_pos + n
            if prev_end is not None and start_pos // ps == (prev_end - 1) // ps:
                boundary = (start_pos // ps + 1) * ps
                free_index = free_index[boundary - start_pos :]
                start_pos = boundary
            prev_end = seg_end
            self.free_segment(free_index, start_pos=start_pos)
