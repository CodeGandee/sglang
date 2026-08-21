# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Lazy cache-provider registration for attention backends."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence
from typing import Any

from sglang.srt.mem_cache.cache_lifecycle import CacheLifecycleCallbacks


@dataclasses.dataclass(frozen=True, slots=True)
class AttentionBackendCacheResult:
    """Complete pool stack returned by an attention-backend cache provider."""

    max_total_num_tokens: int
    max_running_requests: int
    full_max_total_num_tokens: int | None
    swa_max_total_num_tokens: int | None
    req_to_token_pool: Any
    token_to_kv_pool: Any
    token_to_kv_pool_allocator: Any
    memory_pool_config: Any
    unified_memory_pool: Any = None
    cache_lifecycle: CacheLifecycleCallbacks | None = None


CacheProviderFactory = Callable[..., AttentionBackendCacheResult | None]


@dataclasses.dataclass(frozen=True, slots=True)
class SelectedCacheProvider:
    backend_names: tuple[str, ...]
    factory: CacheProviderFactory


ATTENTION_BACKEND_CACHE_PROVIDERS: dict[str, CacheProviderFactory] = {}


def register_attention_backend_cache_provider(name: str):
    """Register a lazy provider factory for one attention backend name."""

    normalized = name.strip()
    if not normalized:
        raise ValueError("attention backend cache provider name must not be empty")

    def decorator(factory: CacheProviderFactory):
        existing = ATTENTION_BACKEND_CACHE_PROVIDERS.get(normalized)
        if existing is not None and existing is not factory:
            raise ValueError(
                f"attention backend cache provider {normalized!r} is already registered"
            )
        ATTENTION_BACKEND_CACHE_PROVIDERS[normalized] = factory
        return factory

    return decorator


def select_attention_backend_cache_provider(
    backend_names: Sequence[str],
) -> SelectedCacheProvider | None:
    """Select one provider without invoking any registered factory."""

    matches: list[tuple[str, CacheProviderFactory]] = []
    for backend_name in dict.fromkeys(reversed(tuple(backend_names))):
        factory = ATTENTION_BACKEND_CACHE_PROVIDERS.get(backend_name)
        if factory is not None:
            matches.append((backend_name, factory))
    if not matches:
        return None
    factories = {id(factory) for _, factory in matches}
    if len(factories) > 1:
        names = ", ".join(name for name, _ in matches)
        raise ValueError(
            "split attention backends selected incompatible cache providers: " + names
        )
    return SelectedCacheProvider(
        backend_names=tuple(name for name, _ in matches),
        factory=matches[0][1],
    )
