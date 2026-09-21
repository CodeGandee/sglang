# to be combined with the sparse coordinator class and sparse algorithm family

import logging
from dataclasses import dataclass
from typing import Dict, List, NamedTuple, Optional, Tuple, Union

import torch
from sglang.kernels.ops.kvcache.hisparse import (
    copy_cache_planned_mla,
    load_cache_to_device_buffer_dsv4_mla,
    load_cache_to_device_buffer_mla,
    plan_cache_to_device_buffer_mla,
    plan_prediction_staging_mla,
    publish_prediction_staging_ready_mla,
    resolve_prediction_staging_mla,
)
from sglang.srt.configs.model_config import dsa_layer_skips_topk, is_deepseek_dsa
from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.allocator.hisparse import (
    DeepSeekV4HiSparseTokenToKVPoolAllocator,
    HiSparseTokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.hisparse_memory_pool import (
    HiSparseDSATokenToKVPool,
)
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.memory_pool_host import DeepSeekV4PagedHostPool
from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost
from sglang.srt.utils import get_device_module, is_hip

device_module = get_device_module()

_is_hip = is_hip()

logger = logging.getLogger(__name__)


class HiSparseAct(NamedTuple):
    start_event: device_module.Event
    finish_event: device_module.Event
    req: Req


class HiSparseTokenStats(NamedTuple):
    device_tokens: int
    device_token_usage: float
    host_tokens: int
    host_token_usage: float


@dataclass(frozen=True)
class _HiSparseRequestIdentity:
    slot: int
    generation: int
    object_id: int


@dataclass(frozen=True)
class _HiSparsePendingPlan:
    request: _HiSparseRequestIdentity
    request_indices_ptr: int
    step: int
    anchor_layer: int
    group_layers: tuple[int, ...]
    representation: str
    num_reqs: int
    table: torch.Tensor
    miss_src: torch.Tensor
    miss_dst: torch.Tensor
    miss_count: torch.Tensor
    staging: "_HiSparseStagePlan | None" = None


@dataclass(frozen=True)
class _HiSparseStagePlan:
    """One immutable view of a prediction-derived stage fill.

    The tensors remain device-owned.  Host scalar conversion is deliberately
    deferred to the diagnostic receipt so ordinary inference does not add a
    synchronization just to account for optional staging.
    """

    slot: int
    anchor_layer: int
    logical_ids: torch.Tensor
    host_locs: torch.Tensor
    valid_count: torch.Tensor
    buffer: torch.Tensor
    step: int
    lease_epoch: int
    request: _HiSparseRequestIdentity
    project_identity: object
    representation: str
    ready_tag: torch.Tensor
    expected_tag: int
    producer_done: device_module.Event | None = None
    reader_done: device_module.Event | None = None


def _tensor_collection_nbytes(value: object) -> int:
    """Return owned tensor bytes for a tensor or nested tensor collection."""
    if isinstance(value, torch.Tensor):
        return int(value.numel() * value.element_size())
    if isinstance(value, (list, tuple)):
        return sum(_tensor_collection_nbytes(item) for item in value)
    return 0


def resolve_shared_index_layers(
    *,
    hf_text_config,
    pp_size: int,
    is_speculative: bool,
) -> Optional[List[bool]]:
    """Per-layer "reuses the previous layer's top-k index" pattern, or None.

    Mirrors DeepseekV2AttentionMLA's skip_topk derivation (index_topk_pattern /
    index_topk_freq / cli_factor); None when the model has no sharing or the
    prefetch cannot run (PP, speculative decoding, kill-switch).
    """
    if not is_deepseek_dsa(hf_text_config):
        return None
    num_layers = hf_text_config.num_hidden_layers
    cli_factor = getattr(hf_text_config, "cli_factor", 1) or 1
    if cli_factor > 1:
        pattern = [i % cli_factor != 0 for i in range(num_layers)]
    else:
        pattern = [dsa_layer_skips_topk(hf_text_config, i) for i in range(num_layers)]
    if not any(pattern):
        return None
    if pp_size != 1 or is_speculative:
        logger.warning(
            "HiSparse shared-index prefetch is unsupported under pipeline "
            "parallelism / speculative decoding; falling back to synchronous "
            "swap-in."
        )
        return None
    if envs.SGLANG_DISABLE_HISPARSE_PREFETCH.get():
        logger.info(
            "HiSparse shared-index prefetch disabled via "
            "SGLANG_DISABLE_HISPARSE_PREFETCH; using synchronous swap-in."
        )
        return None
    return pattern


def _build_prefetch_groups(
    is_shared_index_layer: List[bool],
) -> Tuple[Dict[int, List[int]], List[int]]:
    """Group consecutive shared-index (skip) layers under their anchor layer.

    Returns (groups, slot): anchor layer_id -> ordered skip layers, and each
    skip layer's position in its group (indexes the per-slot prefetch events).
    """
    groups: Dict[int, List[int]] = {}
    slot = [0] * len(is_shared_index_layer)
    anchor = None
    for i, is_shared in enumerate(is_shared_index_layer):
        if not is_shared:
            anchor = i  # compute layer; anchors the skip layers after it
            continue
        assert anchor is not None, (
            f"shared-index (skip) layer {i} has no preceding compute layer; "
            "the model's index-topk pattern is invalid"
        )
        group = groups.setdefault(anchor, [])
        slot[i] = len(group)
        group.append(i)
    return groups, slot


class HiSparseCoordinator:
    def __init__(
        self,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: Union[
            HiSparseTokenToKVPoolAllocator,
            DeepSeekV4HiSparseTokenToKVPoolAllocator,
        ],
        top_k: int,
        device_buffer_size: int,
        device: str,
        tp_group,
        host_to_device_ratio: int = 2,
        swap_in_block_size: int = 960,
        shared_index_layers: Optional[List[bool]] = None,
    ):
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.top_k = top_k
        self.device_buffer_size = device_buffer_size
        self.device = device
        self.swap_in_block_size = swap_in_block_size
        # Timing probe: skip the host->device KV bytes to measure the "IO is
        # free" floor. Produces garbage output; benchmarking only.
        self.skip_io = envs.SGLANG_DEBUG_HISPARSE_SKIP_IO.get()
        self.compress_ratio = self.token_to_kv_pool_allocator.compress_ratio

        self.is_dsv4_hisparse = isinstance(
            self.token_to_kv_pool_allocator, DeepSeekV4HiSparseTokenToKVPoolAllocator
        )
        if self.is_dsv4_hisparse:
            self.mem_pool_device = self.token_to_kv_pool_allocator.hisparse_kvcache
            page_size = self.mem_pool_device.page_size
            num_host_pages = (
                self.token_to_kv_pool_allocator.size_full // self.compress_ratio
                + page_size
                - 1
            ) // page_size
            self.mem_pool_host = DeepSeekV4PagedHostPool(
                pool_name="dsv4_hisparse_c4",
                device_buffers=self.mem_pool_device.kv_buffer,
                item_bytes=self.mem_pool_device.bytes_per_page_padded,
                num_host_pages=num_host_pages,
                slot_page_size=page_size,
                layout="layer_first",
            )
            self.item_size_bytes = (
                self.mem_pool_device.kv_cache_total_dim
                * self.mem_pool_device.store_dtype.itemsize
            )
        else:
            assert isinstance(
                self.token_to_kv_pool_allocator, HiSparseTokenToKVPoolAllocator
            )
            self.mem_pool_device: HiSparseDSATokenToKVPool = (
                self.token_to_kv_pool_allocator.get_kvcache()
            )
            self.mem_pool_host = MLATokenToKVPoolHost(
                device_pool=self.mem_pool_device,
                host_to_device_ratio=host_to_device_ratio,
                host_size=0,
                page_size=self.mem_pool_device.page_size,
                layout="layer_first",
                override_kv_cache_dim=self.mem_pool_device.kv_cache_dim,
            )
            self.item_size_bytes = self.mem_pool_host.token_stride_size
        self.page_size = self.mem_pool_device.page_size

        max_num_req_slots = req_to_token_pool.req_to_token.shape[0]
        max_context_len = req_to_token_pool.max_context_len
        max_compressed_context_len = (
            max_context_len + self.compress_ratio - 1
        ) // self.compress_ratio

        # to have an extra page for new tokens
        self.padded_buffer_size = (
            self.device_buffer_size + self.mem_pool_device.page_size
        )

        self.req_to_device_buffer = torch.zeros(
            (max_num_req_slots, self.padded_buffer_size),
            dtype=torch.int64,
            device=device,
        )
        self.req_device_buffer_size = torch.zeros(
            max_num_req_slots, dtype=torch.int64, device="cpu"
        )
        self.req_to_host_pool = torch.full(
            (max_num_req_slots, max_compressed_context_len + self.page_size),
            -1,
            dtype=torch.int64,
            device=device,
        )
        self.req_to_host_pool_allocated_len = torch.zeros(
            max_num_req_slots, dtype=torch.int64, device="cpu"
        )

        self.write_staging_stream = device_module.Stream()
        self.decode_backup_stream = device_module.Stream()
        self.ack_staging_queue: List[HiSparseAct] = []
        self.decode_producer_stream = None
        self._backup_done_event = device_module.Event()
        self._has_pending_backup = False

        self.tp_group = tp_group
        self.tp_world_size = torch.distributed.get_world_size(group=self.tp_group)

        # initialize data structures for swap-in kernel
        layer_num = self.mem_pool_device.layer_num
        self.req_device_buffer_tokens = torch.full(
            (layer_num, max_num_req_slots, self.padded_buffer_size),
            -1,
            dtype=torch.int32,
            device=device,
        )
        self.req_device_buffer_token_locs = torch.full(
            (layer_num, max_num_req_slots, self.padded_buffer_size),
            -1,
            dtype=torch.int32,
            device=device,
        )
        self._lru_init = torch.arange(
            self.device_buffer_size, dtype=torch.int16, device=device
        )
        self.lru_slots = (
            self._lru_init.view(1, 1, -1)
            .repeat(layer_num, max_num_req_slots, 1)
            .contiguous()
        )
        self._device_buffer_arange_i32 = torch.arange(
            self.device_buffer_size, dtype=torch.int32, device=device
        )

        # Pre-allocated output buffer for swap_in_selected_pages (CUDA-graph safe)
        self.top_k_device_locs_buffer = torch.full(
            (max_num_req_slots, self.top_k), -1, dtype=torch.int32, device=device
        )
        self.raw_indices_buffer = torch.full(
            (max_num_req_slots, self.top_k), -1, dtype=torch.int32, device=device
        )
        # Scalar tensor: number of real (non-padded) requests in the batch.
        # Updated before each graph replay so padded blocks early-return.
        self.num_real_reqs = torch.zeros(1, dtype=torch.int32, device=device)

        # CPU flag: True means "skip backup on the next decode step" because
        # staging already backed up all prefill tokens.  Cleared after one step.
        self._skip_first_backup = [False] * max_num_req_slots

        self._init_shared_index_prefetch(
            shared_index_layers=shared_index_layers,
            layer_num=layer_num,
            max_num_req_slots=max_num_req_slots,
        )

    def _init_shared_index_prefetch(
        self,
        shared_index_layers: Optional[List[bool]],
        layer_num: int,
        max_num_req_slots: int,
    ) -> None:
        """Set up the plan-then-IO prefetch for shared-index (IndexShare) models:
        the anchor's kernel records its miss plan and skip layers replay it on
        `prefetch_stream`, overlapping their IO with the intervening compute."""
        if shared_index_layers is not None and len(shared_index_layers) != layer_num:
            # Attention-layer count differs from num_hidden_layers (e.g. Longcat
            # doubles it): pattern would be misindexed, fall back to synchronous.
            logger.warning(
                "HiSparse shared-index prefetch disabled: pattern length %d != "
                "KV pool layer_num %d; using synchronous swap-in.",
                len(shared_index_layers),
                layer_num,
            )
            shared_index_layers = None
        self._is_shared_index_layer = list(shared_index_layers or [False] * layer_num)
        self.enable_prefetch = any(self._is_shared_index_layer)
        self._prefetch_groups, self._prefetch_slot = _build_prefetch_groups(
            self._is_shared_index_layer
        )
        if not self.enable_prefetch:
            return

        # Small fixed grid for the copy-only kernel: low SM footprint so the
        # copies overlap compute with little contention.
        self._prefetch_copy_blocks = 4
        max_group_size = max(len(g) for g in self._prefetch_groups.values())
        self.prefetch_stream = device_module.Stream()
        self._prefetch_events = [device_module.Event() for _ in range(max_group_size)]
        # Plan recorded by the current anchor, replayed by its skip layers. One
        # buffer set suffices: the last skip layer's event wait orders the next
        # anchor's writes after this group's copies.
        self._miss_src = torch.zeros(
            (max_num_req_slots, self.top_k), dtype=torch.int64, device=self.device
        )
        self._miss_dst = torch.zeros(
            (max_num_req_slots, self.top_k), dtype=torch.int32, device=self.device
        )
        self._miss_count = torch.zeros(
            (max_num_req_slots,), dtype=torch.int32, device=self.device
        )
        logger.info(
            "HiSparse: shared-index prefetch (plan-then-IO) enabled; %d anchor "
            "group(s), %d skip layer(s) of %d total.",
            len(self._prefetch_groups),
            sum(self._is_shared_index_layer),
            layer_num,
        )

    def set_decode_producer_stream(self, stream) -> None:
        self.decode_producer_stream = stream

    @property
    def split_materialization_enabled(self) -> bool:
        """Whether this instance was pinned to native plan/materialize."""
        return bool(getattr(self, "_split_materialization_enabled", False))

    @property
    def split_materialization_callable(self) -> str:
        """Return the stable receipt name for this instance's retrieval path."""
        selected = (
            self._run_split_anchor
            if self.split_materialization_enabled
            else self._run_swap_in_kernel
        )
        return f"{selected.__module__}.{selected.__qualname__}"

    @property
    def split_worker_reusable(self) -> bool:
        """Whether owned device work has a proven-safe reuse boundary."""
        return bool(getattr(self, "_split_worker_reusable", True))

    @property
    def prediction_staging_enabled(self) -> bool:
        """Whether startup-pinned synchronous prediction staging is active."""
        return bool(getattr(self, "_prediction_staging_enabled", False))

    @property
    def staging_allocation_receipt(self) -> dict[str, int | bool | str]:
        """Return fixed staging allocation accounting without tensor handles."""
        if not self.prediction_staging_enabled:
            return {
                "schema": "sglang.hisparse.prediction-staging.v1",
                "enabled": False,
                "slot_count": 0,
                "capacity_rows_per_slot": 0,
                "row_stride_bytes": int(self.item_size_bytes),
                "payload_bytes": 0,
                "metadata_bytes": 0,
                "matching_workspace_bytes": 0,
                "repair_workspace_bytes": 0,
                "observation_bytes": 0,
                "repair_capacity_rows": 0,
                "additional_device_bytes": 0,
                "admission_device_bytes": 0,
                "native_device_bytes": 0,
                "native_host_bytes": 0,
                "total_bytes": 0,
            }
        return dict(self._staging_allocation_receipt)

    @property
    def staging_runtime_receipt(self) -> dict[str, object] | None:
        """Return host-only staging lease and host-publication counters."""
        if not self.prediction_staging_enabled:
            return None
        self._retire_completed_staging_slots()
        if not hasattr(self, "_staging_admitted_counts"):
            self._staging_admitted_counts = {}
        if not hasattr(self, "_staging_admission_events"):
            self._staging_admission_events = []
        return {
            "schema": "sglang.hisparse.prediction-staging-runtime.v1",
            "slot_stats": [dict(stats) for stats in self._staging_slot_stats],
            "skipped_admissions": self._staging_skipped_admissions,
            "generated_publications": self._staging_generated_publications,
            "last_generated_positions": list(self._staging_last_generated_positions),
            "admission_events": [
                dict(event)
                for event in getattr(self, "_staging_admission_events", [])
            ],
        }

    @property
    def staging_observation(self) -> dict[str, object] | None:
        """Return a synchronized diagnostic snapshot of the last stage use.

        This property is for installed observers only.  The normal forward
        path records device counters and never calls it, so diagnostics cannot
        introduce per-layer scalar downloads into inference.
        """
        if not self.prediction_staging_enabled:
            return None
        self._synchronize_staging_observation()
        receipt = dict(self._staging_observation)
        receipt["allocation"] = self.staging_allocation_receipt
        return receipt

    def enable_split_materialization(
        self,
        *,
        logical_batch_size: int,
        eager: bool,
        staging: bool = False,
        additional_device_bytes: int = 0,
        predictive_overlap: bool = False,
    ) -> None:
        """Pin this coordinator to the admitted GLM plan/materialize path.

        Parameters
        ----------
        logical_batch_size:
            Qualified request batch size. M4.2 admits exactly one request.
        eager:
            Whether CUDA graph replay is disabled for the worker.

        Raises
        ------
        RuntimeError
            The method was called more than once, after request allocation, or
            on a coordinator outside the admitted BF16 MLA GLM profile.
        """
        if self.split_materialization_enabled:
            raise RuntimeError("HiSparse split materialization is already enabled")
        if logical_batch_size != 1 or not eager:
            raise RuntimeError(
                "HiSparse split materialization requires eager logical batch size 1"
            )
        if self.is_dsv4_hisparse:
            raise RuntimeError("HiSparse split materialization does not admit DSV4")
        if self.skip_io:
            raise RuntimeError(
                "HiSparse split materialization rejects the debug skip-IO probe"
            )
        expected = {
            "top-k": (self.top_k, 2048),
            "device buffer": (self.device_buffer_size, 4096),
            "swap block": (self.swap_in_block_size, 960),
            "TP world size": (self.tp_world_size, 1),
            "MLA row bytes": (self.item_size_bytes, 576 * 2),
            "layer count": (self.mem_pool_device.layer_num, 10),
        }
        mismatches = [
            f"{name}={actual!r}, expected {wanted!r}"
            for name, (actual, wanted) in expected.items()
            if actual != wanted
        ]
        kv_dtype = self.mem_pool_device.kv_buffer[0].dtype
        if kv_dtype != torch.bfloat16:
            mismatches.append(f"KV dtype={kv_dtype!r}, expected torch.bfloat16")
        expected_groups = {2: [3, 4, 5], 6: [7, 8, 9]}
        if not self.enable_prefetch or self._prefetch_groups != expected_groups:
            mismatches.append(
                "shared-index groups="
                f"{self._prefetch_groups!r}, expected {expected_groups!r}"
            )
        if bool(torch.any(self.req_device_buffer_size)):
            mismatches.append("request storage is already allocated")
        if mismatches:
            raise RuntimeError(
                "unsupported HiSparse split materialization profile: "
                + "; ".join(mismatches)
            )

        if not isinstance(staging, bool):
            raise TypeError("HiSparse prediction staging selection is malformed")
        if (
            isinstance(additional_device_bytes, bool)
            or not isinstance(additional_device_bytes, int)
            or additional_device_bytes < 0
        ):
            raise RuntimeError(
                "HiSparse additional device-byte reservation is malformed"
            )
        if not staging and additional_device_bytes:
            raise RuntimeError(
                "HiSparse additional device-byte reservation requires staging"
            )

        # Split-only state is installed here, not in __init__, so feature-off
        # workers retain the fused route without another state owner.
        self._split_worker_reusable = True
        self._split_generation = 0
        self._split_requests: dict[int, _HiSparseRequestIdentity] = {}
        self._split_step_request: _HiSparseRequestIdentity | None = None
        self._split_next_request: _HiSparseRequestIdentity | None = None
        self._split_aborted_generation: int | None = None
        # Native traversal ordinal: one prepared logical decode step may run
        # several complete traversals (seed, diagnostics, then real/hint).
        self._split_step = 0
        self._split_anchor_index = 0
        self._split_step_request_ptr: int | None = None
        self._split_anchor_layers = tuple(
            layer
            for layer, shared in enumerate(self._is_shared_index_layer)
            if not shared
        )
        self._split_pending: _HiSparsePendingPlan | None = None
        self._split_followers_seen = 0
        self._prediction_staging_enabled = False
        self._staging_views: dict[int, object] = {}
        self._staging_identity = None
        self._staging_target_step = -1
        self._staging_history_limit = 0
        self._staging_request_slot = -1
        self._staging_slot_next = 0
        self._staging_slot_epochs = [0, 0]
        self._staging_active: dict[int, _HiSparseStagePlan] = {}
        self._staging_admitted_counts: dict[
            int, tuple[torch.Tensor, torch.Tensor]
        ] = {}
        self._staging_admission_events: list[dict[str, int | str]] = []
        self._staging_device_records: dict[int, dict[str, object]] = {}
        self._staging_slots_busy = [False, False]
        self._staging_slot_events = (None, None)
        self._staging_observation_anchor_ids: tuple[int, ...] = ()
        self._staging_stats: dict[str, torch.Tensor] = {}
        self._staging_slot_stats = [
            {
                "slot": slot,
                "admissions": 0,
                "reuses": 0,
                "retirements": 0,
                "busy": False,
            }
            for slot in range(2)
        ]
        self._staging_skipped_admissions = 0
        self._staging_generated_publications = 0
        self._staging_last_generated_positions: tuple[int, ...] = ()
        self._staging_pending_generated_positions: tuple[int, ...] = ()
        # M4.4 resources are created only after startup explicitly enables
        # prediction staging.  The existing prefetch stream remains the
        # native-exact follower queue; speculative and urgent work get their
        # own streams and lease-scoped events.
        self._overlap_enabled = False
        self._speculative_stream = None
        self._urgent_stream = None
        self._overlap_compute_stream = None
        self._staging_ready_tags: tuple[torch.Tensor, ...] = ()
        self._staging_producer_events: tuple[object, ...] = ()
        self._staging_reader_events: tuple[object, ...] = ()
        self._urgent_done_event = None
        self._prediction_ready_event = None
        self._overlap_stream_priorities: dict[str, int] = {}
        self._staging_observation = {
            "schema": "sglang.hisparse.prediction-staging-observation.v1",
            "request_slot": -1,
            "target_step": -1,
            "identity": None,
            "anchors": {},
            "stage_h2d_bytes": 0,
            "promotion_d2d_bytes": 0,
            "repair_h2d_bytes": 0,
            "follower_bytes": 0,
            "unused_stage_bytes": 0,
            "skipped_stage_rows": 0,
        }
        if staging:
            self._enable_prediction_staging(
                additional_device_bytes,
                predictive_overlap=predictive_overlap,
            )
        self._split_materialization_enabled = True

    def _enable_prediction_staging(
        self, additional_device_bytes: int, *, predictive_overlap: bool = False
    ) -> None:
        """Allocate the bounded native-owned staging ring."""
        if self.item_size_bytes % self.mem_pool_device.kv_buffer[0].element_size():
            raise RuntimeError("HiSparse staging row stride is not element aligned")
        row_elements = (
            self.item_size_bytes // self.mem_pool_device.kv_buffer[0].element_size()
        )
        kv_shape = tuple(self.mem_pool_device.kv_buffer[0].shape[1:])
        capacity = self.top_k
        anchor_count = len(self._split_anchor_layers)
        metadata_bytes = 2 * capacity * (8 + 8 + 4) + 8 * 4
        if predictive_overlap:
            # One int64 device tag per fixed lease; tags are part of the
            # startup memory receipt rather than an unaccounted side array.
            metadata_bytes += 2 * 8
        matching_workspace_bytes = capacity * (8 + 4) + 4
        repair_workspace_bytes = capacity * (8 + 4) + 4
        observation_bytes = anchor_count * capacity * (8 + 8 + 8 + 4 + 8 + 4 + 8 + 4)
        observation_bytes += anchor_count * 6 * 4
        payload_bytes = 2 * capacity * self.item_size_bytes
        stage_owned_bytes = (
            payload_bytes
            + metadata_bytes
            + matching_workspace_bytes
            + repair_workspace_bytes
            + observation_bytes
        )
        required_bytes = stage_owned_bytes + additional_device_bytes
        device_type = torch.device(self.device).type
        if device_type == "cuda":
            free_bytes, _ = torch.cuda.mem_get_info(self.device)
            if free_bytes < required_bytes:
                raise RuntimeError(
                    "insufficient device memory for HiSparse prediction staging"
                )
        self._staging_buffers = tuple(
            torch.empty(
                (capacity, *kv_shape),
                dtype=self.mem_pool_device.kv_buffer[0].dtype,
                device=self.device,
            )
            for _ in range(2)
        )
        self._staging_logical_ids = torch.empty(
            (2, capacity), dtype=torch.int64, device=self.device
        )
        self._staging_host_locs = torch.empty(
            (2, capacity), dtype=torch.int64, device=self.device
        )
        self._staging_dst_locs = torch.empty(
            (2, capacity), dtype=torch.int32, device=self.device
        )
        self._staging_valid_counts = torch.zeros(
            (2,), dtype=torch.int32, device=self.device
        )
        self._staging_eligible_counts = torch.zeros(
            (2,), dtype=torch.int32, device=self.device
        )
        self._staging_skipped_counts = torch.zeros(
            (2,), dtype=torch.int32, device=self.device
        )
        self._staging_one_request = torch.ones(
            (1,), dtype=torch.int32, device=self.device
        )
        self._staging_zero_count = torch.zeros(
            (1,), dtype=torch.int32, device=self.device
        )
        self._staging_promotion_src = torch.empty(
            (1, capacity), dtype=torch.int64, device=self.device
        )
        self._staging_promotion_dst = torch.empty(
            (1, capacity), dtype=torch.int32, device=self.device
        )
        self._staging_promotion_count = torch.zeros(
            (1,), dtype=torch.int32, device=self.device
        )
        self._staging_repair_src = torch.empty(
            (1, capacity), dtype=torch.int64, device=self.device
        )
        self._staging_repair_dst = torch.empty(
            (1, capacity), dtype=torch.int32, device=self.device
        )
        self._staging_repair_count = torch.zeros(
            (1,), dtype=torch.int32, device=self.device
        )
        self._staging_observation_counts = torch.zeros(
            (anchor_count, 6), dtype=torch.int32, device=self.device
        )
        self._staging_observation_plans = {
            "stage_logical": torch.empty(
                (anchor_count, capacity), dtype=torch.int64, device=self.device
            ),
            "stage_source": torch.empty(
                (anchor_count, capacity), dtype=torch.int64, device=self.device
            ),
            "promotion_source": torch.empty(
                (anchor_count, capacity), dtype=torch.int64, device=self.device
            ),
            "promotion_destination": torch.empty(
                (anchor_count, capacity), dtype=torch.int32, device=self.device
            ),
            "repair_source": torch.empty(
                (anchor_count, capacity), dtype=torch.int64, device=self.device
            ),
            "repair_destination": torch.empty(
                (anchor_count, capacity), dtype=torch.int32, device=self.device
            ),
            "follower_source": torch.empty(
                (anchor_count, capacity), dtype=torch.int64, device=self.device
            ),
            "follower_destination": torch.empty(
                (anchor_count, capacity), dtype=torch.int32, device=self.device
            ),
        }
        self._staging_observation_rows = {
            layer_id: row for row, layer_id in enumerate(self._split_anchor_layers)
        }
        if device_type == "cuda":
            self._staging_slot_events = tuple(
                torch.cuda.Event(enable_timing=False) for _ in range(2)
            )
        native_device_bytes = int(self.mem_pool_device.get_kv_size_bytes())
        native_host_bytes = _tensor_collection_nbytes(self.mem_pool_host.kv_buffer)
        actual_payload_bytes = sum(
            int(buffer.numel() * buffer.element_size())
            for buffer in self._staging_buffers
        )
        actual_metadata_bytes = sum(
            int(tensor.numel() * tensor.element_size())
            for tensor in (
                self._staging_logical_ids,
                self._staging_host_locs,
                self._staging_dst_locs,
                self._staging_valid_counts,
                self._staging_eligible_counts,
                self._staging_skipped_counts,
                self._staging_one_request,
                self._staging_zero_count,
            )
        )
        if predictive_overlap:
            actual_metadata_bytes += 2 * 8
        actual_matching_workspace_bytes = sum(
            int(tensor.numel() * tensor.element_size())
            for tensor in (
                self._staging_promotion_src,
                self._staging_promotion_dst,
                self._staging_promotion_count,
            )
        )
        actual_repair_workspace_bytes = sum(
            int(tensor.numel() * tensor.element_size())
            for tensor in (
                self._staging_repair_src,
                self._staging_repair_dst,
                self._staging_repair_count,
            )
        )
        actual_observation_bytes = _tensor_collection_nbytes(
            tuple(self._staging_observation_plans.values())
        ) + _tensor_collection_nbytes(self._staging_observation_counts)
        actual_total_bytes = (
            actual_payload_bytes
            + actual_metadata_bytes
            + actual_matching_workspace_bytes
            + actual_repair_workspace_bytes
            + actual_observation_bytes
        )
        self._staging_allocation_receipt = {
            "schema": "sglang.hisparse.prediction-staging.v1",
            "enabled": True,
            "slot_count": 2,
            "capacity_rows_per_slot": capacity,
            "row_stride_bytes": int(self.item_size_bytes),
            "payload_bytes": actual_payload_bytes,
            "metadata_bytes": actual_metadata_bytes,
            "matching_workspace_bytes": actual_matching_workspace_bytes,
            "repair_workspace_bytes": actual_repair_workspace_bytes,
            "observation_bytes": actual_observation_bytes,
            "repair_capacity_rows": capacity,
            "additional_device_bytes": additional_device_bytes,
            "admission_device_bytes": actual_total_bytes + additional_device_bytes,
            "native_device_bytes": native_device_bytes,
            "native_host_bytes": native_host_bytes,
            "total_bytes": actual_total_bytes,
            "row_elements": row_elements,
        }
        self._prediction_staging_enabled = True
        if predictive_overlap:
            self._initialize_overlap_resources()

    def _initialize_overlap_resources(self) -> None:
        """Create the bounded M4.4 stream/event bundle once at startup.

        This method intentionally has no effect when staging is disabled.  A
        coordinator using the accepted fused/native route therefore retains
        the same allocations and stream topology as M4.3.
        """
        if getattr(self, "_overlap_enabled", False):
            return
        if not self.prediction_staging_enabled:
            return
        # Current-stream capture is the compute lane.  Stream priorities are
        # hints only; correctness comes from explicit event dependencies.
        self._overlap_compute_stream = device_module.current_stream()
        try:
            self._urgent_stream = device_module.Stream(priority=-1)
            self._speculative_stream = device_module.Stream(priority=0)
        except TypeError:
            # The CPU test device shim and older ROCm wrappers do not expose
            # the priority keyword.  Separate streams remain mandatory.
            self._urgent_stream = device_module.Stream()
            self._speculative_stream = device_module.Stream()
        self._overlap_stream_priorities = {
            "compute": int(getattr(self._overlap_compute_stream, "priority", 0)),
            "urgent": int(getattr(self._urgent_stream, "priority", -1)),
            "native_exact": int(getattr(self.prefetch_stream, "priority", 0)),
            "speculative": int(getattr(self._speculative_stream, "priority", 0)),
        }
        self._staging_ready_tags = tuple(
            torch.zeros((1,), dtype=torch.int64, device=self.device) for _ in range(2)
        )
        self._staging_producer_events = tuple(
            device_module.Event(enable_timing=False) for _ in range(2)
        )
        self._staging_reader_events = tuple(
            device_module.Event(enable_timing=False) for _ in range(2)
        )
        # Reuse the existing reader-event slot array for overlap mode rather
        # than introducing a second retirement owner.
        self._staging_slot_events = self._staging_reader_events
        self._urgent_done_event = device_module.Event(enable_timing=False)
        self._prediction_ready_event = device_module.Event(enable_timing=False)
        self._overlap_enabled = True

    @property
    def overlap_enabled(self) -> bool:
        """Whether the fixed M4.4 stream and ready-tag bundle is active."""
        return bool(getattr(self, "_overlap_enabled", False))

    @property
    def predictive_overlap_enabled(self) -> bool:
        """Stable startup flag for the M4.4 predictive scheduler."""
        return bool(getattr(self, "_overlap_enabled", False))

    @property
    def overlap_stream_priorities(self) -> dict[str, int]:
        """Effective stream priorities captured at overlap startup."""
        return dict(self._overlap_stream_priorities)

    def enable_predictive_overlap(
        self,
        *,
        stage_slot_count: int,
        rows_per_lease: int,
        urgent_priority: int,
        speculative_priority: int,
        native_exact_role: str,
    ) -> dict[str, int]:
        """Validate or activate the fixed M4.4 native overlap bundle.

        Startup may call this after split materialization has been enabled.  A
        repeated call is idempotent only when every bounded resource value
        matches the already-installed bundle.
        """
        if stage_slot_count != 2 or rows_per_lease != self.top_k:
            raise RuntimeError("predictive overlap lease bounds differ from native staging")
        if urgent_priority != -1 or speculative_priority != 0:
            raise RuntimeError("predictive overlap stream priorities are unsupported")
        if native_exact_role != "prefetch_stream":
            raise RuntimeError("predictive overlap native-exact role is unsupported")
        if not self.prediction_staging_enabled:
            raise RuntimeError("predictive overlap requires prediction staging")
        self._initialize_overlap_resources()
        return self.overlap_stream_priorities

    @property
    def overlap_resource_receipt(self) -> dict[str, object] | None:
        """Return immutable-shaped startup metadata for the overlap bundle."""
        if not getattr(self, "_overlap_enabled", False):
            return None
        return {
            "schema": "sglang.hisparse.predictive-overlap-resources.v1",
            "slot_count": 2,
            "capacity_rows_per_slot": int(self.top_k),
            "native_exact_stream": "prefetch_stream",
            "urgent_stream": "urgent_stream",
            "speculative_stream": "speculative_stream",
            "stream_priorities": dict(self._overlap_stream_priorities),
            "ready_tags_device_resident": True,
        }

    def bind_prediction_staging(
        self,
        req: Req,
        *,
        identity: object,
        target_step: int,
        committed_history_limit: int,
        eligible_views: dict[int, object],
    ) -> None:
        """Bind one exact project prediction view set to the live native request."""
        if not self.prediction_staging_enabled:
            return
        self._assert_split_worker_reusable()
        slot = int(getattr(req, "req_pool_idx", -1))
        native_identity = self._split_requests.get(slot)
        if native_identity is None or native_identity.object_id != id(req):
            raise RuntimeError("staging request generation is stale")
        if (
            target_step < 0
            or committed_history_limit < 0
            or committed_history_limit > self.req_to_host_pool.shape[1]
        ):
            raise RuntimeError("staging logical identity is malformed")
        if not isinstance(eligible_views, dict):
            raise TypeError("staging prediction views are malformed")
        host_map = self.req_to_host_pool[slot]
        if (
            host_map.dtype != torch.int64
            or not host_map.is_contiguous()
            or host_map.device != self._staging_buffers[0].device
        ):
            raise RuntimeError("staging host mapping is malformed")
        if int(getattr(identity, "request_pool_index", -1)) != slot:
            raise RuntimeError("staging project/native request identity differs")
        request_id = str(getattr(identity, "request_id", ""))
        native_request_id = getattr(req, "rid", None)
        if not request_id or (
            native_request_id is not None and request_id != str(native_request_id)
        ):
            raise RuntimeError("staging project/native request identity differs")
        admitted_anchors = set(self._split_anchor_layers)
        for anchor, view in eligible_views.items():
            if (
                isinstance(anchor, bool)
                or not isinstance(anchor, int)
                or anchor not in admitted_anchors
            ):
                raise RuntimeError("staging prediction anchor is not admitted")
            tag = getattr(view, "tag", None)
            expected_group = (anchor, *self._prefetch_groups.get(anchor, ()))
            if (
                tag is None
                or getattr(tag, "identity", None) != identity
                or int(getattr(tag, "target_step", -1)) != target_step
                or int(getattr(tag, "anchor", -1)) != anchor
                or int(getattr(tag, "group", -1)) != anchor
                or tuple(getattr(tag, "group_layers", ())) != expected_group
                or getattr(tag, "representation", None) != "logical_position"
                or int(getattr(tag, "committed_history_limit", -1))
                != committed_history_limit
            ):
                raise RuntimeError("staging prediction tag is stale or mis-keyed")
            logical_ids = getattr(view, "logical_ids", None)
            valid_count = getattr(view, "valid_count", None)
            if (
                not isinstance(logical_ids, torch.Tensor)
                or logical_ids.dtype != torch.int32
                or logical_ids.shape != (self.top_k,)
                or not logical_ids.is_contiguous()
                or logical_ids.device != self._staging_buffers[0].device
                or not isinstance(valid_count, torch.Tensor)
                or valid_count.dtype != torch.int32
                or valid_count.numel() != 1
                or not valid_count.is_contiguous()
                or valid_count.device != logical_ids.device
            ):
                raise RuntimeError("staging prediction view tensors are malformed")
        self._staging_identity = identity
        self._staging_target_step = int(target_step)
        self._staging_history_limit = int(committed_history_limit)
        self._staging_request_slot = slot
        self._staging_views = dict(eligible_views)
        self._staging_observation_anchor_ids = self._split_anchor_layers
        self._staging_device_records = {}
        self._staging_admitted_counts = {}
        self._staging_admission_events = []
        self._staging_observation = {
            "schema": "sglang.hisparse.prediction-staging-observation.v1",
            "request_slot": slot,
            "target_step": int(target_step),
            "identity": {
                "request_id": str(getattr(identity, "request_id", "")),
                "request_pool_index": int(
                    getattr(identity, "request_pool_index", slot)
                ),
                "generation": int(
                    getattr(identity, "generation", native_identity.generation)
                ),
            },
            "anchors": {},
            "stage_h2d_bytes": 0,
            "promotion_d2d_bytes": 0,
            "repair_h2d_bytes": 0,
            "follower_bytes": 0,
            "unused_stage_bytes": 0,
            "skipped_stage_rows": 0,
        }
        # Prediction tensors are produced before the carrier traversal.  A
        # stream event captures that dependency once; ordinary decode never
        # queries a scalar readiness value on the host.
        prediction_ready_event = getattr(self, "_prediction_ready_event", None)
        if prediction_ready_event is not None:
            prediction_ready_event.record(device_module.current_stream())

    def clear_prediction_staging(self, identity: object, target_step: int) -> None:
        """Clear the project consumer binding after the pair forward."""
        if not self.prediction_staging_enabled:
            return
        if (
            self._staging_identity == identity
            and self._staging_target_step == target_step
        ):
            self.retire_prediction_step(identity, target_step)
            self._staging_views = {}
            self._staging_identity = None
            self._staging_target_step = -1
            self._staging_history_limit = 0
            self._staging_request_slot = -1

    def retire_prediction_step(self, identity: object, target_step: int) -> bool:
        """Attempt scoped retirement after one real step has committed.

        The method only queries lease-owned events and never synchronizes the
        device.  A ``False`` result leaves the two fixed slots busy until a
        later admission observes their producer and reader completion.
        """
        if not self.prediction_staging_enabled:
            return True
        if self._staging_identity is not None and (
            self._staging_identity != identity
            or self._staging_target_step != target_step
        ):
            raise RuntimeError("predictive staging retirement identity is stale")
        self._retire_completed_staging_slots()
        return not any(self._staging_slots_busy)

    def _synchronize_staging_observation(self) -> None:
        """Synchronize only for an explicit post-forward diagnostic read."""
        device_module.current_stream().synchronize()
        if self.enable_prefetch:
            self.prefetch_stream.synchronize()
        self._retire_completed_staging_slots()
        anchors: dict[str, dict[str, object]] = {}
        totals = {
            "stage_h2d_bytes": 0,
            "promotion_d2d_bytes": 0,
            "repair_h2d_bytes": 0,
            "follower_bytes": 0,
            "unused_stage_bytes": 0,
            "skipped_stage_rows": 0,
        }
        records = self._staging_device_records
        anchor_ids = self._staging_observation_anchor_ids or tuple(records)
        for layer_id in anchor_ids:
            record = records.get(layer_id)
            if record is None:
                anchors[str(layer_id)] = {
                    "eligible_prediction_rows": 0,
                    "staged_rows": 0,
                    "promoted_rows": 0,
                    "repaired_rows": 0,
                    "follower_rows": 0,
                    "skipped_stage_rows": 0,
                    "unused_stage_rows": 0,
                    "stage_logical_ids": [],
                    "stage_source_rows": [],
                    "promotion_stage_rows": [],
                    "promotion_destination_rows": [],
                    "repair_source_rows": [],
                    "repair_destination_rows": [],
                    "follower_source_rows": [],
                    "follower_destination_rows": [],
                    "follower_bytes": 0,
                    "unused_stage_bytes": 0,
                }
                continue
            row = int(record["observation_row"])
            (
                eligible_count,
                staged_count,
                promoted_count,
                repaired_count,
                skipped_count,
                follower_count,
            ) = self._staging_observation_counts[row].detach().cpu().tolist()
            plans = self._staging_observation_plans
            receipt = {
                "eligible_prediction_rows": eligible_count,
                "staged_rows": staged_count,
                "promoted_rows": promoted_count,
                "repaired_rows": repaired_count,
                "follower_rows": 0,
                "skipped_stage_rows": skipped_count,
                "unused_stage_rows": max(0, staged_count - promoted_count),
                "stage_logical_ids": plans["stage_logical"][row, :staged_count]
                .detach()
                .cpu()
                .tolist(),
                "stage_source_rows": plans["stage_source"][row, :staged_count]
                .detach()
                .cpu()
                .tolist(),
                "promotion_stage_rows": plans["promotion_source"][row, :promoted_count]
                .detach()
                .cpu()
                .tolist(),
                "promotion_destination_rows": plans["promotion_destination"][
                    row, :promoted_count
                ]
                .detach()
                .cpu()
                .tolist(),
                "repair_source_rows": plans["repair_source"][row, :repaired_count]
                .detach()
                .cpu()
                .tolist(),
                "repair_destination_rows": plans["repair_destination"][
                    row, :repaired_count
                ]
                .detach()
                .cpu()
                .tolist(),
            }
            follower_sources = (
                plans["follower_source"][row, :follower_count].detach().cpu().tolist()
            )
            follower_destinations = (
                plans["follower_destination"][row, :follower_count]
                .detach()
                .cpu()
                .tolist()
            )
            follower_layers = int(record["follower_layers"])
            receipt["follower_source_rows"] = follower_sources * follower_layers
            receipt["follower_destination_rows"] = (
                follower_destinations * follower_layers
            )
            receipt["follower_rows"] = len(receipt["follower_source_rows"])
            receipt["follower_bytes"] = receipt["follower_rows"] * self.item_size_bytes
            receipt["unused_stage_rows"] = max(
                0, int(receipt["staged_rows"]) - int(receipt["promoted_rows"])
            )
            receipt["unused_stage_bytes"] = (
                int(receipt["unused_stage_rows"]) * self.item_size_bytes
            )
            receipt["stage_h2d_bytes"] = staged_count * self.item_size_bytes
            receipt["promotion_d2d_bytes"] = promoted_count * self.item_size_bytes
            receipt["repair_h2d_bytes"] = repaired_count * self.item_size_bytes
            anchors[str(layer_id)] = receipt
            for field in totals:
                if field == "skipped_stage_rows":
                    totals[field] += int(receipt.get(field, 0))
                elif field == "unused_stage_bytes":
                    totals[field] += int(receipt["unused_stage_bytes"])
                elif field == "follower_bytes":
                    totals[field] += int(receipt["follower_bytes"])
                else:
                    totals[field] += int(receipt.get(field, 0))
        self._staging_observation.update({"anchors": anchors, **totals})

    def _stage_prediction_rows(
        self, layer_id: int
    ) -> tuple[_HiSparseStagePlan | None, torch.Tensor, torch.Tensor]:
        """Filter one anchor and stage rows on the independent P stream."""
        view = self._staging_views.get(layer_id)
        if view is None:
            return None, self._staging_zero_count, self._staging_zero_count
        if not hasattr(self, "_staging_admitted_counts"):
            self._staging_admitted_counts = {}
        if not hasattr(self, "_staging_admission_events"):
            self._staging_admission_events = []
        existing = self._staging_active.get(layer_id)
        if existing is not None:
            counts = self._staging_admitted_counts.get(layer_id)
            if counts is None:
                raise RuntimeError("staged prediction lease lost admission metadata")
            return existing, counts[0], counts[1]
        self._retire_completed_staging_slots()
        slot = None
        for offset in range(len(self._staging_slots_busy)):
            index = (self._staging_slot_next + offset) % len(self._staging_slots_busy)
            busy = self._staging_slots_busy[index]
            if not busy:
                slot = index
                break
        valid_count = view.valid_count
        if slot is None:
            self._staging_skipped_admissions += 1
            self._staging_admission_events.append(
                {
                    "anchor": layer_id,
                    "status": "skipped",
                    "reason": "leases-busy",
                    "phase": self._staging_admission_phase(layer_id),
                }
            )
            return None, valid_count, valid_count
        self._staging_slot_next = (slot + 1) % len(self._staging_slots_busy)
        self._staging_slots_busy[slot] = True
        self._staging_slot_epochs[slot] += 1
        slot_stats = self._staging_slot_stats[slot]
        if slot_stats["admissions"]:
            slot_stats["reuses"] += 1
        slot_stats["admissions"] += 1
        slot_stats["busy"] = True
        self._staging_admission_events.append(
            {
                "anchor": layer_id,
                "status": "admitted",
                "slot": slot,
                "phase": self._staging_admission_phase(layer_id),
            }
        )
        stage = self._staging_buffers[slot]
        if not getattr(self, "_overlap_enabled", False):
            plan_prediction_staging_mla(
                logical_ids=view.logical_ids,
                valid_count=valid_count,
                host_cache_locs=self.req_to_host_pool[self._staging_request_slot],
                history_limit=self._staging_history_limit,
                staged_logical_ids=self._staging_logical_ids[slot],
                staged_host_locs=self._staging_host_locs[slot],
                staged_dst_locs=self._staging_dst_locs[slot],
                staged_count=self._staging_valid_counts[slot : slot + 1],
                eligible_count=self._staging_eligible_counts[slot : slot + 1],
                skipped_count=self._staging_skipped_counts[slot : slot + 1],
            )
            copy_cache_planned_mla(
                miss_src=self._staging_host_locs[slot : slot + 1],
                miss_dst=self._staging_dst_locs[slot : slot + 1],
                miss_count=self._staging_valid_counts[slot : slot + 1],
                num_real_reqs=self._staging_one_request,
                host_cache=self.mem_pool_host.kv_buffer[layer_id],
                device_buffer=stage,
                item_size_bytes=self.item_size_bytes,
            )
            plan = _HiSparseStagePlan(
                slot=slot,
                anchor_layer=layer_id,
                logical_ids=self._staging_logical_ids[slot],
                host_locs=self._staging_host_locs[slot],
                valid_count=self._staging_valid_counts[slot],
                buffer=stage,
                step=self._staging_target_step,
                lease_epoch=self._staging_slot_epochs[slot],
                request=self._split_requests[self._staging_request_slot],
                project_identity=self._staging_identity,
                representation="mla-bf16-width-576",
                ready_tag=self._staging_valid_counts[slot],
                expected_tag=0,
                reader_done=self._staging_slot_events[slot],
            )
            self._staging_active[layer_id] = plan
            self._staging_admitted_counts[layer_id] = (
                self._staging_eligible_counts[slot],
                self._staging_skipped_counts[slot],
            )
            return (
                plan,
                self._staging_eligible_counts[slot],
                self._staging_skipped_counts[slot],
            )
        epoch = self._staging_slot_epochs[slot]
        request = self._split_requests[self._staging_request_slot]
        expected_tag = self._encode_stage_tag(
            request,
            target_step=self._staging_target_step,
            anchor_layer=layer_id,
            lease_epoch=epoch,
        )
        ready_tag = self._staging_ready_tags[slot]
        producer_done = self._staging_producer_events[slot]
        speculative = self._speculative_stream
        if speculative is None:
            raise RuntimeError("predictive staging has no speculative stream")
        with device_module.stream(speculative):
            prediction_ready_event = getattr(self, "_prediction_ready_event", None)
            if prediction_ready_event is not None:
                speculative.wait_event(prediction_ready_event)
            # Invalidate the previous tag on the producer stream before any
            # new metadata is visible to an urgent resolver.
            ready_tag.zero_()
            plan_prediction_staging_mla(
                logical_ids=view.logical_ids,
                valid_count=valid_count,
                host_cache_locs=self.req_to_host_pool[self._staging_request_slot],
                history_limit=self._staging_history_limit,
                staged_logical_ids=self._staging_logical_ids[slot],
                staged_host_locs=self._staging_host_locs[slot],
                staged_dst_locs=self._staging_dst_locs[slot],
                staged_count=self._staging_valid_counts[slot : slot + 1],
                eligible_count=self._staging_eligible_counts[slot : slot + 1],
                skipped_count=self._staging_skipped_counts[slot : slot + 1],
            )
            copy_cache_planned_mla(
                miss_src=self._staging_host_locs[slot : slot + 1],
                miss_dst=self._staging_dst_locs[slot : slot + 1],
                miss_count=self._staging_valid_counts[slot : slot + 1],
                num_real_reqs=self._staging_one_request,
                host_cache=self.mem_pool_host.kv_buffer[layer_id],
                device_buffer=stage,
                item_size_bytes=self.item_size_bytes,
            )
            publish_prediction_staging_ready_mla(
                ready_tag=ready_tag,
                expected_tag=expected_tag,
            )
            producer_done.record(speculative)
        plan = _HiSparseStagePlan(
            slot=slot,
            anchor_layer=layer_id,
            logical_ids=self._staging_logical_ids[slot],
            host_locs=self._staging_host_locs[slot],
            valid_count=self._staging_valid_counts[slot],
            buffer=stage,
            step=self._staging_target_step,
            lease_epoch=epoch,
            request=request,
            project_identity=self._staging_identity,
            representation="mla-bf16-width-576",
            ready_tag=ready_tag,
            expected_tag=expected_tag,
            producer_done=producer_done,
            reader_done=self._staging_reader_events[slot],
        )
        self._staging_active[layer_id] = plan
        self._staging_admitted_counts[layer_id] = (
            self._staging_eligible_counts[slot],
            self._staging_skipped_counts[slot],
        )
        return (
            plan,
            self._staging_eligible_counts[slot],
            self._staging_skipped_counts[slot],
        )

    @staticmethod
    def _staging_admission_phase(layer_id: int) -> str:
        """Return the fixed M4.4 layer-timed admission phase for one anchor."""
        if layer_id in (0, 1):
            return "step-entry"
        if layer_id == 2:
            return "after-layer-0-reader"
        if layer_id == 6:
            return "after-layer-4-reader"
        return "unsupported"

    @staticmethod
    def _encode_stage_tag(
        request: _HiSparseRequestIdentity,
        *,
        target_step: int,
        anchor_layer: int,
        lease_epoch: int,
    ) -> int:
        """Pack the qualified request/step/anchor/epoch identity into int64."""
        values = (request.slot, request.generation, target_step, anchor_layer, lease_epoch)
        if any(value < 0 for value in values):
            raise RuntimeError("negative predictive stage identity")
        # The qualified profile has a 4096-slot request pool, 16-bit step and
        # generation counters, ten managed layers, and a bounded two-lease
        # epoch.  Keeping the top bit clear lets the tag live in torch.int64.
        slot, generation, step, anchor, epoch = values
        if slot >= (1 << 12) or generation >= (1 << 16) or step >= (1 << 16):
            raise RuntimeError("predictive stage identity exceeds tag capacity")
        if anchor >= (1 << 8) or epoch >= (1 << 8):
            raise RuntimeError("predictive stage anchor/epoch exceeds tag capacity")
        return (
            (slot << 51)
            | (generation << 35)
            | (step << 19)
            | (anchor << 11)
            | (epoch << 3)
            | 0x5
        )

    def _retire_completed_staging_slots(self) -> None:
        """Reclaim leases whose final-reader events have completed."""
        active_slots = {plan.slot for plan in self._staging_active.values()}
        for slot, busy in enumerate(self._staging_slots_busy):
            if not busy or slot in active_slots:
                continue
            if getattr(self, "_overlap_enabled", False):
                producer = self._staging_producer_events[slot]
                reader = self._staging_reader_events[slot]
            else:
                producer = reader = self._staging_slot_events[slot]
            if producer is not None and reader is not None and producer.query() and reader.query():
                self._staging_slots_busy[slot] = False
                self._staging_slot_stats[slot]["busy"] = False
                self._staging_slot_stats[slot]["retirements"] += 1

    def _materialize_staged_anchor(
        self,
        layer_id: int,
        plan: _HiSparseStagePlan | None,
        eligible_count: torch.Tensor,
        skipped_count: torch.Tensor,
        miss_src: torch.Tensor,
        miss_dst: torch.Tensor,
        miss_count: torch.Tensor,
        follower_layers: int,
        plan_ready_event: object | None = None,
    ) -> None:
        if plan is not None and (
            plan.anchor_layer != layer_id
            or plan.step != self._staging_target_step
            or plan.lease_epoch != self._staging_slot_epochs[plan.slot]
            or plan.request != self._split_step_request
            or plan.project_identity != self._staging_identity
            or plan.representation != "mla-bf16-width-576"
        ):
            raise RuntimeError("stale HiSparse prediction-staging lease")
        staged_host_locs = (
            plan.host_locs if plan is not None else self._staging_host_locs[0]
        )
        staged_count = (
            plan.valid_count if plan is not None else self._staging_zero_count[0]
        )
        urgent = (
            self._urgent_stream
            if getattr(self, "_overlap_enabled", False)
            else device_module.current_stream()
        )
        if urgent is None:
            raise RuntimeError("predictive staging has no urgent stream")
        with device_module.stream(urgent):
            if plan_ready_event is not None:
                urgent.wait_event(plan_ready_event)
            resolve_prediction_staging_mla(
                miss_src=miss_src,
                miss_dst=miss_dst,
                miss_count=miss_count,
                staged_host_locs=staged_host_locs,
                staged_count=staged_count.view(1),
                ready_tag=(
                    plan.ready_tag
                    if plan is not None and getattr(self, "_overlap_enabled", False)
                    else None
                ),
                expected_tag=(
                    plan.expected_tag
                    if plan is not None and getattr(self, "_overlap_enabled", False)
                    else None
                ),
                promotion_src=self._staging_promotion_src,
                promotion_dst=self._staging_promotion_dst,
                promotion_count=self._staging_promotion_count,
                repair_src=self._staging_repair_src,
                repair_dst=self._staging_repair_dst,
                repair_count=self._staging_repair_count,
            )
            if plan is not None:
                copy_cache_planned_mla(
                    miss_src=self._staging_promotion_src,
                    miss_dst=self._staging_promotion_dst,
                    miss_count=self._staging_promotion_count,
                    num_real_reqs=self._staging_one_request,
                    host_cache=plan.buffer,
                    device_buffer=self.mem_pool_device.kv_buffer[layer_id],
                    item_size_bytes=self.item_size_bytes,
                )
            copy_cache_planned_mla(
                miss_src=self._staging_repair_src,
                miss_dst=self._staging_repair_dst,
                miss_count=self._staging_repair_count,
                num_real_reqs=self._staging_one_request,
                host_cache=self.mem_pool_host.kv_buffer[layer_id],
                device_buffer=self.mem_pool_device.kv_buffer[layer_id],
                item_size_bytes=self.item_size_bytes,
            )
            urgent_done_event = getattr(self, "_urgent_done_event", None)
            if urgent_done_event is not None:
                urgent_done_event.record(urgent)
        row = self._staging_observation_rows[layer_id]
        counts = self._staging_observation_counts[row]
        counts[0].copy_(eligible_count.reshape(()))
        counts[1].copy_(staged_count)
        counts[2].copy_(self._staging_promotion_count[0])
        counts[3].copy_(self._staging_repair_count[0])
        counts[4].copy_(skipped_count.reshape(()))
        counts[5].copy_(miss_count[0])
        plans = self._staging_observation_plans
        if plan is not None:
            plans["stage_logical"][row].copy_(plan.logical_ids)
            plans["stage_source"][row].copy_(plan.host_locs)
        plans["promotion_source"][row].copy_(self._staging_promotion_src[0])
        plans["promotion_destination"][row].copy_(self._staging_promotion_dst[0])
        plans["repair_source"][row].copy_(self._staging_repair_src[0])
        plans["repair_destination"][row].copy_(self._staging_repair_dst[0])
        plans["follower_source"][row].copy_(miss_src[0])
        plans["follower_destination"][row].copy_(miss_dst[0])
        if plan is not None:
            event = plan.reader_done
            if event is not None:
                event.record(urgent)
            else:
                self._staging_slots_busy[plan.slot] = False
                self._staging_slot_stats[plan.slot]["busy"] = False
                self._staging_slot_stats[plan.slot]["retirements"] += 1
            self._staging_active.pop(layer_id, None)
            self._staging_admitted_counts.pop(layer_id, None)
        self._staging_device_records[layer_id] = {
            "observation_row": row,
            "follower_layers": follower_layers,
        }

    def abort_split_materialization(self, *, safe_to_reuse: bool) -> None:
        """Invalidate the current request after a failed split step.

        Parameters
        ----------
        safe_to_reuse:
            ``True`` only when owned stream fences remain usable. The method
            drains them before allowing a later request generation. ``False``
            quarantines this coordinator for process retirement.
        """
        if not self.split_materialization_enabled:
            return
        active = self._split_step_request
        if active is not None:
            self._split_aborted_generation = active.generation
        if not safe_to_reuse:
            self._split_worker_reusable = False
            return
        self._drain_split_materialization()
        self._split_pending = None
        self._split_followers_seen = 0

    def _drain_split_materialization(self) -> None:
        if not self.split_materialization_enabled:
            return
        try:
            current_stream = device_module.current_stream()
            if self.decode_producer_stream is not None:
                current_stream.wait_stream(self.decode_producer_stream)
            self.wait_for_pending_backup()
            self.prefetch_stream.synchronize()
            if getattr(self, "_overlap_enabled", False):
                if self._speculative_stream is not None:
                    self._speculative_stream.synchronize()
                if self._urgent_stream is not None:
                    self._urgent_stream.synchronize()
            current_stream.synchronize()
            if self.prediction_staging_enabled:
                for slot, busy in enumerate(self._staging_slots_busy):
                    if busy:
                        self._staging_slot_stats[slot]["retirements"] += 1
                    self._staging_slot_stats[slot]["busy"] = False
                self._staging_active.clear()
                getattr(self, "_staging_admitted_counts", {}).clear()
                getattr(self, "_staging_admission_events", []).clear()
                self._staging_device_records.clear()
                self._staging_slots_busy[:] = [False, False]
                self._staging_views = {}
                self._staging_identity = None
                self._staging_target_step = -1
                self._staging_history_limit = 0
                self._staging_request_slot = -1
        except BaseException:
            self._split_worker_reusable = False
            raise

    def _assert_split_worker_reusable(self) -> None:
        if not self.split_worker_reusable:
            raise RuntimeError(
                "HiSparse split materialization worker is unsafe; retire it"
            )

    def _assert_can_activate_split_request(self, req: Req) -> None:
        if not self.split_materialization_enabled:
            return
        self._assert_split_worker_reusable()
        if req.req_pool_idx is None:
            raise RuntimeError("HiSparse split request has no native request slot")
        if int(req.req_pool_idx) in self._split_requests:
            raise RuntimeError("HiSparse split request slot is already allocated")

    def _activate_split_request(self, req: Req) -> None:
        if not self.split_materialization_enabled:
            return
        self._split_generation += 1
        identity = _HiSparseRequestIdentity(
            slot=int(req.req_pool_idx),
            generation=self._split_generation,
            object_id=id(req),
        )
        self._split_requests[identity.slot] = identity

    def destroy(self) -> None:
        # Drain in-flight transfers so the buffer is idle, then unregister it.
        # See HostKVCache.destroy for why the explicit unregister matters.
        if self.split_materialization_enabled:
            self._drain_split_materialization()
        self.write_staging_stream.synchronize()
        self.decode_backup_stream.synchronize()
        if self.enable_prefetch:
            # Skip-layer copies read the pinned host pool on the prefetch stream.
            self.prefetch_stream.synchronize()
        self.mem_pool_host.destroy()

    def get_token_stats(self) -> HiSparseTokenStats:
        device_allocator = self.token_to_kv_pool_allocator.hisparse_attn_allocator
        device_capacity = device_allocator.size
        device_tokens = device_capacity - device_allocator.available_size()
        host_capacity = self.mem_pool_host.size
        host_tokens = host_capacity - self.mem_pool_host.available_size()
        return HiSparseTokenStats(
            device_tokens=device_tokens,
            device_token_usage=(
                device_tokens / device_capacity if device_capacity > 0 else 0.0
            ),
            host_tokens=host_tokens,
            host_token_usage=(
                host_tokens / host_capacity if host_capacity > 0 else 0.0
            ),
        )

    def admit_request_into_staging(self, req: Req) -> None:
        req.hisparse_staging = True

        full_kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : req.extend_range.end
        ].to(dtype=torch.int64, copy=True)
        device_indices = (
            self.mem_pool_device.translate_loc_from_full_to_hisparse_device(
                full_kv_indices
            )
        )

        prefill_len = len(device_indices)
        host_indices = self.mem_pool_host.alloc_paged_token_slots(
            self.req_to_host_pool,
            self.req_to_host_pool_allocated_len,
            req.req_pool_idx,
            0,
            prefill_len,
        )

        start_event = device_module.Event()
        finish_event = device_module.Event()
        start_event.record()
        with device_module.stream(self.write_staging_stream):
            start_event.wait(self.write_staging_stream)
            self.mem_pool_host.backup_from_device_all_layer(
                self.mem_pool_device,
                host_indices,
                device_indices,
                io_backend="kernel",
            )
            finish_event.record()
            if host_indices.is_cuda:
                host_indices.record_stream(self.write_staging_stream)
            if device_indices.is_cuda:
                device_indices.record_stream(self.write_staging_stream)

        self.ack_staging_queue.append(HiSparseAct(start_event, finish_event, req))

    def admit_request_direct(self, req: Req) -> None:
        """Direct-to-host path: KV data already resides in host pool via RDMA.

        Skips staging DMA entirely. Only allocates a small device buffer
        (4KB) for decode-time swap-in, then marks the request as ready.
        Host indices were already written to req_to_host_pool.

        Metadata fixups after alloc_device_buffer():
        - alloc_device_buffer() sets device_buffer_tokens = [0, 1, ..., buf_size-1],
          which tells the swap-in kernel that those tokens are cached in the device
          buffer.  In the staging path this is correct (prefill filled the buffer),
          but here the buffer is empty.
        """
        self.alloc_device_buffer(req)

        host_len = self.host_token_len(req.kv.kv_allocated_len)
        if host_len <= self.device_buffer_size:
            # Short sequences (seq_len <= device_buffer_size): the kernel fast path
            # returns device_buffer_locs directly without any host loading, so we
            # must preload all tokens from host pool into the device buffer
            # TODO(hzh0425): Optimize this.
            self._preload_to_device_buffer(req)
        else:
            # Long sequence: reset device_buffer_tokens to -1 so the kernel
            # sees all slots as empty -> every top-k lookup is a miss -> host load.
            self.req_device_buffer_tokens[
                :, req.req_pool_idx, : self.device_buffer_size
            ] = -1

        req.hisparse_staging = False
        self._skip_first_backup[req.req_pool_idx] = True
        logger.debug("HiSparse: admitting request %s directly", req.rid)

    def host_token_len(self, kv_allocated_len: int) -> int:
        if self.is_dsv4_hisparse:
            return kv_allocated_len // self.compress_ratio
        return kv_allocated_len

    def _preload_to_device_buffer(self, req: Req) -> None:
        """Preload all tokens from host pool into the device buffer."""
        n = self.host_token_len(req.kv.kv_allocated_len)
        host_indices = self.req_to_host_pool[req.req_pool_idx, :n]
        device_locs = self.req_to_device_buffer[req.req_pool_idx, :n]

        for layer_id in range(self.mem_pool_device.layer_num):
            self.mem_pool_host.load_to_device_per_layer(
                self.mem_pool_device,
                host_indices,
                device_locs,
                layer_id,
                io_backend="kernel",
            )

    def alloc_device_buffer(self, req: Req) -> None:
        self._assert_can_activate_split_request(req)
        if self.is_dsv4_hisparse:
            allocated_len = req.extend_range.end
            alloc_size = self.padded_buffer_size
        else:
            allocated_len = req.kv.kv_allocated_len
            page_size = self.mem_pool_device.page_size
            # Allocate only enough for current tokens (page-aligned).
            # When prefill already fills device_buffer_size, include the reserved page.
            alloc_size = min(
                ((allocated_len + page_size - 1) // page_size) * page_size,
                self.device_buffer_size,
            )
            if alloc_size == self.device_buffer_size:
                alloc_size = self.padded_buffer_size

        compressed_logical_indices = (
            self.mem_pool_device.translate_loc_from_full_to_compressed(
                self.req_to_token_pool.req_to_token[req.req_pool_idx, :allocated_len]
            )
        )
        compressed_len = len(compressed_logical_indices)

        buffer_indices = self.token_to_kv_pool_allocator.alloc_device_buffer(
            compressed_logical_indices, alloc_size
        )
        if buffer_indices is None:
            logger.error(
                "HiSparse: alloc_device_buffer failed for req %s "
                "(compressed_len=%d, alloc_size=%d)",
                req.rid,
                compressed_len,
                alloc_size,
            )
            raise RuntimeError("HiSparse alloc_device_buffer returned None")

        buffer_indices = buffer_indices.to(torch.int32)
        self.req_to_device_buffer[req.req_pool_idx, :alloc_size] = buffer_indices
        self.req_device_buffer_size[req.req_pool_idx] = alloc_size

        self.req_device_buffer_tokens[
            :, req.req_pool_idx, : self.device_buffer_size
        ] = self._device_buffer_arange_i32
        self.req_device_buffer_token_locs[:, req.req_pool_idx, :alloc_size] = (
            buffer_indices[:alloc_size]
        )
        self._activate_split_request(req)

    def _grow_device_buffers(
        self,
        seq_lens: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        req_pool_indices_cpu: torch.Tensor,
    ) -> torch.Tensor:
        """Grow device buffers for requests whose sequence length exceeds current capacity."""
        current_caps = self.req_device_buffer_size[req_pool_indices_cpu]
        short_reqs_cpu = seq_lens_cpu <= self.device_buffer_size
        needs_grow_cpu = short_reqs_cpu & (seq_lens_cpu > current_caps)

        if torch.any(needs_grow_cpu):
            page_size = self.mem_pool_device.page_size
            grow_indices = torch.where(needs_grow_cpu)[0]

            # Compute all grow sizes on CPU, then do a single bulk allocation
            req_idxs = []
            old_caps = []
            new_caps = []
            grow_sizes = []
            total_grow = 0
            for i in grow_indices.tolist():
                req_idx = int(req_pool_indices_cpu[i])
                current_cap = int(current_caps[i])
                seq_len = int(seq_lens_cpu[i])

                new_cap = min(
                    ((seq_len + page_size - 1) // page_size) * page_size,
                    self.device_buffer_size,
                )
                if new_cap == self.device_buffer_size:
                    new_cap = self.padded_buffer_size
                grow_size = new_cap - current_cap
                if grow_size <= 0:
                    continue
                req_idxs.append(req_idx)
                old_caps.append(current_cap)
                new_caps.append(new_cap)
                grow_sizes.append(grow_size)
                total_grow += grow_size

            if total_grow > 0:
                all_new_indices = (
                    self.token_to_kv_pool_allocator.hisparse_attn_allocator.alloc(
                        total_grow
                    )
                )
                if all_new_indices is None:
                    logger.error(
                        "HiSparse: _grow_device_buffers bulk alloc failed "
                        "(total_grow=%d)",
                        total_grow,
                    )
                    raise RuntimeError(
                        f"HiSparse _grow_device_buffers failed (total_grow={total_grow})"
                    )

                offset = 0
                for req_idx, current_cap, new_cap, grow_size in zip(
                    req_idxs, old_caps, new_caps, grow_sizes
                ):
                    chunk = all_new_indices[offset : offset + grow_size]
                    offset += grow_size
                    self.req_to_device_buffer[req_idx, current_cap:new_cap] = chunk
                    self.req_device_buffer_token_locs[
                        :, req_idx, current_cap:new_cap
                    ] = chunk
                    self.req_device_buffer_size[req_idx] = new_cap

        reserved_positions = (seq_lens - 1).clamp(max=self.device_buffer_size)
        return self.req_to_device_buffer[req_pool_indices, reserved_positions]

    def has_ongoing_staging(self) -> bool:
        return len(self.ack_staging_queue) > 0

    def collect_ready_reqs(self) -> List[Req]:
        ready_reqs: List[Req] = []
        if len(self.ack_staging_queue) == 0:
            return ready_reqs

        finish_count = 0
        for _, finish_event, _ in self.ack_staging_queue:
            if not finish_event.query():
                break
            finish_count += 1
        queue_size = torch.tensor(finish_count, dtype=torch.int, device="cpu")
        if self.tp_world_size > 1:
            # synchronize TP workers to make sure the same update to scheduler
            torch.distributed.all_reduce(
                queue_size,
                op=torch.distributed.ReduceOp.MIN,
                group=self.tp_group,
            )
        finish_count = int(queue_size.item())
        while finish_count > 0:
            _, _, req = self.ack_staging_queue.pop(0)
            # prepare device buffer and update req
            self.alloc_device_buffer(req)
            self._skip_first_backup[req.req_pool_idx] = True
            req.hisparse_staging = False
            finish_count -= 1
            ready_reqs.append(req)
        return ready_reqs

    def map_last_loc_to_buffer(
        self,
        seq_lens: torch.Tensor,
        out_cache_loc: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        req_pool_indices_cpu: torch.Tensor,
    ) -> None:
        self._bind_split_step_request(req_pool_indices_cpu)
        self._eager_backup_previous_token(
            seq_lens, req_pool_indices, seq_lens_cpu, req_pool_indices_cpu
        )

        if not self.is_dsv4_hisparse:
            # Grow device buffers if needed and resolve the latest-token slot.
            reserved_buffer_loc = self._grow_device_buffers(
                seq_lens, req_pool_indices, seq_lens_cpu, req_pool_indices_cpu
            )
            self.req_device_buffer_token_locs[
                :, req_pool_indices, self.device_buffer_size
            ] = reserved_buffer_loc.to(torch.int32)

            compressed_locs = self.token_to_kv_pool_allocator.get_last_loc_compressed(
                out_cache_loc
            )
            # ROCm: the decode remap creates a temporary hisparse device slot per
            # new token (via the page_size==1 allocator path). Free the stale
            # slot before pointing the mapping at the reserved device-buffer slot,
            # otherwise the temporary slots leak and corrupt later swap-in lookups.
            # CUDA keeps the original behavior: the swap-in kernel consumes only
            # top_k_device_locs, so stale mapping entries are harmless there.
            if _is_hip:
                previous_locs = self.mem_pool_device._translate_loc_to_hisparse_device(
                    compressed_locs
                )
                stale_locs = previous_locs[
                    (previous_locs > 0) & (previous_locs != reserved_buffer_loc)
                ]
                if stale_locs.numel() > 0:
                    self.token_to_kv_pool_allocator.free_hisparse_indices(stale_locs)

            self.mem_pool_device.full_to_hisparse_device_index_mapping[
                compressed_locs
            ] = reserved_buffer_loc
            return

        active_reqs = seq_lens % self.compress_ratio == 0
        if not torch.any(active_reqs):
            return

        active_seq_lens = seq_lens[active_reqs]
        active_out_cache_loc = out_cache_loc[active_reqs]
        active_req_pool_indices = req_pool_indices[active_reqs]

        compressed_seq_lens = active_seq_lens // self.compress_ratio
        reserved_positions = (compressed_seq_lens - 1).clamp(
            max=self.device_buffer_size
        )
        reserved_buffer_loc = self.req_to_device_buffer[
            active_req_pool_indices, reserved_positions
        ]

        self.req_device_buffer_token_locs[
            :, active_req_pool_indices, self.device_buffer_size
        ] = reserved_buffer_loc.to(torch.int32)

        compressed_locs = self.token_to_kv_pool_allocator.get_last_loc_compressed(
            active_out_cache_loc
        )
        self.mem_pool_device.full_to_hisparse_device_index_mapping[compressed_locs] = (
            reserved_buffer_loc
        )

    def _bind_split_step_request(self, req_pool_indices_cpu: torch.Tensor) -> None:
        if not self.split_materialization_enabled:
            return
        self._assert_split_worker_reusable()
        if req_pool_indices_cpu.numel() != 1:
            raise RuntimeError("HiSparse split materialization requires B1")
        request_slot = int(req_pool_indices_cpu[0])
        identity = self._split_requests.get(request_slot)
        if identity is None:
            raise RuntimeError(
                "HiSparse split step has no allocated request generation"
            )
        if self._split_aborted_generation == identity.generation:
            raise RuntimeError("aborted HiSparse request cannot bind another step")
        if self._split_anchor_index != 0:
            raise RuntimeError("HiSparse split request changed during a partial step")
        self._split_next_request = identity

    def bind_split_materialization_request(self, req: Req) -> None:
        """Authorize a validated request to continue split materialization.

        Parameters
        ----------
        req:
            Live native request already validated by the owning execution path.

        Raises
        ------
        RuntimeError
            The request generation is stale or aborted, or the preceding
            native traversal has not reached its fully consumed final group.
        """
        if not self.split_materialization_enabled:
            return
        self._assert_split_worker_reusable()
        request_slot = getattr(req, "req_pool_idx", None)
        if request_slot is None:
            raise RuntimeError("HiSparse split request has no native request slot")
        identity = self._split_requests.get(int(request_slot))
        if identity is None or identity.object_id != id(req):
            raise RuntimeError(
                "stale request generation cannot bind HiSparse split materialization"
            )
        if self._split_aborted_generation == identity.generation:
            raise RuntimeError("aborted HiSparse request cannot bind another step")
        if self._split_anchor_index != 0:
            raise RuntimeError("HiSparse split request changed during a partial step")

        pending = self._split_pending
        if pending is not None:
            self._validate_split_plan(pending)
            follower_count = len(pending.group_layers) - 1
            if (
                pending.anchor_layer != self._split_anchor_layers[-1]
                or pending.step != self._split_step
                or self._split_followers_seen != follower_count
            ):
                raise RuntimeError(
                    "HiSparse split request bound before the prior traversal completed"
                )

        bound = self._split_next_request
        if bound is not None:
            if bound != identity:
                raise RuntimeError("HiSparse split request binding changed generation")
            return
        if pending is None or self._split_step_request != identity:
            raise RuntimeError(
                "HiSparse split continuation has no completed request traversal"
            )
        self._split_next_request = identity

    def _eager_backup_previous_token(
        self,
        seq_lens: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        req_pool_indices_cpu: torch.Tensor,
    ) -> None:
        """Back up the previous compressed token to host memory.

        Each newly produced compressed token (one per `compress_ratio` decode
        steps) must be backed up to host so the swap-in kernel can later
        recover it.

        Two cases are skipped:
        - The first decode step right after staging: all prefill tokens were
          already backed up during staging, so there is nothing new to save.
        - Steps where `(seq_len - 1) % compress_ratio != 0`: no new compressed
          token was produced this step.
        """
        # Build the list of batch positions that need a host backup.
        # Skip the first decode step after staging (prefill already backed up),
        # and skip non-aligned steps that did not produce a new compressed token.
        backup_indices = []
        for i in range(len(seq_lens_cpu)):
            req_idx = int(req_pool_indices_cpu[i])
            if self._skip_first_backup[req_idx]:
                self._skip_first_backup[req_idx] = False
                continue
            if (int(seq_lens_cpu[i]) - 1) % self.compress_ratio == 0:
                backup_indices.append(i)

        if not backup_indices:
            return

        backup_indices_gpu = torch.tensor(
            backup_indices, dtype=torch.int64, device=self.device
        )
        backup_req_indices = req_pool_indices[backup_indices_gpu]

        # The previous compressed token's position and its device buffer slot:
        #  compressed_pos = (seq_len - 1) // compress_ratio - 1
        #  - short: slot = compressed_pos          (within the regular buffer)
        #  - long:  slot = device_buffer_size      (the reserved slot)
        prev_seq_lens = seq_lens[backup_indices_gpu] - 1
        compressed_prev_seq_lens = prev_seq_lens // self.compress_ratio
        actual_compressed_pos = compressed_prev_seq_lens - 1

        buffer_slot = actual_compressed_pos.clamp(max=self.device_buffer_size)

        device_locs = self.req_to_device_buffer[backup_req_indices, buffer_slot]

        host_locs_list = []
        for i in backup_indices:
            req_idx = int(req_pool_indices_cpu[i])
            start_pos = (int(seq_lens_cpu[i]) - 1) // self.compress_ratio - 1
            host_locs = self.mem_pool_host.alloc_paged_token_slots(
                self.req_to_host_pool,
                self.req_to_host_pool_allocated_len,
                req_idx,
                start_pos,
                1,
            )
            host_locs_list.append(host_locs)
        host_locs = torch.cat(host_locs_list)

        self.wait_for_pending_backup()
        schedule_stream = device_module.current_stream()
        with device_module.stream(self.decode_backup_stream):
            self.decode_backup_stream.wait_stream(schedule_stream)
            if self.decode_producer_stream is not None:
                self.decode_backup_stream.wait_stream(self.decode_producer_stream)
            self.mem_pool_host.backup_from_device_all_layer(
                self.mem_pool_device,
                host_locs,
                device_locs,
                io_backend="kernel",
            )
            self._backup_done_event.record()
            if host_locs.is_cuda:
                host_locs.record_stream(self.decode_backup_stream)
            if backup_req_indices.is_cuda:
                backup_req_indices.record_stream(self.decode_backup_stream)
            if actual_compressed_pos.is_cuda:
                actual_compressed_pos.record_stream(self.decode_backup_stream)
            if device_locs.is_cuda:
                device_locs.record_stream(self.decode_backup_stream)
        self._has_pending_backup = True
        if self.prediction_staging_enabled:
            self._staging_pending_generated_positions = tuple(
                (int(seq_lens_cpu[index]) - 1) // self.compress_ratio - 1
                for index in backup_indices
            )

    def wait_for_pending_backup(self) -> None:
        if not self._has_pending_backup:
            return
        self._backup_done_event.wait(device_module.current_stream())
        self._has_pending_backup = False
        if self.prediction_staging_enabled:
            positions = self._staging_pending_generated_positions
            self._staging_generated_publications += len(positions)
            self._staging_last_generated_positions = positions
            self._staging_pending_generated_positions = ()

    def naive_load_topk(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        top_k_tokens: torch.Tensor,
        layer_id: int,
    ) -> torch.Tensor:
        """Load top-k selected tokens into device memory and return their device indices.

        This is a naive per-request loop implementation for debugging/validation.
        Production code uses swap_in_selected_pages (JIT CUDA kernel) instead.

        Note: dsv4 hisparse is not supported — DeepSeekV4SingleKVPoolHost has no
        load_to_device_per_layer and indices live in compressed space. Currently
        only used as a kernel oracle in test_hisparse_unit.py (non-dsv4 path).

        Args:
            req_pool_indices: Pool indices for each request.  Shape: (num_reqs,)
            seq_lens: Sequence lengths for each request.  Shape: (num_reqs,)
            top_k_tokens: Selected token positions per request.  Shape: (num_reqs, top_k)
            layer_id: The layer to load KV cache for.

        Returns:
            Device KV cache indices for the selected tokens.  Shape: (num_reqs, top_k)
        """
        assert not self.is_dsv4_hisparse, (
            "naive_load_topk is not implemented for dsv4 hisparse"
        )
        num_reqs = req_pool_indices.size(0)
        top_k_indices = torch.full(
            (num_reqs, self.top_k), -1, dtype=torch.int32, device=self.device
        )

        for i in range(num_reqs):
            seq_len = int(seq_lens[i].item())
            top_n = min(seq_len, self.top_k)
            if top_n == 0:
                continue

            req_idx = int(req_pool_indices[i].item())
            selected_tokens = top_k_tokens[i, :top_n].to(dtype=torch.int64)

            assert torch.all(selected_tokens >= 0), (
                f"Req {req_idx}: selected tokens contain negative positions"
            )
            assert torch.all(selected_tokens < seq_len), (
                f"Req {req_idx}: selected tokens {selected_tokens.tolist()} "
                f"out of range for seq_len={seq_len}"
            )

            if seq_len <= self.device_buffer_size:
                device_indices = self.req_to_device_buffer[req_idx, selected_tokens]
            else:
                device_indices = torch.empty(
                    top_n, dtype=torch.int64, device=self.device
                )

                is_latest_token = selected_tokens == (seq_len - 1)
                needs_host_load = ~is_latest_token

                device_indices[is_latest_token] = self.req_to_device_buffer[
                    req_idx, self.device_buffer_size
                ]

                num_to_load = int(needs_host_load.sum().item())
                if num_to_load > 0:
                    tokens_to_load = selected_tokens[needs_host_load]
                    host_locs = self.req_to_host_pool[req_idx, tokens_to_load]

                    invalid_mask = host_locs < 0
                    if torch.any(invalid_mask):
                        bad_positions = tokens_to_load[invalid_mask].tolist()
                        raise AssertionError(
                            f"Req {req_idx} (seq_len={seq_len}, layer={layer_id}): "
                            f"missing host backup at token positions {bad_positions}"
                        )

                    buffer_locs = self.req_to_device_buffer[req_idx, :num_to_load]
                    device_indices[needs_host_load] = buffer_locs

                    self.mem_pool_host.load_to_device_per_layer(
                        self.mem_pool_device,
                        host_locs,
                        buffer_locs,
                        layer_id,
                        io_backend="kernel",
                    )

            top_k_indices[i, :top_n] = device_indices.to(torch.int32)

        return top_k_indices

    def abort_staging_request(self, req: Req) -> None:
        """Remove a request from the staging queue and free its host + device resources.

        Must be called when aborting a request that has been admitted into staging
        but has not yet completed (i.e. req.hisparse_staging is True).
        """
        # Remove from staging queue
        self.ack_staging_queue = [
            act for act in self.ack_staging_queue if act.req is not req
        ]
        if getattr(self, "_overlap_enabled", False):
            # Cancellation is the one path allowed to drain owned CUDA work.
            # Do it before native request and host-pool rows are returned so a
            # late speculative writer cannot touch a reused request slot.
            self._drain_split_materialization()
        # Wait for any in-flight staging DMA to complete before freeing
        self.write_staging_stream.synchronize()

        prefill_len = req.extend_range.end
        allocated_locs = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, :prefill_len
        ]
        self.token_to_kv_pool_allocator.free_hisparse(allocated_locs)

        # Free host memory that was allocated during admit_request_into_staging
        host_indices = self.mem_pool_host.allocated_host_indices(
            self.req_to_host_pool,
            req.req_pool_idx,
            self.req_to_host_pool_allocated_len[req.req_pool_idx],
        )
        if host_indices.numel() > 0:
            self.mem_pool_host.free(host_indices)
        self.req_to_host_pool[req.req_pool_idx, :] = -1
        self.req_to_host_pool_allocated_len[req.req_pool_idx] = 0
        self._skip_first_backup[req.req_pool_idx] = False
        req.hisparse_staging = False

    def retract_req(self, req: Req) -> None:
        if req.hisparse_staging:
            self.abort_staging_request(req)
        else:
            self.request_finished(req)

    def _prepare_split_request_release(self, req: Req) -> bool:
        if not self.split_materialization_enabled:
            return True
        identity = self._split_requests.get(int(req.req_pool_idx))
        if identity is None:
            # Coordinator cleanup is idempotent after the request-owned native
            # storage has already been returned.
            return bool(
                int(self.req_device_buffer_size[req.req_pool_idx])
                or int(self.req_to_host_pool_allocated_len[req.req_pool_idx])
            )
        if identity.object_id != id(req):
            raise RuntimeError(
                "stale request generation cannot release HiSparse split storage"
            )
        self._assert_split_worker_reusable()
        if self._split_step_request == identity:
            self._drain_split_materialization()
        return True

    def _finish_split_request_release(self, req: Req) -> None:
        if not self.split_materialization_enabled:
            return
        request_slot = int(req.req_pool_idx)
        identity = self._split_requests.get(request_slot)
        if identity is not None and identity.object_id != id(req):
            raise RuntimeError(
                "HiSparse split request identity changed during native release"
            )
        self._split_requests.pop(request_slot, None)
        if self._split_next_request == identity:
            self._split_next_request = None
        if self._split_step_request == identity:
            self._split_step_request = None
            self._split_pending = None
            self._split_followers_seen = 0
            if self._split_aborted_generation == identity.generation:
                self._split_aborted_generation = None
            self._split_anchor_index = 0
            self._split_step_request_ptr = None

    def request_finished(self, req: Req):
        if not self._prepare_split_request_release(req):
            return
        # release resources only after the execution of a potential overlapped batch
        if self.decode_producer_stream is not None:
            device_module.current_stream().wait_stream(self.decode_producer_stream)
        self.wait_for_pending_backup()

        # Use kv_allocated_len (not seqlen): under speculative decoding the
        # allocator can over-allocate beyond the committed seqlen, and those
        # extra slots may carry stale mapping entries pointing at buffer slots
        # we just freed via free_hisparse_indices(all_hi). If left set, the
        # subsequent release_kv_cache -> allocator.free -> free_hisparse path
        # re-frees them (double-free into the page allocator's free list).
        allocated_len = req.kv.kv_allocated_len

        # release memory -- only free actually-allocated buffer indices
        current_cap = int(self.req_device_buffer_size[req.req_pool_idx])
        if current_cap > 0:
            side_buf_hi = self.req_to_device_buffer[req.req_pool_idx, :current_cap]
            all_hi = torch.unique(side_buf_hi[side_buf_hi > 0])
            if all_hi.numel() > 0:
                self.token_to_kv_pool_allocator.free_hisparse_indices(all_hi)

        allocated_locs = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, :allocated_len
        ]
        compressed_locs = self.mem_pool_device.translate_loc_from_full_to_compressed(
            allocated_locs
        )
        self.mem_pool_device.full_to_hisparse_device_index_mapping[compressed_locs] = 0

        host_indices = self.mem_pool_host.allocated_host_indices(
            self.req_to_host_pool,
            req.req_pool_idx,
            self.req_to_host_pool_allocated_len[req.req_pool_idx],
        )
        if host_indices.numel() > 0:
            self.mem_pool_host.free(host_indices)

        # clear req info
        self.req_device_buffer_tokens[:, req.req_pool_idx, :] = -1
        self.req_device_buffer_token_locs[:, req.req_pool_idx, :] = -1
        self.req_to_device_buffer[req.req_pool_idx, :] = 0
        self.req_device_buffer_size[req.req_pool_idx] = 0
        self.req_to_host_pool[req.req_pool_idx, :] = -1
        self.req_to_host_pool_allocated_len[req.req_pool_idx] = 0
        self.lru_slots[:, req.req_pool_idx, :].copy_(self._lru_init)
        self._skip_first_backup[req.req_pool_idx] = False
        self._finish_split_request_release(req)

    def _run_swap_in_kernel(
        self,
        req_pool_indices: torch.Tensor,
        compressed_seq_lens: torch.Tensor,
        top_k_result: torch.Tensor,
        layer_id: int,
        record_plan: bool = False,
    ) -> torch.Tensor:
        """Run the full plan+IO swap-in kernel for one layer; return its slot table.

        record_plan (set on the anchor of a shared-index group) also records the
        miss plan into self._miss_{src,dst,count} for the skip layers to replay.
        """
        num_reqs = req_pool_indices.size(0)
        top_k_indices = self.top_k_device_locs_buffer[:num_reqs]

        swap_in_fn = (
            load_cache_to_device_buffer_dsv4_mla
            if self.is_dsv4_hisparse
            else load_cache_to_device_buffer_mla
        )
        plan = (
            dict(
                miss_src=self._miss_src[:num_reqs],
                miss_dst=self._miss_dst[:num_reqs],
                miss_count=self._miss_count[:num_reqs],
            )
            if record_plan
            else {}
        )
        swap_in_fn(
            top_k_tokens=top_k_result,
            device_buffer_tokens=self.req_device_buffer_tokens[layer_id],
            host_cache_locs=self.req_to_host_pool,
            device_buffer_locs=self.req_device_buffer_token_locs[layer_id],
            host_cache=self.mem_pool_host.kv_buffer[layer_id],
            device_buffer=self.mem_pool_device.kv_buffer[layer_id],
            top_k_device_locs=top_k_indices,
            req_pool_indices=req_pool_indices,
            seq_lens=compressed_seq_lens,
            lru_slots=self.lru_slots[layer_id],
            item_size_bytes=self.item_size_bytes,
            num_top_k=self.top_k,
            hot_buffer_size=self.device_buffer_size,
            page_size=1,
            block_size=self.swap_in_block_size,
            num_real_reqs=self.num_real_reqs,
            skip_io=self.skip_io,
            **plan,
        )
        return top_k_indices

    def _run_copy_only_kernel(self, num_reqs: int, skip_layer: int) -> None:
        """Replay the anchor's recorded miss plan into a skip layer's buffers
        (IO-only; the anchor's slot table stays valid -- lockstep layout)."""
        copy_cache_planned_mla(
            miss_src=self._miss_src[:num_reqs],
            miss_dst=self._miss_dst[:num_reqs],
            miss_count=self._miss_count[:num_reqs],
            num_real_reqs=self.num_real_reqs,
            host_cache=self.mem_pool_host.kv_buffer[skip_layer],
            device_buffer=self.mem_pool_device.kv_buffer[skip_layer],
            item_size_bytes=self.item_size_bytes,
            num_blocks=self._prefetch_copy_blocks,
            is_dsv4_layout=self.is_dsv4_hisparse,
            skip_io=self.skip_io,
        )

    def _validate_split_plan(self, plan: _HiSparsePendingPlan) -> None:
        active = self._split_step_request
        if active is None or plan.request != active:
            raise RuntimeError("stale HiSparse pending-plan request generation")
        if self._split_aborted_generation == active.generation:
            raise RuntimeError("aborted HiSparse request cannot consume a pending plan")
        if plan is not self._split_pending:
            raise RuntimeError("stale HiSparse pending-plan handle")
        if plan.representation != "mla-bf16-width-576":
            raise RuntimeError("HiSparse pending-plan representation changed")

    def _retire_split_plan_before_anchor(self) -> _HiSparsePendingPlan | None:
        pending = self._split_pending
        if pending is None:
            return None
        self._validate_split_plan(pending)
        follower_count = len(pending.group_layers) - 1
        if self._split_followers_seen != follower_count:
            raise RuntimeError(
                "HiSparse pending plan reached the next anchor before every follower"
            )
        # The next anchor is submitted on the compute stream after the prior
        # real and private-hint reads. Every follower also made that stream wait
        # for its copy event, so the borrowed buffers are now reusable.
        self._split_pending = None
        self._split_followers_seen = 0
        return pending

    def _run_split_anchor(
        self,
        req_pool_indices: torch.Tensor,
        compressed_seq_lens: torch.Tensor,
        top_k_result: torch.Tensor,
        layer_id: int,
    ) -> torch.Tensor:
        self._assert_split_worker_reusable()
        if req_pool_indices.size(0) != 1:
            raise RuntimeError("HiSparse split materialization requires B1")

        retired = self._retire_split_plan_before_anchor()
        expected_layer = self._split_anchor_layers[self._split_anchor_index]
        if layer_id != expected_layer:
            raise RuntimeError(
                "HiSparse split anchor order changed: "
                f"received {layer_id}, expected {expected_layer}"
            )
        request_indices_ptr = req_pool_indices.data_ptr()
        if self._split_anchor_index == 0:
            active = self._split_next_request
            if active is None:
                active = self._split_step_request
                completed_reentry = (
                    retired is not None
                    and active is not None
                    and retired.request == active
                    and retired.anchor_layer == self._split_anchor_layers[-1]
                    and retired.step == self._split_step
                    and self._split_requests.get(active.slot) == active
                    and self._split_aborted_generation != active.generation
                )
                if not completed_reentry:
                    raise RuntimeError(
                        "HiSparse split step has no bound request generation"
                    )
                if (
                    retired.request_indices_ptr != request_indices_ptr
                    or self._split_step_request_ptr != request_indices_ptr
                ):
                    self._split_aborted_generation = active.generation
                    raise RuntimeError(
                        "HiSparse completed-step reentry changed request identity"
                    )
            self._split_step_request = active
            self._split_next_request = None
        else:
            active = self._split_step_request
        if active is None:
            raise RuntimeError("HiSparse split materialization has no live request")
        if self._split_aborted_generation == active.generation:
            raise RuntimeError("aborted HiSparse request cannot start another plan")
        if self._split_anchor_index == 0:
            self._split_step += 1
            self._split_step_request_ptr = request_indices_ptr
        elif request_indices_ptr != self._split_step_request_ptr:
            raise RuntimeError("HiSparse split request tensor changed within a step")

        num_reqs = req_pool_indices.size(0)
        table = self.top_k_device_locs_buffer[:num_reqs]
        group = (layer_id, *self._prefetch_groups.get(layer_id, []))
        try:
            stage_plan = None
            eligible_count = None
            skipped_count = None
            if self.prediction_staging_enabled:
                # Optional staging reads only published host history and completes
                # into isolated storage before authoritative placement can change.
                self.wait_for_pending_backup()
                stage_plan, eligible_count, skipped_count = self._stage_prediction_rows(
                    layer_id
                )
                if getattr(self, "_overlap_enabled", False) and layer_id == 0:
                    # Fresh anchors 0 and 1 are admitted at step entry. Anchor 1
                    # is retained in its own lease until layer 1 consumes it;
                    # no compute stream wait is added for its producer.
                    self._stage_prediction_rows(1)
            plan_cache_to_device_buffer_mla(
                top_k_tokens=top_k_result,
                device_buffer_tokens=self.req_device_buffer_tokens[layer_id],
                host_cache_locs=self.req_to_host_pool,
                device_buffer_locs=self.req_device_buffer_token_locs[layer_id],
                top_k_device_locs=table,
                req_pool_indices=req_pool_indices,
                seq_lens=compressed_seq_lens,
                lru_slots=self.lru_slots[layer_id],
                miss_src=self._miss_src[:num_reqs],
                miss_dst=self._miss_dst[:num_reqs],
                miss_count=self._miss_count[:num_reqs],
                num_top_k=self.top_k,
                hot_buffer_size=self.device_buffer_size,
                block_size=self.swap_in_block_size,
                num_real_reqs=self.num_real_reqs,
            )
            plan_ready_event = None
            if getattr(self, "_overlap_enabled", False):
                plan_ready_event = device_module.Event(enable_timing=False)
                plan_ready_event.record(device_module.current_stream())
            pending = _HiSparsePendingPlan(
                request=active,
                request_indices_ptr=request_indices_ptr,
                step=self._split_step,
                anchor_layer=layer_id,
                group_layers=group,
                representation="mla-bf16-width-576",
                num_reqs=num_reqs,
                table=table,
                miss_src=self._miss_src[:num_reqs],
                miss_dst=self._miss_dst[:num_reqs],
                miss_count=self._miss_count[:num_reqs],
                staging=stage_plan,
            )
            self._split_pending = pending
            self._split_followers_seen = 0

            if self.prediction_staging_enabled:
                assert eligible_count is not None and skipped_count is not None
                self._materialize_staged_anchor(
                    layer_id,
                    stage_plan,
                    eligible_count,
                    skipped_count,
                    self._miss_src[:num_reqs],
                    self._miss_dst[:num_reqs],
                    self._miss_count[:num_reqs],
                    len(group[1:]),
                    plan_ready_event=plan_ready_event,
                )
                if (
                    getattr(self, "_overlap_enabled", False)
                    and getattr(self, "_urgent_done_event", None) is not None
                ):
                    device_module.current_stream().wait_event(self._urgent_done_event)
            else:
                self.wait_for_pending_backup()
                self._run_copy_only_kernel(num_reqs, layer_id)
            followers = group[1:]
            if followers:
                self.prefetch_stream.wait_stream(device_module.current_stream())
                with device_module.stream(self.prefetch_stream):
                    for follower in followers:
                        self._run_copy_only_kernel(num_reqs, follower)
                        self._prefetch_events[self._prefetch_slot[follower]].record(
                            self.prefetch_stream
                        )
        except BaseException:
            self._split_aborted_generation = active.generation
            raise

        self._split_anchor_index = (self._split_anchor_index + 1) % len(
            self._split_anchor_layers
        )
        return table

    def _consume_split_follower(
        self, req_pool_indices: torch.Tensor, layer_id: int
    ) -> torch.Tensor:
        pending = self._split_pending
        if pending is None:
            raise RuntimeError("HiSparse split follower has no pending anchor plan")
        self._validate_split_plan(pending)
        if req_pool_indices.size(0) != pending.num_reqs:
            raise RuntimeError("HiSparse split follower batch size changed")
        if req_pool_indices.data_ptr() != pending.request_indices_ptr:
            raise RuntimeError("HiSparse split follower request identity changed")
        next_index = self._split_followers_seen + 1
        if (
            next_index >= len(pending.group_layers)
            or pending.group_layers[next_index] != layer_id
        ):
            raise RuntimeError("HiSparse split follower order changed")
        self._prefetch_events[self._prefetch_slot[layer_id]].wait(
            device_module.current_stream()
        )
        self._split_followers_seen += 1
        return pending.table

    def swap_in_selected_pages(
        self,
        req_pool_indices: torch.Tensor,
        compressed_seq_lens: torch.Tensor,
        top_k_result: torch.Tensor,
        layer_id: int,
    ) -> torch.Tensor:
        """Swap selected top-k tokens into device memory and return their indices.

        With prefetch enabled, anchors swap in synchronously (recording the miss
        plan) and prefetch their skip layers' copies; skip layers just wait.
        """
        if self.split_materialization_enabled:
            if self._is_shared_index_layer[layer_id]:
                return self._consume_split_follower(req_pool_indices, layer_id)
            return self._run_split_anchor(
                req_pool_indices, compressed_seq_lens, top_k_result, layer_id
            )

        if not self.enable_prefetch:
            return self._run_swap_in_kernel(
                req_pool_indices, compressed_seq_lens, top_k_result, layer_id
            )

        num_reqs = req_pool_indices.size(0)
        if self._is_shared_index_layer[layer_id]:
            # Skip layer: wait for its prefetched copy; the anchor's slot table
            # applies (shared index + lockstep buffers).
            slot = self._prefetch_slot[layer_id]
            self._prefetch_events[slot].wait(device_module.current_stream())
            return self.top_k_device_locs_buffer[:num_reqs]

        # Anchor: swap in synchronously (recording the plan), then prefetch the
        # skip layers' copies on the side stream.
        group = self._prefetch_groups.get(layer_id)
        anchor_locs = self._run_swap_in_kernel(
            req_pool_indices,
            compressed_seq_lens,
            top_k_result,
            layer_id,
            record_plan=group is not None,
        )
        if group:
            # Fork: the prefetch stream must observe the anchor's plan (produced
            # on the current stream) before replaying it.
            self.prefetch_stream.wait_stream(device_module.current_stream())
            with device_module.stream(self.prefetch_stream):
                for skip_layer in group:
                    self._run_copy_only_kernel(num_reqs, skip_layer)
                    self._prefetch_events[self._prefetch_slot[skip_layer]].record(
                        self.prefetch_stream
                    )
        return anchor_locs
