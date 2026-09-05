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
"""Parser for opaque, backend-owned attention configuration."""

from __future__ import annotations

import argparse
import dataclasses
import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any


class _DuplicateKeyError(ValueError):
    pass


BackendConfigValidator = Callable[[object], None]


@dataclasses.dataclass(frozen=True, slots=True)
class AttentionBackendConfigOwner:
    """Validation contract for one backend's opaque configuration.

    ``inactive_signature_keys`` must contain only keys that identify this
    backend with high confidence. They let startup reject configuration for
    an inactive backend without interpreting unrelated plugins' opaque
    configuration.
    """

    backend_name: str
    validator: BackendConfigValidator
    inactive_signature_keys: frozenset[str]


ATTENTION_BACKEND_CONFIG_OWNERS: dict[str, AttentionBackendConfigOwner] = {}


def register_attention_backend_config_owner(
    backend_name: str,
    *,
    inactive_signature_keys: Iterable[str] = (),
):
    """Register configuration validation for one attention backend.

    Registration is idempotent for the same callable and key set. Plugins run
    this during process startup, before ``ServerArgs`` resolves its immutable
    attention route.
    """

    normalized = backend_name.strip()
    if not normalized:
        raise ValueError("attention backend config owner name must not be empty")
    signature_keys = frozenset(inactive_signature_keys)
    if any(not isinstance(key, str) or not key for key in signature_keys):
        raise ValueError(
            "attention backend config inactive signature keys must be non-empty strings"
        )

    def decorator(validator: BackendConfigValidator) -> BackendConfigValidator:
        registration = AttentionBackendConfigOwner(
            backend_name=normalized,
            validator=validator,
            inactive_signature_keys=signature_keys,
        )
        existing = ATTENTION_BACKEND_CONFIG_OWNERS.get(normalized)
        if existing is not None and existing != registration:
            raise ValueError(
                f"attention backend config owner {normalized!r} is already registered"
            )
        ATTENTION_BACKEND_CONFIG_OWNERS[normalized] = registration
        return validator

    return decorator


def validate_attention_backend_config(
    value: object,
    backend_names: Sequence[str | None],
) -> None:
    """Validate opaque configuration against the resolved attention route.

    A selected registered owner validates the complete object. With no
    registered owner selected, configuration stays opaque unless a
    high-specificity key identifies one inactive owner. This preserves
    out-of-tree configuration whose owner has not adopted this registry.
    """

    selected_names = tuple(
        dict.fromkeys(name for name in backend_names if isinstance(name, str) and name)
    )
    selected_owners = [
        ATTENTION_BACKEND_CONFIG_OWNERS[name]
        for name in selected_names
        if name in ATTENTION_BACKEND_CONFIG_OWNERS
    ]
    if len(selected_owners) > 1:
        names = ", ".join(owner.backend_name for owner in selected_owners)
        raise ValueError(
            "split attention backends selected incompatible config owners: " + names
        )
    if selected_owners:
        selected_owners[0].validator(value)
        return

    if not isinstance(value, Mapping):
        return
    keys = set(value)
    inactive_owners = [
        owner
        for owner in ATTENTION_BACKEND_CONFIG_OWNERS.values()
        if keys & owner.inactive_signature_keys
    ]
    if not inactive_owners:
        return
    if len(inactive_owners) > 1:
        names = ", ".join(owner.backend_name for owner in inactive_owners)
        raise ValueError(
            "attention backend config is claimed by multiple inactive backends: "
            + names
        )
    owner = inactive_owners[0]
    claimed = ", ".join(sorted(keys & owner.inactive_signature_keys))
    route = ", ".join(selected_names) if selected_names else "automatic native route"
    raise ValueError(
        f"attention backend config keys [{claimed}] require backend "
        f"{owner.backend_name!r}, but the resolved route is {route}"
    )


def _object_without_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKeyError(f"duplicate key {key!r}")
        result[key] = value
    return result


def parse_attention_backend_config(value: str) -> dict[str, Any]:
    """Parse an inline JSON object or an ``@path`` JSON object.

    The parser validates JSON structure only. Backend-specific keys remain
    opaque so an out-of-tree backend can validate them after plugin loading.
    Files are read by the launcher, and the resulting plain dictionary travels
    with ``ServerArgs`` to worker processes.
    """

    source = "inline JSON"
    payload = value
    if value.startswith("@"):
        path_text = value[1:]
        if not path_text:
            raise argparse.ArgumentTypeError(
                "attention backend config @path must name a JSON file"
            )
        path = Path(path_text).expanduser()
        source = str(path)
        try:
            payload = path.read_text(encoding="utf-8")
        except OSError as error:
            raise argparse.ArgumentTypeError(
                f"cannot read attention backend config {path}: {error}"
            ) from error
    try:
        parsed = json.loads(payload, object_pairs_hook=_object_without_duplicate_keys)
    except (json.JSONDecodeError, _DuplicateKeyError) as error:
        raise argparse.ArgumentTypeError(
            f"attention backend config from {source} is invalid JSON: {error}"
        ) from error
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError(
            f"attention backend config from {source} must be a JSON object"
        )
    return parsed
