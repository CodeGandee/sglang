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
import json
from pathlib import Path
from typing import Any


class _DuplicateKeyError(ValueError):
    pass


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
