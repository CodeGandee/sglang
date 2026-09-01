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

from types import SimpleNamespace
from unittest.mock import Mock

from sglang.srt.managers.scheduler import Scheduler


def test_scheduler_releases_optional_cache_provider_resources() -> None:
    provider_release = Mock()
    scheduler = SimpleNamespace(
        hisparse_coordinator=SimpleNamespace(destroy=Mock()),
        tree_cache=SimpleNamespace(release_host_resources=Mock()),
        decode_offload_manager=SimpleNamespace(release_host_resources=Mock()),
        token_to_kv_pool=SimpleNamespace(
            release_host_resources=provider_release,
        ),
    )

    Scheduler.release_host_resources(scheduler)

    scheduler.hisparse_coordinator.destroy.assert_called_once_with()
    scheduler.tree_cache.release_host_resources.assert_called_once_with()
    scheduler.decode_offload_manager.release_host_resources.assert_called_once_with()
    provider_release.assert_called_once_with()
