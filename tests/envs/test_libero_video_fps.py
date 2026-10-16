#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
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
"""LIBERO's recorded video clock follows the configured control frequency."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

pytest.importorskip("libero")


def test_render_fps_is_per_instance_control_frequency(tmp_path, monkeypatch):
    monkeypatch.setenv("NUMBA_CACHE_DIR", str(tmp_path / "numba"))
    from lerobot.envs.libero import LiberoEnv

    task = SimpleNamespace(
        name="test_task", language="test task", problem_folder="test", bddl_file="test.bddl"
    )
    suite = MagicMock()
    suite.get_task.return_value = task
    first = LiberoEnv(suite, 0, "libero_10", control_freq=20, init_states=False)
    second = LiberoEnv(suite, 0, "libero_10", control_freq=30, init_states=False)
    assert first.metadata["render_fps"] == 20
    assert second.metadata["render_fps"] == 30
    assert LiberoEnv.metadata["render_fps"] == 20
    assert first._env is second._env is None
