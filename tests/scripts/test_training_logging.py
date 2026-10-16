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
"""Training failures must close and flush the actual TensorBoard writer."""

from types import SimpleNamespace

import pytest

pytest.importorskip("datasets")
pytest.importorskip("tensorboard")

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from lerobot.common.logging_utils import TrainingLogger
from lerobot.common.tensorboard_utils import TensorBoardLogger
from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.scripts import lerobot_train


@pytest.mark.parametrize("error", [RuntimeError("dataset failed"), KeyboardInterrupt()])
def test_training_error_flushes_logger(tmp_path, monkeypatch, error):
    cfg = TrainPipelineConfig(dataset=DatasetConfig(repo_id="test/data"), output_dir=tmp_path, seed=None)
    logger = TensorBoardLogger(cfg)
    monkeypatch.setattr(cfg, "validate", lambda: None)
    monkeypatch.setattr(
        lerobot_train,
        "make_accelerator",
        lambda cfg: SimpleNamespace(num_processes=1, device=SimpleNamespace(type="cpu")),
    )
    monkeypatch.setattr(lerobot_train.ParallelDims, "from_config", lambda *args: None)
    monkeypatch.setattr(lerobot_train, "init_logging", lambda **kwargs: None)
    monkeypatch.setattr(lerobot_train, "is_main_process", lambda: True)
    monkeypatch.setattr(lerobot_train, "make_training_logger", lambda cfg: TrainingLogger([logger]))

    def fail_dataset(cfg):
        logger.log_dict({"loss": 0.25}, step=1)
        raise error

    monkeypatch.setattr(lerobot_train, "make_train_eval_datasets", fail_dataset)
    with pytest.raises(type(error)) as caught:
        lerobot_train.train(cfg)
    assert caught.value is error
    assert logger._writer is None
    events = EventAccumulator(str(tmp_path / "tensorboard")).Reload()
    assert events.Scalars("train/loss")[0].value == 0.25
