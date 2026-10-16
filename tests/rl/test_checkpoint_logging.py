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
"""Only completed RL checkpoints are advertised to training loggers."""

from unittest.mock import MagicMock

import pytest

pytest.importorskip("datasets")
pytest.importorskip("grpc")
pytest.importorskip("tensorboard")

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorboard.util import tensor_util

from lerobot.common.tensorboard_utils import TensorBoardLogger
from lerobot.envs.configs import AlohaEnv
from lerobot.rl import learner
from lerobot.rl.train_rl import TrainRLServerPipelineConfig
from lerobot.utils.constants import PRETRAINED_MODEL_DIR, TRAINING_STATE_DIR


@pytest.mark.parametrize("failed_stage", [None, "model", "buffer"])
def test_rl_checkpoint_reference_after_save(tmp_path, monkeypatch, failed_stage):
    cfg = TrainRLServerPipelineConfig(output_dir=tmp_path, env=AlohaEnv())
    logger = TensorBoardLogger(cfg)
    log_policy = MagicMock(wraps=logger.log_policy)
    monkeypatch.setattr(logger, "log_policy", log_policy)
    model_saved = []

    def save_model(checkpoint_dir, **kwargs):
        if failed_stage == "model":
            raise OSError("model save failed")
        (checkpoint_dir / TRAINING_STATE_DIR).mkdir(parents=True)
        model_dir = checkpoint_dir / PRETRAINED_MODEL_DIR
        model_dir.mkdir()
        (model_dir / "model.safetensors").write_bytes(b"test weights")
        model_saved.append(checkpoint_dir)

    def save_buffer(**kwargs):
        assert model_saved
        log_policy.assert_not_called()
        if failed_stage == "buffer":
            raise OSError("buffer save failed")

    replay_buffer = MagicMock()
    replay_buffer.to_lerobot_dataset.side_effect = save_buffer
    monkeypatch.setattr(learner, "save_checkpoint", save_model)
    monkeypatch.setattr(learner, "update_last_checkpoint", lambda path: None)

    def save():
        learner.save_training_checkpoint(
            cfg=cfg,
            optimization_step=10,
            online_steps=100,
            interaction_message={"Interaction step": 30},
            policy=MagicMock(),
            optimizers={},
            replay_buffer=replay_buffer,
            metrics_logger=logger,
        )

    try:
        if failed_stage:
            with pytest.raises(OSError, match="save failed"):
                save()
            log_policy.assert_not_called()
        else:
            save()
            checkpoint = model_saved[0]
            log_policy.assert_called_once_with(checkpoint)
            assert (checkpoint / PRETRAINED_MODEL_DIR / "model.safetensors").read_bytes() == b"test weights"
    finally:
        logger.close()

    events = EventAccumulator(str(tmp_path / "tensorboard")).Reload()
    if not failed_stage:
        entry = events.Tensors(f"checkpoints/{checkpoint.name}/text_summary")[0]
        assert entry.step == 10
        assert tensor_util.make_ndarray(entry.tensor_proto).item().decode() == str(checkpoint.resolve())
    else:
        assert set(events.Tags()["tensors"]) == {"config/text_summary", "command/text_summary"}
    # The TensorBoard directory contains events, never a second copy of the weights.
    assert all("tfevents" in path.name for path in (tmp_path / "tensorboard").rglob("*") if path.is_file())
