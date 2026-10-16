#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
import json
import shlex
import sys
from io import BytesIO
from unittest.mock import MagicMock

import pytest

pytest.importorskip("tensorboard")

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorboard.backend.event_processing.plugin_event_accumulator import (
    EventAccumulator as PluginEventAccumulator,
)
from tensorboard.util import tensor_util

from lerobot.common import logging_utils
from lerobot.common.tensorboard_utils import TensorBoardLogger
from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.utils.constants import TRAINING_STATE_DIR, TRAINING_STEP


def make_config(tmp_path):
    cfg = TrainPipelineConfig(dataset=DatasetConfig(repo_id="test/data"), output_dir=tmp_path)
    cfg.tensorboard.enable = True
    return cfg


def set_resume_checkpoint(cfg, step, interaction_step=0):
    cfg.resume = True
    cfg.checkpoint_path = cfg.output_dir / "checkpoints" / f"{step:06d}"
    state_dir = cfg.checkpoint_path / TRAINING_STATE_DIR
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / TRAINING_STEP).write_text(json.dumps({"step": step, "interaction_step": interaction_step}))


def test_events_and_resume(tmp_path):
    cfg = make_config(tmp_path)
    logger = TensorBoardLogger(cfg)
    logger.log_dict({"loss": 1.5, "note": "hello", "nested": {}}, step=3)
    logger.log_dict({"Optimization step": 7, "loss": 0.5}, custom_step_key="Optimization step")
    checkpoint = tmp_path / "checkpoints" / "000003"
    logger.log_policy(checkpoint)
    logger.close()
    logger.close()
    set_resume_checkpoint(cfg, 3)
    logger = TensorBoardLogger(cfg)
    logger.log_dict({"loss": 1.0}, step=4)
    logger.close()

    events = EventAccumulator(str(tmp_path / "tensorboard")).Reload()
    assert [(e.step, e.value) for e in events.Scalars("train/loss")] == [(3, 1.5), (4, 1.0)]
    optimization_events = EventAccumulator(
        str(tmp_path / "tensorboard/steps/step_Optimization%20step")
    ).Reload()
    assert optimization_events.Scalars("train/Optimization step/loss")[0].step == 7
    assert "config/text_summary" in events.Tags()["tensors"]
    assert "train/note/text_summary" in events.Tags()["tensors"]
    assert "checkpoints/000003/text_summary" in events.Tags()["tensors"]


def test_dispatch_and_disabled_backends(tmp_path, monkeypatch):
    cfg = make_config(tmp_path)
    cfg.tensorboard.enable = False
    assert logging_utils.make_training_logger(cfg) is None
    wandb = MagicMock()
    tb = MagicMock()
    monkeypatch.setattr(logging_utils, "WandBLogger", lambda cfg: wandb)
    monkeypatch.setattr(logging_utils, "TensorBoardLogger", lambda cfg: tb)
    cfg.wandb.enable = cfg.tensorboard.enable = True
    logger = logging_utils.make_training_logger(cfg)
    logger.log_dict({"loss": 1}, step=2)
    logger.log_video("video.mp4", 2)
    logger.log_policy(tmp_path)
    logger.close()
    for backend in (wandb, tb):
        backend.log_dict.assert_called_once_with({"loss": 1}, step=2, mode="train", custom_step_key=None)
        backend.log_video.assert_called_once_with("video.mp4", 2, mode="train")
        backend.log_policy.assert_called_once_with(tmp_path)
        backend.close.assert_called_once()


def test_validation_and_custom_directory(tmp_path):
    cfg = make_config(tmp_path)
    cfg.tensorboard.log_dir = str(tmp_path / "custom")
    cfg.tensorboard.disable_artifact = True
    logger = TensorBoardLogger(cfg)
    with pytest.raises(ValueError):
        logger.log_dict({}, mode="other", step=1)
    with pytest.raises(ValueError):
        logger.log_dict({})
    with pytest.raises(ValueError):
        logger.log_video("missing.mp4", 1, mode="other")
    logger.log_policy(tmp_path)
    logger.close()
    events = EventAccumulator(cfg.tensorboard.log_dir).Reload()
    assert set(events.Tags()["tensors"]) == {"config/text_summary", "command/text_summary"}


def test_video(tmp_path):
    av = pytest.importorskip("av")
    import numpy as np

    path = tmp_path / "eval.mp4"
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=10)
        stream.width = stream.height = 16
        stream.pix_fmt = "yuv420p"
        for i in range(3):
            frame = av.VideoFrame.from_ndarray(np.full((16, 16, 3), i * 80, dtype=np.uint8), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    logger = TensorBoardLogger(make_config(tmp_path))
    logger.log_video(str(path), 5, mode="eval")
    logger.close()
    events = EventAccumulator(str(tmp_path / "tensorboard")).Reload()
    video = events.Images("eval/video")[0]
    assert video.step == 5
    assert video.encoded_image_string.startswith(b"GIF")


def test_rollback_purges_stale_steps(tmp_path):
    cfg = make_config(tmp_path)
    logger = TensorBoardLogger(cfg)
    for step in (3, 4, 5):
        logger.log_dict({"loss": 1.5}, step=step)
        logger.log_dict({"reward": 0.0}, step=step, mode="eval")
    logger.log_policy(tmp_path / "000005")
    logger.close()

    set_resume_checkpoint(cfg, 3)
    logger = TensorBoardLogger(cfg)
    logger.log_dict({"loss": 0.5}, step=4)
    logger.close()
    events = EventAccumulator(str(tmp_path / "tensorboard")).Reload()
    assert [(e.step, e.value) for e in events.Scalars("train/loss")] == [(3, 1.5), (4, 0.5)]
    assert [e.step for e in events.Scalars("eval/reward")] == [3]
    tensor_events = PluginEventAccumulator(str(tmp_path / "tensorboard")).Reload()
    assert not tensor_events.Tensors("checkpoints/000005/text_summary")


def test_rl_rollback_uses_independent_counters(tmp_path):
    cfg = make_config(tmp_path)
    logger = TensorBoardLogger(cfg)
    for opt_step, interaction_step in ((2, 20), (3, 30), (4, 40)):
        logger.log_dict({"Optimization step": opt_step, "loss": 1.0}, custom_step_key="Optimization step")
        logger.log_dict(
            {"Interaction step": interaction_step, "reward": 0.0}, custom_step_key="Interaction step"
        )
    logger.close()
    set_resume_checkpoint(cfg, 2, interaction_step=20)
    logger = TensorBoardLogger(cfg)
    logger.log_dict({"Optimization step": 3, "loss": 0.5}, custom_step_key="Optimization step")
    logger.log_dict({"Interaction step": 21, "reward": 1.0}, custom_step_key="Interaction step")
    logger.close()
    opt_events = EventAccumulator(str(tmp_path / "tensorboard/steps/step_Optimization%20step")).Reload()
    interaction_events = EventAccumulator(
        str(tmp_path / "tensorboard/steps/step_Interaction%20step")
    ).Reload()
    assert [(e.step, e.value) for e in opt_events.Scalars("train/Optimization step/loss")] == [
        (2, 1.0),
        (3, 0.5),
    ]
    assert [(e.step, e.value) for e in interaction_events.Scalars("train/Interaction step/reward")] == [
        (20, 0.0),
        (21, 1.0),
    ]


def test_video_limits_preserve_duration(tmp_path):
    av = pytest.importorskip("av")
    image = pytest.importorskip("PIL.Image")
    import numpy as np

    path = tmp_path / "long.mp4"
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=30)
        stream.width, stream.height = 64, 48
        stream.pix_fmt = "yuv420p"
        for i in range(60):
            frame = av.VideoFrame.from_ndarray(np.full((48, 64, 3), i * 4, dtype=np.uint8), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    cfg = make_config(tmp_path)
    cfg.tensorboard.video_max_size = 16
    cfg.tensorboard.video_max_fps = 5
    # A legacy config must no longer truncate the clip either.
    cfg.tensorboard.video_max_duration_s = 0.6
    logger = TensorBoardLogger(cfg)
    logger.log_video(str(path), 5)
    logger.close()
    events = EventAccumulator(str(tmp_path / "tensorboard")).Reload()
    entry = events.Images("train/video")[0]
    assert entry.step == 5
    with image.open(BytesIO(entry.encoded_image_string)) as gif:
        assert gif.size == (16, 12)
        assert gif.n_frames == 10  # 2 seconds at 5 FPS.
        duration_ms = 0
        for index in range(gif.n_frames):
            gif.seek(index)
            duration_ms += gif.info["duration"]
        assert duration_ms == 2000
        # The clip includes its end, beyond the old 0.6-second cap.
        assert np.asarray(gif.convert("RGB")).mean() > 150


@pytest.mark.parametrize("frame_count,source_fps,target_fps", [(360, 30, 10), (35, 30, 7), (3, 10, 10)])
def test_complete_video_duration_and_partial_final_frame(tmp_path, frame_count, source_fps, target_fps):
    av = pytest.importorskip("av")
    image = pytest.importorskip("PIL.Image")
    import numpy as np

    path = tmp_path / "full.mp4"
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=source_fps)
        stream.width = stream.height = 16
        stream.pix_fmt = "yuv420p"
        for index in range(frame_count):
            frame = av.VideoFrame.from_ndarray(
                np.full((16, 16, 3), (index * 17) % 256, dtype=np.uint8), format="rgb24"
            )
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    cfg = make_config(tmp_path)
    cfg.tensorboard.video_max_fps = target_fps
    logger = TensorBoardLogger(cfg)
    logger.log_video(str(path), 1, mode="eval")
    logger.close()
    events = EventAccumulator(str(tmp_path / "tensorboard")).Reload()
    with image.open(BytesIO(events.Images("eval/video")[0].encoded_image_string)) as gif:
        duration_ms = 0
        for index in range(gif.n_frames):
            gif.seek(index)
            duration_ms += gif.info["duration"]
    # The GIF format represents time with a precision of 10 ms.
    assert abs(duration_ms - frame_count / source_fps * 1000) <= 10


def test_closing_backend_failure_still_flushes_tensorboard(tmp_path):
    logger = TensorBoardLogger(make_config(tmp_path))
    logger.log_dict({"loss": 0.25}, step=2)
    failing_backend = MagicMock()
    failing_backend.close.side_effect = RuntimeError("finish failed")
    logging_utils.TrainingLogger([failing_backend, logger]).close()
    events = EventAccumulator(str(tmp_path / "tensorboard")).Reload()
    assert events.Scalars("train/loss")[0].value == 0.25


def test_factory_closes_initialized_backend_on_failure(tmp_path, monkeypatch):
    cfg = make_config(tmp_path)
    cfg.wandb.enable = True
    backend = MagicMock()
    monkeypatch.setattr(logging_utils, "WandBLogger", lambda cfg: backend)
    monkeypatch.setattr(
        logging_utils, "TensorBoardLogger", MagicMock(side_effect=RuntimeError("init failed"))
    )
    with pytest.raises(RuntimeError, match="init failed"):
        logging_utils.make_training_logger(cfg)
    backend.close.assert_called_once()


@pytest.mark.parametrize(
    "setting,value",
    [
        ("video_max_fps", 0),
        ("video_max_size", 0),
        ("flush_secs", 0),
    ],
)
def test_invalid_tensorboard_limits(setting, value):
    from lerobot.configs.default import TensorBoardConfig

    with pytest.raises(ValueError):
        TensorBoardConfig(**{setting: value})


@pytest.mark.parametrize(
    "entrypoint",
    [
        ["python", "/conda/bin/lerobot-train"],
        ["python", "-m", "lerobot.scripts.lerobot_eval"],
        ["python", "-m", "lerobot.rl.learner"],
    ],
)
def test_command_records_arguments_and_working_directory(tmp_path, monkeypatch, entrypoint):
    working_dir = tmp_path / "working directory"
    working_dir.mkdir()
    monkeypatch.chdir(working_dir)
    argv = [*entrypoint, "--policy.path=/models/model with spaces", "--notes=quote's $HOME $(literal)"]
    monkeypatch.setattr(sys, "orig_argv", argv)
    logger = TensorBoardLogger(make_config(tmp_path))
    logger.close()
    events = EventAccumulator(str(tmp_path / "tensorboard")).Reload()
    entry = events.Tensors("command/text_summary")[0]
    text = tensor_util.make_ndarray(entry.tensor_proto).item().decode()
    lines = text.splitlines()
    assert lines[0] == "```bash" and lines[-1] == "```"
    assert shlex.split(lines[1]) == ["cd", str(working_dir)]
    assert shlex.split(lines[2]) == argv
    assert entry.step == 0
