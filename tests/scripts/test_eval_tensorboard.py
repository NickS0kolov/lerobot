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
"""Standalone eval publishes progress, per-task timing and videos in one run."""

import threading
from unittest.mock import MagicMock

import draccus
import pytest
import torch

pytest.importorskip("datasets")
pytest.importorskip("tensorboard")

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from lerobot.common.tensorboard_utils import TensorBoardLogger
from lerobot.configs.default import EvalConfig
from lerobot.configs.eval import EvalPipelineConfig
from lerobot.envs.configs import AlohaEnv
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.scripts import lerobot_eval


def metrics(successes, seconds, video_paths=()):
    return {
        "successes": successes,
        "sum_rewards": [float(success) for success in successes],
        "max_rewards": [float(success) for success in successes],
        "eval_s": seconds,
        "video_paths": list(video_paths),
        "predicted_video_paths": [],
    }


def make_video(path):
    av = pytest.importorskip("av")
    import numpy as np

    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=20)
        stream.width = stream.height = 16
        stream.pix_fmt = "yuv420p"
        for i in range(10):
            frame = av.VideoFrame.from_ndarray(np.full((16, 16, 3), i * 20, dtype=np.uint8), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def test_eval_cli_tensorboard_flags(tmp_path, monkeypatch):
    monkeypatch.setattr("sys.argv", ["lerobot-eval"])
    cfg = draccus.parse(
        EvalPipelineConfig,
        args=[
            "--env.type=aloha",
            "--policy.type=act",
            "--policy.device=cpu",
            "--tensorboard.enable=true",
            "--tensorboard.log_videos=false",
            f"--tensorboard.log_dir={tmp_path}",
            "--tensorboard.video_max_fps=5",
            "--tensorboard.video_max_size=128",
            "--tensorboard.flush_secs=3",
            "--tensorboard.disable_artifact=true",
        ],
    )
    assert cfg.tensorboard.enable and not cfg.tensorboard.log_videos
    assert cfg.tensorboard.log_dir == str(tmp_path)
    encoded = cfg.to_dict()
    assert encoded["tensorboard"]["video_max_fps"] == 5
    assert encoded["tensorboard"]["video_max_size"] == 128
    assert encoded["tensorboard"]["disable_artifact"] is True
    assert encoded["env"]["type"] == "aloha"


@pytest.mark.parametrize("log_videos", [True, False])
def test_progress_weighted_sr_time_and_task_videos(tmp_path, log_videos):
    cfg = EvalPipelineConfig(env=AlohaEnv(), output_dir=tmp_path)
    cfg.tensorboard.log_videos = log_videos
    video = tmp_path / "episode.mp4"
    make_video(video)
    logger = TensorBoardLogger(cfg)
    progress = lerobot_eval._EvalProgressLogger(logger)
    try:
        progress("suite_a", 0, metrics([True], 2.5, [str(video)]))
        progress("suite_b", 1, metrics([False, False, True], 7.5, [str(video), str(video)]))
    finally:
        logger.close()
    events = EventAccumulator(str(tmp_path / "tensorboard")).Reload()
    assert [(e.step, e.value) for e in events.Scalars("eval/overall/pc_success")] == [(1, 100), (4, 50)]
    assert events.Scalars("eval/suite_a/task_00/eval_s")[0].value == 2.5
    assert events.Scalars("eval/suite_b/task_01/eval_s")[0].value == 7.5
    if log_videos:
        assert set(events.Tags()["images"]) == {
            "eval/suite_a/task_00/video",
            "eval/suite_b/task_01/video",
            "eval/suite_b/task_01/video_01",
        }
    else:
        assert events.Tags()["images"] == []
    # All tasks share this one run directory.
    assert not list((tmp_path / "tensorboard").iterdir())[0].is_dir()


@pytest.mark.parametrize("max_parallel_tasks", [1, 2])
def test_task_callback_runs_in_caller_thread(tmp_path, monkeypatch, max_parallel_tasks):
    envs = {"suite": {i: MagicMock() for i in range(2)}}
    policy = torch.nn.Linear(1, 1)
    calls = []
    parent = threading.get_ident()

    def run_one(group, task_id, env, **kwargs):
        return group, task_id, metrics([bool(task_id)], 1.5)

    def on_task(group, task_id, task_metrics):
        assert threading.get_ident() == parent
        calls.append((group, task_id))

    monkeypatch.setattr(lerobot_eval, "run_one", run_one)
    info = lerobot_eval.eval_policy_all(
        envs,
        policy,
        None,
        None,
        None,
        None,
        n_episodes=1,
        max_parallel_tasks=max_parallel_tasks,
        task_callback=on_task,
    )
    assert set(calls) == {("suite", 0), ("suite", 1)}
    assert info["overall"]["pc_success"] == 50
    assert policy.training
    for env in envs["suite"].values():
        env.close.assert_called_once()


@pytest.mark.parametrize(
    "enabled,log_videos,disable_artifact",
    [
        (False, True, False),
        (True, True, False),
        (True, False, True),
    ],
)
def test_eval_main_logging_and_cleanup(tmp_path, monkeypatch, enabled, log_videos, disable_artifact):
    cfg = EvalPipelineConfig(
        env=AlohaEnv(),
        eval=EvalConfig(n_episodes=1, batch_size=1),
        policy=ACTConfig(device="cpu"),
        output_dir=tmp_path,
    )
    cfg.policy.pretrained_path = tmp_path / "checkpoint" / "pretrained_model"
    cfg.tensorboard.enable = enabled
    cfg.tensorboard.log_videos = log_videos
    cfg.tensorboard.disable_artifact = disable_artifact
    policy = torch.nn.Linear(1, 1)
    policy.config = cfg.policy
    envs = {"aloha": {0: MagicMock()}}
    close_envs = MagicMock()
    monkeypatch.setattr(lerobot_eval, "make_env", lambda *args, **kwargs: envs)
    monkeypatch.setattr(lerobot_eval, "close_envs", close_envs)
    monkeypatch.setattr(lerobot_eval, "make_policy", lambda **kwargs: policy)
    monkeypatch.setattr(lerobot_eval, "make_pre_post_processors", lambda **kwargs: (None, None))
    monkeypatch.setattr(lerobot_eval, "make_env_pre_post_processors", lambda **kwargs: (None, None))

    def evaluate(**kwargs):
        callback = kwargs["task_callback"]
        if enabled:
            assert callback is not None
            callback("aloha", 0, metrics([True], 1.5))
        else:
            assert callback is None
        return {
            "per_task": [],
            "per_group": {},
            "overall": {
                "pc_success": 100,
                "n_episodes": 1,
                "avg_sum_reward": 1,
                "avg_max_reward": 1,
                "eval_s": 2.5,
                "eval_ep_s": 2.5,
            },
        }

    monkeypatch.setattr(lerobot_eval, "eval_policy_all", evaluate)
    lerobot_eval.eval_main(cfg)
    close_envs.assert_called_once_with(envs)
    assert (tmp_path / "eval_info.json").is_file()
    if enabled:
        events = EventAccumulator(str(tmp_path / "tensorboard")).Reload()
        assert len(events.Scalars("eval/overall/pc_success")) == 1
        assert events.Scalars("eval/overall/eval_s")[0].value == 2.5
        checkpoint_tags = [tag for tag in events.Tags()["tensors"] if tag.startswith("checkpoints/")]
        assert bool(checkpoint_tags) is not disable_artifact
    else:
        assert not (tmp_path / "tensorboard").exists()


def test_eval_failure_closes_env_and_flushes_logger(tmp_path, monkeypatch):
    cfg = EvalPipelineConfig(env=AlohaEnv(), policy=ACTConfig(device="cpu"), output_dir=tmp_path)
    cfg.tensorboard.enable = True
    logger = TensorBoardLogger(cfg)
    monkeypatch.setattr(lerobot_eval, "TensorBoardLogger", lambda cfg: logger)
    envs = {"aloha": {0: MagicMock()}}
    monkeypatch.setattr(lerobot_eval, "make_env", lambda *args, **kwargs: envs)
    close_envs = MagicMock()
    monkeypatch.setattr(lerobot_eval, "close_envs", close_envs)

    def fail_policy(**kwargs):
        logger.log_dict({"partial": 1.0}, step=1, mode="eval")
        raise RuntimeError("policy failed")

    monkeypatch.setattr(lerobot_eval, "make_policy", fail_policy)
    with pytest.raises(RuntimeError, match="policy failed"):
        lerobot_eval.eval_main(cfg)
    close_envs.assert_called_once_with(envs)
    assert logger._writer is None
    events = EventAccumulator(str(tmp_path / "tensorboard")).Reload()
    assert events.Scalars("eval/partial")[0].value == 1.0
