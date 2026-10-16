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
"""Dispatch training logs to all enabled backends."""

import logging
from pathlib import Path

from lerobot.configs.train import TrainPipelineConfig

from .tensorboard_utils import TensorBoardLogger
from .wandb_utils import WandBLogger


class TrainingLogger:
    def __init__(self, loggers: list[WandBLogger | TensorBoardLogger]):
        self.loggers = loggers

    def log_dict(
        self, d: dict, step: int | None = None, mode: str = "train", custom_step_key: str | None = None
    ):
        for logger in self.loggers:
            logger.log_dict(d, step=step, mode=mode, custom_step_key=custom_step_key)

    def log_policy(self, checkpoint_dir: Path):
        for logger in self.loggers:
            logger.log_policy(checkpoint_dir)

    def log_video(self, video_path: str, step: int, mode: str = "train"):
        for logger in self.loggers:
            logger.log_video(video_path, step, mode=mode)

    def close(self):
        for logger in self.loggers:
            try:
                logger.close()
            except Exception:
                # Still flush the other backend, preserving any training exception.
                logging.exception("Failed to close %s", type(logger).__name__)


def make_training_logger(cfg: TrainPipelineConfig) -> TrainingLogger | None:
    loggers = []
    try:
        if cfg.wandb.enable and cfg.wandb.project:
            loggers.append(WandBLogger(cfg))
        if cfg.tensorboard.enable:
            loggers.append(TensorBoardLogger(cfg))
    except BaseException:
        TrainingLogger(loggers).close()
        raise
    return TrainingLogger(loggers) if loggers else None
