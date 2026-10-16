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
"""Local TensorBoard logging with the same logging methods as WandBLogger."""

import atexit
import json
import logging
import math
import shlex
import sys
import time
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING
from urllib.parse import quote

from lerobot.utils.constants import CHECKPOINTS_DIR, LAST_CHECKPOINT_LINK, TRAINING_STATE_DIR, TRAINING_STEP
from lerobot.utils.import_utils import _av_available, _tensorboard_available, require_package
from lerobot.utils.io_utils import load_json

if TYPE_CHECKING:
    from lerobot.configs.eval import EvalPipelineConfig
    from lerobot.configs.train import TrainPipelineConfig

if TYPE_CHECKING or _av_available:
    import av

if TYPE_CHECKING or _tensorboard_available:
    from tensorboard.compat.proto.event_pb2 import Event, SessionLog
    from tensorboard.compat.proto.summary_pb2 import Summary
    from torch.utils.tensorboard import SummaryWriter


class TensorBoardLogger:
    def __init__(self, cfg: "TrainPipelineConfig | EvalPipelineConfig"):
        require_package("tensorboard", "tensorboard")
        self.cfg = cfg.tensorboard
        self.log_dir = Path(self.cfg.log_dir) if self.cfg.log_dir else Path(cfg.output_dir) / "tensorboard"
        self.env_fps = cfg.env.fps if cfg.env else 30
        self._resume_steps = {}
        if getattr(cfg, "resume", False):
            checkpoint_dir = (
                getattr(cfg, "checkpoint_path", None)
                or Path(cfg.output_dir) / CHECKPOINTS_DIR / LAST_CHECKPOINT_LINK
            )
            state = load_json(checkpoint_dir / TRAINING_STATE_DIR / TRAINING_STEP)
            self._resume_steps = {
                None: int(state["step"]),
                "Optimization step": int(state["step"]),
                "Interaction step": int(state.get("interaction_step", 0)),
            }
        self._writers = {}
        self._writer = self._get_writer()
        try:
            self._writer.add_text(
                "config",
                "```json\n" + json.dumps(cfg.to_dict(), indent=2, default=str) + "\n```",
                global_step=self._resume_steps.get(None, 0),
            )
            command = shlex.join(sys.orig_argv)
            self._writer.add_text(
                "command",
                f"```bash\ncd {shlex.quote(str(Path.cwd()))}\n{command}\n```",
                global_step=self._resume_steps.get(None, 0),
            )
            self._writer.flush()
        except BaseException:
            self.close()
            raise
        atexit.register(self.close)
        logging.info("TensorBoard logs: %s; view with: tensorboard --logdir %s", self.log_dir, self.log_dir)

    def _get_writer(self, custom_step_key: str | None = None):
        if custom_step_key not in self._writers:
            # TensorBoard purges a whole run at one step. Independent RL counters
            # therefore need separate run directories, each with its own restart marker.
            log_dir = self.log_dir
            if custom_step_key is not None:
                log_dir = log_dir / "steps" / ("step_" + quote(custom_step_key, safe=""))
            resume_step = self._resume_steps.get(custom_step_key)
            if resume_step is not None:
                # Event filenames sort by whole seconds, then PID and an unpadded
                # counter. A restart in the same second can otherwise sort first.
                now = time.time()
                if any(log_dir.glob(f"events.out.tfevents.{int(now):010d}.*")):
                    time.sleep(int(now) + 1 - now)
            writer = SummaryWriter(
                log_dir=str(log_dir),
                flush_secs=self.cfg.flush_secs,
                purge_step=resume_step + 1 if resume_step is not None else None,
            )
            if resume_step is not None:
                # The tensor-plugin reader ignores the first START in older logs
                # without a session marker. Repeat the same boundary so both
                # scalar and tensor readers discard only steps after the checkpoint.
                writer.file_writer.add_event(
                    Event(step=resume_step + 1, session_log=SessionLog(status=SessionLog.START))
                )
            self._writers[custom_step_key] = writer
        return self._writers[custom_step_key]

    def log_dict(
        self, d: dict, step: int | None = None, mode: str = "train", custom_step_key: str | None = None
    ):
        if mode not in {"train", "eval"}:
            raise ValueError(mode)
        if step is None and custom_step_key is None:
            raise ValueError("Either step or custom_step_key must be provided.")
        if custom_step_key is not None:
            step = int(d[custom_step_key])
        writer = self._get_writer(custom_step_key)
        # RL interaction and optimization counters have independent timelines.
        prefix = f"{mode}/{custom_step_key}" if custom_step_key else mode
        for key, value in d.items():
            if custom_step_key is not None and key == custom_step_key:
                continue
            if isinstance(value, (int, float)):
                writer.add_scalar(f"{prefix}/{key}", value, global_step=step)
            elif isinstance(value, str):
                writer.add_text(f"{prefix}/{key}", value, global_step=step)
            else:
                logging.warning("TensorBoard ignored key %s with unsupported type %s", key, type(value))

    def log_policy(self, checkpoint_dir: Path, *, step: int | None = None):
        """Record a local checkpoint reference; model files already live on disk."""
        if not self.cfg.disable_artifact:
            if step is None:
                step = int(checkpoint_dir.name) if checkpoint_dir.name.isdecimal() else 0
            reference = str(checkpoint_dir.resolve()) if checkpoint_dir.exists() else str(checkpoint_dir)
            self._writer.add_text(f"checkpoints/{checkpoint_dir.name}", reference, global_step=step)
            self._writer.flush()

    def log_video(self, video_path: str, step: int, mode: str = "train", *, name: str = "video"):
        if mode not in {"train", "eval"}:
            raise ValueError(mode)
        if not self.cfg.log_videos:
            return
        require_package("av", "tensorboard")
        with TemporaryDirectory(prefix="lerobot-tensorboard-video-") as directory:
            gif_path = Path(directory) / "video.gif"
            size = self._write_video_gif(video_path, gif_path)
            if size is None:
                return
            width, height = size
            # Like SummaryWriter.add_video, publish an animated GIF in Images.
            # Only the compressed payload is loaded in full, never the decoded clip.
            summary = Summary(
                value=[
                    Summary.Value(
                        tag=f"{mode}/{name}",
                        image=Summary.Image(
                            width=width,
                            height=height,
                            colorspace=3,
                            encoded_image_string=gif_path.read_bytes(),
                        ),
                    )
                ]
            )
            self._writer.file_writer.add_summary(summary, global_step=step)
            self._writer.flush()

    def _write_video_gif(self, video_path: str, gif_path: Path) -> tuple[int, int] | None:
        """Resample the entire clip while keeping only current/previous frames in RAM."""
        with av.open(str(video_path)) as source:
            stream = source.streams.video[0]
            source_fps = float(stream.average_rate) if stream.average_rate else self.env_fps
            fps = min(source_fps, self.cfg.video_max_fps)
            duration = float(stream.duration * stream.time_base) if stream.duration is not None else None
            if duration is None:
                # Containers without duration metadata need a streaming first pass.
                start_time = None
                for index, frame in enumerate(source.decode(stream)):
                    timestamp = frame.time if frame.time is not None else index / source_fps
                    if start_time is None:
                        start_time = timestamp
                    frame_duration = (
                        float(frame.duration * frame.time_base) if frame.duration else 1 / source_fps
                    )
                    duration = timestamp - start_time + frame_duration
                source.seek(stream.start_time or 0, stream=stream)
            if duration is None or duration <= 0:
                return None

            frame_count = math.ceil(duration * fps - 1e-9)
            # GIF delays have 10 ms precision. Preserve the final partial interval,
            # rather than rounding playback up to a whole output frame.
            final_delay = max(1, round(duration * 100) - round((frame_count - 1) / fps * 100))
            scale = min(1.0, self.cfg.video_max_size / max(stream.width, stream.height))
            width, height = max(1, int(stream.width * scale)), max(1, int(stream.height * scale))
            with av.open(
                str(gif_path), "w", format="gif", options={"loop": "0", "final_delay": str(final_delay)}
            ) as output:
                target = output.add_stream("gif", rate=Fraction(str(fps)).limit_denominator(10000))
                target.width, target.height = width, height
                target.pix_fmt = "rgb8"

                def encode(frame, index):
                    resized = frame.reformat(width=width, height=height, format="rgb8")
                    resized.pts = index
                    resized.time_base = Fraction(1, 1) / Fraction(str(fps)).limit_denominator(10000)
                    for packet in target.encode(resized):
                        output.mux(packet)

                sample_index = 0
                previous = None
                start_time = None
                for index, frame in enumerate(source.decode(stream)):
                    timestamp = frame.time if frame.time is not None else index / source_fps
                    if start_time is None:
                        start_time = timestamp
                    elapsed = timestamp - start_time
                    if previous is not None:
                        while sample_index < frame_count and sample_index / fps < elapsed - 1e-9:
                            encode(previous, sample_index)
                            sample_index += 1
                    previous = frame
                if previous is None:
                    return None
                while sample_index < frame_count:
                    encode(previous, sample_index)
                    sample_index += 1
                for packet in target.encode():
                    output.mux(packet)
            return width, height

    def close(self):
        atexit.unregister(self.close)
        writers, self._writers = self._writers, {}
        for writer in writers.values():
            try:
                writer.close()
            except Exception:
                logging.exception("Failed to close a TensorBoard writer")
        self._writer = None
