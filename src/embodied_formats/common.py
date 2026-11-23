"""Core read-only abstractions and data models shared across all 5 dataset formats."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class EpisodeMeta:
    """Unified metadata for a single robot trajectory / episode."""

    episode_id: int
    num_steps: int
    fps: float
    language_instruction: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class StepSample:
    """A single time-step sample returned by read_step()."""

    episode_id: int
    step_index: int
    timestamp_s: float
    images: dict[str, np.ndarray]  # camera_name -> HxWx3 uint8 RGB
    state: np.ndarray | None
    action: np.ndarray | None
    language_instruction: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class WindowSample:
    """A temporal window sample (obs_horizon history + action_horizon future chunk)."""

    episode_id: int
    step_index: int
    timestamps_s: np.ndarray  # shape [obs_horizon]
    images: dict[str, np.ndarray]  # camera_name -> [obs_horizon, H, W, 3] uint8 RGB
    states: np.ndarray | None  # shape [obs_horizon, state_dim]
    action_chunk: np.ndarray | None  # shape [action_horizon, action_dim]
    language_instruction: str = ""


@dataclass
class FieldSchema:
    """Description of one field/channel/dataset inside a format."""

    name: str
    dtype: str
    shape: str
    encoding: str = ""
    details: str = ""


@dataclass
class InspectReport:
    """Low-level structural anatomy report for a dataset."""

    format_name: str
    path: str
    disk_bytes: int
    file_count: int
    num_episodes: int
    total_steps: int
    avg_fps: float
    schema: list[FieldSchema] = field(default_factory=list)
    storage_details: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


class BaseFormatReader(ABC):
    """Unified read-only interface for inspecting and sampling a dataset format.

    Each concrete subclass deliberately uses low-level libraries (mcap, h5py, zarr,
    pyarrow + PyAV, pure-Python TFRecord/Protobuf) rather than high-level training
    wrappers so the underlying storage mechanics are completely transparent.
    """

    format_name: str = "base"

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {self.path}")

    @abstractmethod
    def inspect(self) -> InspectReport:
        """Inspect low-level disk layout, schema, compression, and indexing."""

    @abstractmethod
    def list_episodes(self) -> list[EpisodeMeta]:
        """Return metadata for all episodes in this dataset."""

    @abstractmethod
    def read_step(self, episode_id: int, step_index: int) -> StepSample:
        """Read a single (episode_id, step_index) observation + action."""

    def read_window(
        self,
        episode_id: int,
        step_index: int,
        obs_horizon: int = 2,
        action_horizon: int = 16,
    ) -> WindowSample:
        """Read a temporal window simulating VLA / Diffusion Policy / ACT chunking.

        Default implementation composes read_step() with boundary clamping; formats
        with native range slicing (HDF5, Zarr, Parquet) override this for efficiency.
        """
        episodes = self.list_episodes()
        ep = episodes[episode_id]
        n = ep.num_steps
        if n <= 0:
            raise ValueError(f"Episode {episode_id} has no steps")

        obs_indices = [max(0, min(n - 1, step_index - (obs_horizon - 1 - i))) for i in range(obs_horizon)]
        act_indices = [max(0, min(n - 1, step_index + i)) for i in range(action_horizon)]

        obs_steps = [self.read_step(episode_id, idx) for idx in obs_indices]
        timestamps = np.array([s.timestamp_s for s in obs_steps], dtype=np.float64)

        images: dict[str, np.ndarray] = {}
        for cam in obs_steps[0].images:
            images[cam] = np.stack([s.images[cam] for s in obs_steps], axis=0)

        states = (
            np.stack([s.state for s in obs_steps], axis=0)
            if obs_steps[0].state is not None
            else None
        )

        # Avoid re-reading step_index if already in obs_steps
        step_cache = {idx: s for idx, s in zip(obs_indices, obs_steps)}
        actions = []
        for idx in act_indices:
            s = step_cache.get(idx) or self.read_step(episode_id, idx)
            if s.action is not None:
                actions.append(s.action)
        action_chunk = np.stack(actions, axis=0) if actions else None

        return WindowSample(
            episode_id=episode_id,
            step_index=step_index,
            timestamps_s=timestamps,
            images=images,
            states=states,
            action_chunk=action_chunk,
            language_instruction=obs_steps[-1].language_instruction,
        )

    def close(self) -> None:
        """Release any open file handles or video decoders."""

    def __enter__(self) -> BaseFormatReader:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()


def dir_disk_usage(path: Path) -> tuple[int, int]:
    """Return (total_bytes, file_count) for a file or directory tree."""
    if path.is_file():
        return path.stat().st_size, 1
    total_bytes = 0
    file_count = 0
    for p in path.rglob("*"):
        if p.is_file():
            total_bytes += p.stat().st_size
            file_count += 1
    return total_bytes, file_count


def format_bytes(num_bytes: int | float) -> str:
    """Human-readable byte formatter."""
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(value) < 1024.0:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} PB"
