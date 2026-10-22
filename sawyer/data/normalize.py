"""Per-dataset action normalization from 1st / 99th percentiles.

The six pose deltas map [q01, q99] -> [-1, 1] and are clipped, so extreme teleop spikes saturate instead of
dominating the loss. The gripper delta is left as is: closedness deltas already lie in [-1, 1], and grasp / release
steps are rare enough (about 2% of UR5 steps) that its percentiles can sit at 0 and clipping would erase them.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

GRIPPER = 6  # index of dgripper in the 7-d action


@dataclass
class ActionStats:
    q01: list[float]
    q99: list[float]

    @classmethod
    def from_actions(cls, actions: np.ndarray) -> "ActionStats":
        """actions (N, 7) per-step deltas."""
        q01, q99 = np.percentile(actions, [1, 99], axis=0)
        q01[GRIPPER], q99[GRIPPER] = -1.0, 1.0  # identity map for the gripper
        return cls(q01.tolist(), q99.tolist())

    def normalize(self, actions: np.ndarray) -> np.ndarray:
        q01, q99 = np.asarray(self.q01), np.asarray(self.q99)
        out = 2 * (actions - q01) / np.maximum(q99 - q01, 1e-8) - 1
        return np.clip(out, -1, 1).astype(np.float32)

    def denormalize(self, actions: np.ndarray) -> np.ndarray:
        """Model output -> per-step deltas in m / rad / closedness, for the robot controller."""
        q01, q99 = np.asarray(self.q01), np.asarray(self.q99)
        return ((actions + 1) / 2 * (q99 - q01) + q01).astype(np.float32)


def save_stats(stats: dict[str, ActionStats], path: str | Path) -> None:
    Path(path).write_text(json.dumps({name: asdict(s) for name, s in stats.items()}, indent=2))


def load_stats(path: str | Path) -> dict[str, ActionStats]:
    return {name: ActionStats(**s) for name, s in json.loads(Path(path).read_text()).items()}
