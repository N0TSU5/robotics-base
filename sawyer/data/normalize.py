"""Per-dataset action normalization from 1st / 99th percentiles, and per-dataset state-change scales.

The six pose deltas map [q01, q99] -> [-1, 1] and are clipped, so extreme teleop spikes saturate instead of
dominating the loss. The gripper delta is left as is: closedness deltas already lie in [-1, 1], and grasp / release
steps are rare enough (about 2% of UR5 steps) that its percentiles can sit at 0 and clipping would erase them.

state_scale is the mean absolute per-step change of each of the 10 state-target dims. The state loss divides its
error by it, so a slow robot's small motions (UR5: ~1 cm per step) weigh as much as a fast robot's large ones. The
scale is shared within each group (xyz, 6D rotation, gripper): single rotation components can barely move (UR5:
0.0006), and dividing by them alone would blow tiny noise up into the dominant error.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

GRIPPER = 6  # index of dgripper in the 7-d action


STATE_SCALE_FLOOR = 1e-4  # dims that never change in a dataset (e.g. a gripper that never moves) stay finite
STATE_GROUPS = (slice(0, 3), slice(3, 9), slice(9, 10))  # xyz, 6D rotation, gripper


@dataclass
class ActionStats:
    q01: list[float]
    q99: list[float]
    state_scale: list[float] | None = None  # (10,) mean |per-step change| of the state target; None = no scaling

    @classmethod
    def from_actions(cls, actions: np.ndarray) -> "ActionStats":
        """actions (N, 7) per-step deltas."""
        q01, q99 = np.percentile(actions, [1, 99], axis=0)
        q01[GRIPPER], q99[GRIPPER] = -1.0, 1.0  # identity map for the gripper
        return cls(q01.tolist(), q99.tolist())

    def state_scale_array(self) -> np.ndarray:
        """(10,) divisor for state-target errors: the group-mean change, floored."""
        if self.state_scale is None:
            return np.ones(10, dtype=np.float32)
        per_dim = np.asarray(self.state_scale)
        out = np.empty(10)
        for group in STATE_GROUPS:
            out[group] = per_dim[group].mean()
        return np.maximum(out, STATE_SCALE_FLOOR).astype(np.float32)

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
