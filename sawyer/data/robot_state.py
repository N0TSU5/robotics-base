"""Robot state conversions for the IPEC-COMMUNITY LeRobot ports.

Every dataset is first reduced to a common 7-d end-effector pose: xyz (m), euler xyz (rad, scipy extrinsic "xyz",
as RoboJEPA uses), gripper closedness in [0, 1]. Three things are derived from it:
  - input state: the pose zero-padded to 64, the raw layout RoboJEPA's copied state encoder expects
  - state target: xyz + 6D rotation + gripper (10-d), what the predictor's state head regresses
  - actions: RoboJEPA's state delta between consecutive steps (dxyz, euler of R_{t+1} R_t^T, dgripper)
"""

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from sawyer.inputs import STATE_DIM


@dataclass(frozen=True)
class StateSpec:
    rotation: str  # "euler": roll, pitch, yaw at [3:6] then a pad slot; "quat": qx, qy, qz, qw at [3:7]
    gripper_is_openness: bool  # True when 1 = open, so it must be inverted to closedness


# observation.state is 8-d in every port, with the gripper last. Polarity checked on sample episodes.
STATE_SPECS = {
    "droid": StateSpec("euler", gripper_is_openness=False),
    "bridge": StateSpec("euler", gripper_is_openness=True),
    "kuka": StateSpec("quat", gripper_is_openness=False),
    "ur5": StateSpec("quat", gripper_is_openness=False),
}


def to_pose7(raw: np.ndarray, spec: StateSpec) -> np.ndarray:
    """(T, 8) observation.state -> (T, 7) xyz, euler xyz, gripper closedness."""
    if spec.rotation == "euler":
        euler = raw[:, 3:6]
    else:
        euler = Rotation.from_quat(raw[:, 3:7]).as_euler("xyz")
    gripper = np.clip(raw[:, 7:8], 0.0, 1.0)
    if spec.gripper_is_openness:
        gripper = 1.0 - gripper
    return np.concatenate([raw[:, :3], euler, gripper], axis=1).astype(np.float32)


def to_input_state(pose7: np.ndarray) -> np.ndarray:
    """(T, 7) -> (T, 64), zero-padded and unnormalized, as RoboJEPA's state encoder was trained."""
    out = np.zeros((len(pose7), STATE_DIM), dtype=np.float32)
    out[:, :7] = pose7
    return out


def to_state_target(pose7: np.ndarray) -> np.ndarray:
    """(T, 7) -> (T, 10): xyz, first two columns of the rotation matrix, gripper. Continuous where euler wraps at ±pi."""
    rot = Rotation.from_euler("xyz", pose7[:, 3:6]).as_matrix()  # (T, 3, 3)
    rot6d = np.concatenate([rot[:, :, 0], rot[:, :, 1]], axis=1)
    return np.concatenate([pose7[:, :3], rot6d, pose7[:, 6:7]], axis=1).astype(np.float32)


def to_actions(pose7: np.ndarray) -> np.ndarray:
    """(T, 7) -> (T - 1, 7): dxyz, relative rotation R_{t+1} R_t^T (world frame) as euler xyz, dgripper."""
    rot = Rotation.from_euler("xyz", pose7[:, 3:6])
    rel = (rot[1:] * rot[:-1].inv()).as_euler("xyz")
    return np.concatenate([np.diff(pose7[:, :3], axis=0), rel, np.diff(pose7[:, 6:7], axis=0)], axis=1).astype(np.float32)
