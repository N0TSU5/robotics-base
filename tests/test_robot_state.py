from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
from scipy.spatial.transform import Rotation

from sawyer.data.robot_state import STATE_SPECS, to_actions, to_input_state, to_pose7, to_state_target

SAMPLES = Path(__file__).parents[1] / "sample_data"
PORTS = {"droid": "droid", "bridge": "bridge_orig_lerobot", "kuka": "kuka_lerobot", "ur5": "berkeley_autolab_ur5_lerobot"}


def _raw(name, episode=0):
    path = SAMPLES / PORTS[name] / f"episode_{episode:06d}.parquet"
    if not path.exists():
        pytest.skip(f"{path} not downloaded")
    return np.stack(pq.read_table(path).column("observation.state").to_pylist()).astype(np.float64)


def _random_poses(T=20, seed=0):
    rng = np.random.default_rng(seed)
    rot = Rotation.random(T, random_state=seed).as_euler("xyz")
    return np.concatenate([rng.normal(size=(T, 3)), rot, rng.uniform(size=(T, 1))], axis=1)


def test_euler_layout_drops_pad_and_keeps_closedness():
    raw = _raw("droid")
    pose = to_pose7(raw, STATE_SPECS["droid"])
    np.testing.assert_allclose(pose[:, :6], raw[:, :6], rtol=1e-6)
    np.testing.assert_allclose(pose[:, 6], np.clip(raw[:, 7], 0, 1), rtol=1e-6)


def test_bridge_gripper_inverted_to_closedness():
    raw = _raw("bridge")
    pose = to_pose7(raw, STATE_SPECS["bridge"])
    assert pose[0, 6] < 0.05  # episode starts with the gripper open (raw ~0.99)
    np.testing.assert_allclose(pose[:, 6], 1 - np.clip(raw[:, 7], 0, 1), rtol=1e-6)


@pytest.mark.parametrize("name", ["kuka", "ur5"])
def test_quaternion_layout_gives_same_rotation(name):
    raw = _raw(name)
    pose = to_pose7(raw, STATE_SPECS[name])
    angle = (Rotation.from_euler("xyz", pose[:, 3:6]) * Rotation.from_quat(raw[:, 3:7]).inv()).magnitude()
    assert angle.max() < 1e-5


def test_ur5_gripper_is_closedness():
    pose = to_pose7(_raw("ur5", episode=2), STATE_SPECS["ur5"])  # pick-and-place: open, grasp, release
    assert pose[0, 6] == 0 and pose[len(pose) // 2, 6] == 1 and pose[-1, 6] == 0


def test_input_state_is_zero_padded_raw_pose():
    pose = _random_poses().astype(np.float32)
    state = to_input_state(pose)
    assert state.shape == (20, 64)
    np.testing.assert_array_equal(state[:, :7], pose)
    assert not state[:, 7:].any()


def test_state_target_is_continuous_across_roll_wrap():
    roll = np.array([np.pi - 0.01, -np.pi + 0.01])  # 0.02 rad apart, 2pi apart as numbers
    pose = np.zeros((2, 7))
    pose[:, 3] = roll
    target = to_state_target(pose)
    assert np.abs(np.diff(target[:, 3:9], axis=0)).max() < 0.03
    assert np.abs(to_actions(pose)[0, 3]) < 0.03  # the action is wrap-safe too


def test_actions_integrate_back_to_poses():
    pose = _random_poses()
    actions = to_actions(pose)
    xyz, rot, grip = pose[0, :3].copy(), Rotation.from_euler("xyz", pose[0, 3:6]), pose[0, 6]
    for t, a in enumerate(actions, start=1):
        xyz, rot, grip = xyz + a[:3], Rotation.from_euler("xyz", a[3:6]) * rot, grip + a[6]
        np.testing.assert_allclose(xyz, pose[t, :3], atol=1e-5)
        assert (rot * Rotation.from_euler("xyz", pose[t, 3:6]).inv()).magnitude() < 1e-5
        np.testing.assert_allclose(grip, pose[t, 6], atol=1e-5)
