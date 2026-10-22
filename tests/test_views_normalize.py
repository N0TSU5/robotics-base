from collections import Counter

import numpy as np

from sawyer.data.normalize import GRIPPER, ActionStats, load_stats, save_stats
from sawyer.data.views import select_views


def test_wrist_always_kept_with_one_random_static_view():
    rng = np.random.default_rng(0)
    cameras = ["left", "right", "wrist"]  # DROID
    picks = [select_views(cameras, rng) for _ in range(2000)]
    assert all(len(p) == 2 and 2 in p for p in picks)
    statics = Counter(next(i for i in p if i != 2) for p in picks)
    assert abs(statics[0] / 2000 - 0.5) < 0.05
    first_slot = Counter(p[0] for p in picks)
    assert abs(first_slot[2] / 2000 - 0.5) < 0.05  # wrist lands in either slot


def test_no_wrist_takes_two_random_statics():
    rng = np.random.default_rng(0)
    picks = [select_views(["right", "side", "side"], rng) for _ in range(300)]  # Bridge without image_3
    assert all(len(p) == 2 and len(set(p)) == 2 for p in picks)
    assert {i for p in picks for i in p} == {0, 1, 2}


def test_single_camera_leaves_second_slot_empty():
    assert select_views(["left"], np.random.default_rng(0)) == [0]  # kuka


def test_percentile_normalization_clips_spikes_and_round_trips():
    rng = np.random.default_rng(0)
    actions = rng.normal(0, 0.01, size=(10000, 7))
    actions[:, GRIPPER] = 0
    actions[:50, GRIPPER] = 1.0  # rare grasp events (0.5% of steps)
    actions[0, 0] = 5.0  # teleop spike
    stats = ActionStats.from_actions(actions)
    norm = stats.normalize(actions)
    assert norm.min() >= -1 and norm.max() <= 1
    assert norm[0, 0] == 1.0  # spike saturates
    assert (norm[:50, GRIPPER] == 1.0).all() and (norm[50:, GRIPPER] == 0).all()  # gripper untouched
    inside = (np.abs(norm[:, :6]) < 1).all(axis=1)
    np.testing.assert_allclose(stats.denormalize(norm)[inside], actions[inside], atol=1e-6)


def test_stats_save_and_load(tmp_path):
    stats = {"ur5": ActionStats([-1.0] * 7, [1.0] * 7)}
    save_stats(stats, tmp_path / "stats.json")
    assert load_stats(tmp_path / "stats.json") == stats
