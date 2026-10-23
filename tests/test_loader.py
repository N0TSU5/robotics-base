import json
import os
import shutil
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from sawyer.data import loader as loader_mod
from sawyer.data.hub import HubCache
from sawyer.data.index import DATASETS, EpisodeIndex, Window
from sawyer.data.loader import FixedWindows, WindowDataset, WindowLoader, decode_frames
from sawyer.data.normalize import ActionStats
from sawyer.data.robot_state import STATE_SPECS, to_actions, to_pose7
from sawyer.inputs import CAMERAS, EMBODIMENTS

ROOT = Path(__file__).parents[1]
SAMPLES, META = ROOT / "sample_data", ROOT / "data/meta"
CONFIGS = {c.name: c for c in DATASETS}
STATS = {name: ActionStats([-0.05] * 6 + [-1.0], [0.05] * 6 + [1.0]) for name in CONFIGS}


def _setup(tmp_path, name, episode, length, files):
    """Index with one episode, plus a cache dir pre-filled from sample files: {video_key or 'parquet': sample path}."""
    info_path = META / name / "meta/info.json"
    paths = [SAMPLES / p for p in files.values()]
    if not info_path.exists() or not all(p.exists() for p in paths):
        pytest.skip("sample data or meta not downloaded")
    config = CONFIGS[name]
    index = EpisodeIndex(config, json.loads(info_path.read_text()), [{"episode_index": episode, "length": length, "tasks": ["do it"]}])
    for key, src in files.items():
        rel = index.data_file(episode) if key == "parquet" else index.video_file(episode, key)
        dst = tmp_path / "cache" / config.repo / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(SAMPLES / src, dst)
    loader = WindowLoader({name: index}, STATS, HubCache(tmp_path / "cache", max_bytes=1e12))
    return index, loader


def _window(index, start, episode):
    return Window(index.config.name, episode, start, index.stride, "do it", True)


def test_ur5_sample_layout_and_targets(tmp_path):
    index, loader = _setup(tmp_path, "ur5", 2, 81, {
        "parquet": "berkeley_autolab_ur5_lerobot/episode_000002.parquet",
        "observation.images.image": "berkeley_autolab_ur5_lerobot/image.mp4",
        "observation.images.hand_image": "berkeley_autolab_ur5_lerobot/hand_image.mp4",
    })  # fmt: skip
    s = loader.load(_window(index, 10, 2), np.random.default_rng(0))
    assert s["frames"].shape == (9, 2, 224, 224, 3) and s["frames"].dtype == np.uint8
    assert sorted(s["camera"]) == sorted([CAMERAS["front"], CAMERAS["wrist"]]) and s["view_valid"].all()
    assert s["embodiment"] == EMBODIMENTS.index("ur5")
    assert s["state"].shape == (9, 64) and s["state_target"].shape == (9, 10)
    assert s["action_valid"].all()

    raw = pq.read_table(SAMPLES / "berkeley_autolab_ur5_lerobot/episode_000002.parquet").column("observation.state")
    raw = np.stack(raw.to_pylist())
    pose = to_pose7(raw[10:27].astype(np.float64), STATE_SPECS["ur5"])
    actions = STATS["ur5"].normalize(to_actions(pose))
    np.testing.assert_allclose(s["action_target"][3, 5], actions[8], atol=1e-6)  # chunk at step 3, offset 5
    np.testing.assert_allclose(s["state"][:, :7], pose[:9], atol=1e-6)

    wrist_slot = list(s["camera"]).index(CAMERAS["wrist"])
    expected = decode_frames(SAMPLES / "berkeley_autolab_ur5_lerobot/hand_image.mp4", np.arange(10, 19), 224)
    np.testing.assert_array_equal(s["frames"][:, wrist_slot], expected)


def test_action_chunks_past_episode_end_are_masked(tmp_path):
    index, loader = _setup(tmp_path, "ur5", 2, 81, {
        "parquet": "berkeley_autolab_ur5_lerobot/episode_000002.parquet",
        "observation.images.image": "berkeley_autolab_ur5_lerobot/image.mp4",
        "observation.images.hand_image": "berkeley_autolab_ur5_lerobot/hand_image.mp4",
    })  # fmt: skip
    s = loader.load(_window(index, 81 - 9, 2), np.random.default_rng(0))  # last possible window: 8 actions left
    t, k = np.meshgrid(np.arange(9), np.arange(8), indexing="ij")
    np.testing.assert_array_equal(s["action_valid"], t + k < 8)
    assert not s["action_target"][~s["action_valid"]].any()


def test_droid_frames_follow_the_stride(tmp_path):
    index, loader = _setup(tmp_path, "droid", 0, 240, {
        "parquet": "droid/episode_000000.parquet",
        "observation.images.exterior_image_1_left": "droid/exterior_image_1_left.mp4",
        "observation.images.exterior_image_2_left": "droid/exterior_image_2_left.mp4",
        "observation.images.wrist_image_left": "droid/wrist_image_left.mp4",
    })  # fmt: skip
    for seed in range(4):
        s = loader.load(_window(index, 5, 0), np.random.default_rng(seed))
        assert CAMERAS["wrist"] in s["camera"] and s["view_valid"].all()
    wrist_slot = list(s["camera"]).index(CAMERAS["wrist"])
    expected = decode_frames(SAMPLES / "droid/wrist_image_left.mp4", (5 + np.arange(9)) * 3, 224)
    np.testing.assert_array_equal(s["frames"][:, wrist_slot], expected)


def test_bridge_blank_cameras_are_skipped(tmp_path):
    index, loader = _setup(tmp_path, "bridge", 1, 45, {
        "parquet": "bridge_orig_lerobot/episode_000001.parquet",
        **{f"observation.images.image_{k}": f"bridge_orig_lerobot/image_{k}.mp4" for k in range(4)},  # 1-3 blank
    })  # fmt: skip
    for seed in range(6):
        s = loader.load(_window(index, 0, 1), np.random.default_rng(seed))
        assert list(s["view_valid"]) == [True, False] and s["camera"][0] == CAMERAS["right"]
        assert s["frames"][:, 0].max() > 0 and not s["frames"][:, 1].any()


def test_bridge_with_all_cameras_keeps_wrist(tmp_path):
    index, loader = _setup(tmp_path, "bridge", 16, 23, {
        "parquet": "bridge_orig_lerobot/episode_000016.parquet",
        **{f"observation.images.image_{k}": f"bridge_orig_lerobot/ep16_image_{k}.mp4" for k in range(4)},
    })  # fmt: skip
    for seed in range(6):
        s = loader.load(_window(index, 3, 16), np.random.default_rng(seed))
        assert CAMERAS["wrist"] in s["camera"] and s["view_valid"].all()


def test_cache_evicts_least_recently_used(tmp_path):
    cache = HubCache(tmp_path, max_bytes=250)
    for i, age in enumerate([300, 100, 200]):  # file 0 oldest, then 2, then 1
        p = tmp_path / "repo" / f"f{i}"
        p.parent.mkdir(exist_ok=True)
        p.write_bytes(b"x" * 100)
        os.utime(p, (1e9 - age, 1e9 - age))
    cache.prune()
    assert sorted(p.name for p in (tmp_path / "repo").iterdir()) == ["f1", "f2"]


def test_dataset_skips_failing_windows(monkeypatch):
    calls = {"n": 0}

    def flaky(self, window, rng):
        calls["n"] += 1
        if calls["n"] <= 3:
            raise OSError("network down")
        return {"dataset": window.dataset}

    monkeypatch.setattr(loader_mod.WindowLoader, "load", flaky)
    index = EpisodeIndex(CONFIGS["ur5"], {"fps": 5}, [{"episode_index": 0, "length": 20, "tasks": ["x"]}])
    sample = next(iter(WindowDataset([index], STATS, "/nonexistent", 1e9)))
    assert sample == {"dataset": "ur5"} and calls["n"] == 4


def test_dataset_groups_samples_by_view_count(monkeypatch):
    counts = iter([1, 2, 2, 1, 2, 1, 1, 2] * 10)

    def fake(self, window, rng):
        n = next(counts)
        return {"view_valid": np.array([True, n == 2])}

    monkeypatch.setattr(loader_mod.WindowLoader, "load", fake)
    index = EpisodeIndex(CONFIGS["ur5"], {"fps": 5}, [{"episode_index": 0, "length": 20, "tasks": ["x"]}])
    stream = iter(WindowDataset([index], STATS, "/nonexistent", 1e9, group_size=3))
    for _ in range(6):
        group = [int(next(stream)["view_valid"].sum()) for _ in range(3)]
        assert len(set(group)) == 1


def test_episode_reuse_draws_extra_windows_from_the_same_episode(monkeypatch):
    seen = []

    def fake(self, window, rng):
        seen.append(window)
        return {"view_valid": np.array([True, True]), "episode": window.episode}

    monkeypatch.setattr(loader_mod.WindowLoader, "load", fake)
    eps = [{"episode_index": e, "length": 40, "tasks": ["x"]} for e in range(50)]
    index = EpisodeIndex(CONFIGS["ur5"], {"fps": 5}, eps)
    stream = iter(WindowDataset([index], STATS, "/nonexistent", 1e9, windows_per_episode=4, shuffle_buffer=32))
    out = [next(stream)["episode"] for _ in range(200)]
    for k in range(0, 40, 4):  # loads come in runs of 4 from one episode, with different starts
        run = seen[k : k + 4]
        assert len({w.episode for w in run}) == 1 and all(0 <= w.start <= 40 - 9 for w in run)
    # the shuffle buffer breaks up the runs in what training sees
    assert sum(a == b for a, b in zip(out, out[1:])) < 0.5 * len(out)


def test_kuka_actions_are_masked_in_training_only(monkeypatch):
    def fake(self, window, rng):
        return {"view_valid": np.array([True, False]), "action_valid": np.ones((9, 8), bool), "dataset": window.dataset}

    monkeypatch.setattr(loader_mod.WindowLoader, "load", fake)
    eps = [{"episode_index": 0, "length": 20, "tasks": ["x"]}]
    kuka, ur5 = EpisodeIndex(CONFIGS["kuka"], {"fps": 5}, eps), EpisodeIndex(CONFIGS["ur5"], {"fps": 5}, eps)
    assert not CONFIGS["kuka"].train_actions and CONFIGS["ur5"].train_actions
    stream = iter(WindowDataset([kuka, ur5], STATS, "/nonexistent", 1e9))
    samples = [next(stream) for _ in range(200)]
    for s in samples:
        assert s["action_valid"].any() == (s["dataset"] != "kuka")
    assert {s["dataset"] for s in samples} == {"kuka", "ur5"}
    window = Window("kuka", 0, 0, kuka.stride, "x", True)
    assert FixedWindows([kuka], STATS, [window], "/nonexistent", 1e9)[0]["action_valid"].all()  # validation keeps them
