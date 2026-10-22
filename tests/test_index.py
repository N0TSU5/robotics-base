from collections import Counter

import numpy as np

from sawyer.data.index import WINDOW_STEPS, DatasetConfig, EpisodeIndex, WindowSampler, validation_windows


def _config(name, weight):
    return DatasetConfig(name, f"repo/{name}", "franka", {"cam": "left"}, weight)


def _episodes(lengths, tasks):
    return [{"episode_index": i, "length": n, "tasks": [t]} for i, (n, t) in enumerate(zip(lengths, tasks))]


def test_short_episodes_dropped_and_stride_from_fps():
    # 15 fps -> stride 3: a 9-step window needs 25 raw frames
    ix = EpisodeIndex(_config("a", 1.0), {"fps": 15}, episodes=_episodes([24, 25, 100], ["x", "y", "z"]))
    assert ix.stride == 3 and ix.num_dropped == 1
    assert list(ix.episode) == [1, 2] and list(ix.steps) == [9, 34]


def test_windows_fit_and_text_pools_are_separate():
    rng = np.random.default_rng(0)
    ix = EpisodeIndex(_config("a", 1.0), {"fps": 5}, episodes=_episodes([10, 40, 12, 30], ["pick", "", "  ", "place"]))
    for has_text in (True, False):
        for _ in range(500):
            w = ix.sample(rng, has_text)
            row = list(ix.episode).index(w.episode)
            assert 0 <= w.start and w.start + WINDOW_STEPS <= ix.steps[row]
            assert bool(w.instruction) == has_text  # no-text comes from the empty-instruction episodes
            assert w.has_text == has_text


def test_no_text_drops_instruction_when_dataset_has_no_empty_episodes():
    rng = np.random.default_rng(0)
    ix = EpisodeIndex(_config("ur5", 1.0), {"fps": 5}, episodes=_episodes([20, 20], ["sweep", "pour"]))
    w = ix.sample(rng, has_text=False)
    assert not w.has_text and w.instruction in ("sweep", "pour")


def test_episodes_drawn_in_proportion_to_their_windows():
    rng = np.random.default_rng(0)
    ix = EpisodeIndex(_config("a", 1.0), {"fps": 5}, episodes=_episodes([9, 18], ["a", "b"]))  # 1 vs 10 windows
    counts = Counter(ix.sample(rng, True).episode for _ in range(11000))
    assert abs(counts[1] / counts[0] - 10) < 1.5


def test_mixture_weights_and_nine_to_one_split():
    rng = np.random.default_rng(0)
    eps = _episodes([30] * 4, ["a", "b", "", "c"])
    sampler = WindowSampler([EpisodeIndex(_config(n, w), {"fps": 5}, eps) for n, w in [("d", 0.55), ("b", 0.30), ("u", 0.07), ("k", 0.08)]])
    windows = [sampler.sample(rng) for _ in range(40000)]
    share = Counter(w.dataset for w in windows)
    for name, weight in [("d", 0.55), ("b", 0.30), ("u", 0.07), ("k", 0.08)]:
        assert abs(share[name] / len(windows) - weight) < 0.01
    assert abs(np.mean([not w.has_text for w in windows]) - 0.1) < 0.006


def test_episode_file_paths_follow_lerobot_v2_layout():
    info = {
        "fps": 5,
        "chunks_size": 1000,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
    }
    ix = EpisodeIndex(_config("a", 1.0), info, _episodes([20], ["x"]))
    assert ix.data_file(27019) == "data/chunk-027/episode_027019.parquet"
    assert ix.video_file(16, "observation.images.image_3") == "videos/chunk-000/observation.images.image_3/episode_000016.mp4"


def test_validation_split_is_disjoint_stable_and_two_percent():
    eps = _episodes([20] * 5000, ["x"] * 5000)
    train = EpisodeIndex(_config("a", 1.0), {"fps": 5}, eps, split="train")
    val = EpisodeIndex(_config("a", 1.0), {"fps": 5}, eps, split="val")
    assert set(train.episode).isdisjoint(val.episode) and len(train.episode) + len(val.episode) == 5000
    assert abs(len(val.episode) / 5000 - 0.02) < 0.005
    again = EpisodeIndex(_config("a", 1.0), {"fps": 5}, eps, split="val")
    assert list(again.episode) == list(val.episode)


def test_validation_windows_are_fixed_and_grouped_by_dataset():
    eps = _episodes([30] * 200, ["x"] * 200)
    indexes = [EpisodeIndex(_config(n, 1.0), {"fps": 5}, eps, split="val") for n in ("d", "b")]
    windows = validation_windows(indexes, per_dataset=16)
    assert [w.dataset for w in windows] == ["d"] * 16 + ["b"] * 16
    assert windows == validation_windows(indexes, per_dataset=16) and all(w.has_text for w in windows)
