"""Dataset registry, episode index, and the window sampler.

A window is WINDOW_STEPS consecutive steps at CONTROL_HZ inside one episode. The sampler picks a dataset by mixture
weight, decides text / no-text with a 9:1 split, then picks a window uniformly among that dataset's eligible windows.
No-text windows come from episodes with an empty instruction when the dataset has any (DROID, Bridge), otherwise the
instruction is dropped (UR5, kuka).
"""

import dataclasses
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download

CONTROL_HZ = 5
WINDOW_STEPS = 9  # step 0 plus 8 future steps: 1.6 s
NO_TEXT_PROB = 0.1  # 9:1 text / no-text
VAL_EVERY = 50  # 1 episode in 50 (2%) is held out for validation, chosen by a fixed hash of its index


def is_validation(episode: np.ndarray) -> np.ndarray:
    return (episode.astype(np.int64) * 2654435761 % 2**32) % VAL_EVERY == 0


@dataclass(frozen=True)
class DatasetConfig:
    name: str  # key into robot_state.STATE_SPECS
    repo: str  # IPEC-COMMUNITY LeRobot v2 port on the Hub
    embodiment: str  # key into inputs.EMBODIMENTS
    cameras: dict[str, str]  # video feature key -> inputs.CAMERAS name
    weight: float  # share of sampled windows
    placeholder_views: bool = False  # unrecorded cameras stored as all-black videos: fetch all, then pick


DATASETS = [
    DatasetConfig("droid", "IPEC-COMMUNITY/droid_lerobot", "franka", {
        "observation.images.exterior_image_1_left": "left",
        "observation.images.exterior_image_2_left": "right",
        "observation.images.wrist_image_left": "wrist",
    }, weight=0.56),
    DatasetConfig("bridge", "IPEC-COMMUNITY/bridge_orig_lerobot", "bridge", {
        "observation.images.image_0": "right",  # over-the-shoulder, RoboJEPA's "right_back"
        "observation.images.image_1": "side",  # image_1..3 are all-black placeholders when not recorded
        "observation.images.image_2": "side",
        "observation.images.image_3": "wrist",
    }, weight=0.36, placeholder_views=True),
    DatasetConfig("ur5", "IPEC-COMMUNITY/berkeley_autolab_ur5_lerobot", "ur5", {
        "observation.images.image": "front",
        "observation.images.hand_image": "wrist",
    }, weight=0.03),
    DatasetConfig("kuka", "IPEC-COMMUNITY/kuka_lerobot", "kuka", {
        "observation.images.image": "left",
    }, weight=0.05),
]  # fmt: skip


@dataclass(frozen=True)
class Window:
    dataset: str
    episode: int  # episode_index in the port
    start: int  # first step, counted in CONTROL_HZ steps
    stride: int  # raw frames per step
    instruction: str
    has_text: bool


class EpisodeIndex:
    """Episodes of one dataset that fit a window, split into a text pool and a no-text pool."""

    def __init__(self, config: DatasetConfig, info: dict, episodes: list[dict], split: str | None = None):
        """split: "train" drops the held-out episodes, "val" keeps only them, None keeps everything."""
        self.config = config
        self.info = info
        self.stride = math.ceil(info["fps"] / CONTROL_HZ)
        episode = np.array([e["episode_index"] for e in episodes])
        steps = (np.array([e["length"] for e in episodes]) - 1) // self.stride + 1
        text = [next((t.strip() for t in e["tasks"] if t.strip()), "") for e in episodes]
        keep = steps >= WINDOW_STEPS
        if split is not None:
            keep &= is_validation(episode) == (split == "val")
        self.episode, self.steps = episode[keep], steps[keep]
        self.instruction = [t for t, k in zip(text, keep) if k]
        self.has_text = np.array([bool(t) for t in self.instruction])
        self.num_dropped = int((~keep).sum())

        self._row = {int(e): i for i, e in enumerate(self.episode)}

        text_pool = np.flatnonzero(self.has_text)
        empty_pool = np.flatnonzero(~self.has_text)
        self.pools = {True: text_pool, False: empty_pool if len(empty_pool) else np.arange(len(self.episode))}
        # Episodes are drawn proportionally to their window count, so every window in a pool is equally likely.
        self.cum_windows = {k: np.cumsum(self.steps[p] - WINDOW_STEPS + 1) for k, p in self.pools.items()}

    def data_file(self, episode: int) -> str:
        """Repo path of an episode's parquet (LeRobot v2 layout)."""
        chunk = episode // self.info["chunks_size"]
        return self.info["data_path"].format(episode_chunk=chunk, episode_index=episode)

    def video_file(self, episode: int, video_key: str) -> str:
        chunk = episode // self.info["chunks_size"]
        return self.info["video_path"].format(episode_chunk=chunk, video_key=video_key, episode_index=episode)

    def resample_start(self, window: Window, rng: np.random.Generator) -> Window:
        """Another uniformly chosen window of the same episode (same instruction and text decision)."""
        steps = self.steps[self._row[window.episode]]
        return dataclasses.replace(window, start=int(rng.integers(steps - WINDOW_STEPS + 1)))

    def sample(self, rng: np.random.Generator, has_text: bool) -> Window:
        pool, cum = self.pools[has_text], self.cum_windows[has_text]
        w = rng.integers(cum[-1])
        i = pool[np.searchsorted(cum, w, side="right")]
        start = int(rng.integers(self.steps[i] - WINDOW_STEPS + 1))
        return Window(self.config.name, int(self.episode[i]), start, self.stride, self.instruction[i], has_text)


class WindowSampler:
    def __init__(self, indexes: list[EpisodeIndex], no_text_prob: float = NO_TEXT_PROB):
        self.indexes = indexes
        weights = np.array([ix.config.weight for ix in indexes])
        self.probs = weights / weights.sum()
        self.no_text_prob = no_text_prob

    def sample(self, rng: np.random.Generator) -> Window:
        index = self.indexes[rng.choice(len(self.indexes), p=self.probs)]
        return index.sample(rng, has_text=rng.random() >= self.no_text_prob)


def validation_windows(indexes: list[EpisodeIndex], per_dataset: int, seed: int = 0) -> list[Window]:
    """A fixed, text-conditioned set of windows per dataset, in dataset order, for repeatable evaluation."""
    rng = np.random.default_rng(seed)
    return [ix.sample(rng, has_text=True) for ix in indexes for _ in range(per_dataset)]


def load_index(config: DatasetConfig, meta_root: str | Path, split: str | None = None) -> EpisodeIndex:
    """Download (once) and parse meta/info.json and meta/episodes.jsonl of a port."""
    local = Path(meta_root) / config.name
    info = json.loads(Path(hf_hub_download(config.repo, "meta/info.json", repo_type="dataset", local_dir=local)).read_text())
    path = hf_hub_download(config.repo, "meta/episodes.jsonl", repo_type="dataset", local_dir=local)
    with open(path) as f:
        episodes = [json.loads(line) for line in f]
    return EpisodeIndex(config, info, episodes, split)
