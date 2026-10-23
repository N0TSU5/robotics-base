"""Window -> training sample: frames, cameras, input states, state targets, and action-chunk targets."""

import logging
import os
from concurrent.futures import ThreadPoolExecutor

import av
import numpy as np
import pyarrow.parquet as pq
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from sawyer.data.hub import HubCache
from sawyer.data.index import WINDOW_STEPS, EpisodeIndex, Window, WindowSampler
from sawyer.data.normalize import ActionStats
from sawyer.data.robot_state import STATE_SPECS, to_actions, to_input_state, to_pose7, to_state_target
from sawyer.data.views import MAX_VIEWS, select_views
from sawyer.inputs import CAMERAS, EMBODIMENTS
from sawyer.model import ACTION_HORIZON

logger = logging.getLogger(__name__)


def decode_frames(path, indices: np.ndarray, size: int) -> np.ndarray:
    """Decode the frames at `indices` (ascending) and squash-resize them -> (len(indices), size, size, 3) uint8."""
    wanted = {int(i): k for k, i in enumerate(indices)}
    out = np.empty((len(indices), size, size, 3), dtype=np.uint8)
    found = 0
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        stream.thread_count = 2  # several DataLoader workers share the CPU
        for i, frame in enumerate(container.decode(stream)):
            if i in wanted:
                out[wanted[i]] = frame.reformat(width=size, height=size, format="rgb24", interpolation="BICUBIC").to_ndarray()
                found += 1
            if i >= indices[-1]:
                break
    if found != len(indices):
        raise ValueError(f"{path}: needed frame {indices[-1]}, video has fewer")
    return out


class WindowLoader:
    def __init__(self, indexes: dict[str, EpisodeIndex], stats: dict[str, ActionStats], cache: HubCache, image_size: int = 224):
        self.indexes = indexes
        self.stats = stats
        self.cache = cache
        self.image_size = image_size
        # Each Hub request costs ~1 s mostly in latency, so an episode's files are fetched concurrently.
        self.pool = ThreadPoolExecutor(max_workers=4)

    def load(self, window: Window, rng: np.random.Generator) -> dict:
        index = self.indexes[window.dataset]
        config, stride = index.config, window.stride
        video_keys, names = zip(*config.cameras.items())
        # Views are picked before downloading, except where unrecorded cameras hide behind black placeholder videos.
        fetch = list(range(len(names))) if config.placeholder_views else select_views(list(names), rng)
        files = [index.data_file(window.episode)] + [index.video_file(window.episode, video_keys[i]) for i in fetch]
        paths = list(self.pool.map(lambda f: self.cache.get(config.repo, f), files))

        raw = pq.read_table(paths[0], columns=["observation.state"])
        raw = np.stack(raw.column(0).to_pylist()).astype(np.float64)
        # Steps of the window plus up to ACTION_HORIZON more, so late steps still get (partly valid) action chunks.
        n = min(WINDOW_STEPS + ACTION_HORIZON, (len(raw) - 1) // stride + 1 - window.start)
        pose = to_pose7(raw[(window.start + np.arange(n)) * stride], STATE_SPECS[config.name])
        actions = self.stats[config.name].normalize(to_actions(pose))  # (n - 1, 7)
        step = np.arange(WINDOW_STEPS)[:, None] + np.arange(ACTION_HORIZON)[None, :]  # action index of chunk (t, k)
        action_valid = step < len(actions)
        action_target = np.where(action_valid[..., None], actions[np.minimum(step, len(actions) - 1)], 0.0)

        frame_idx = (window.start + np.arange(WINDOW_STEPS)) * stride
        decoded = {i: decode_frames(p, frame_idx, self.image_size) for i, p in zip(fetch, paths[1:])}
        available = [i for i in fetch if decoded[i].max() >= 8]  # Bridge's unrecorded cameras are all black
        if not available or (not config.placeholder_views and len(available) < len(fetch)):
            raise ValueError(f"{window.dataset} episode {window.episode}: blank camera")
        picks = [available[i] for i in select_views([names[i] for i in available], rng)]
        frames, cameras = np.stack([decoded[i] for i in picks]), [CAMERAS[names[i]] for i in picks]

        view_valid = np.zeros(MAX_VIEWS, dtype=bool)
        view_valid[: len(cameras)] = True
        frames = np.concatenate([frames, np.zeros((MAX_VIEWS - len(cameras), *frames.shape[1:]), np.uint8)])
        return {
            "frames": frames.transpose(1, 0, 2, 3, 4),  # (T, V, H, W, 3) uint8
            "camera": np.array(cameras + [0] * (MAX_VIEWS - len(cameras))),
            "view_valid": view_valid,
            "state": to_input_state(pose[:WINDOW_STEPS]),
            "state_target": to_state_target(pose[:WINDOW_STEPS]),
            "action_target": action_target.astype(np.float32),
            "action_valid": action_valid,
            "embodiment": EMBODIMENTS.index(config.embodiment),
            "instruction": window.instruction,
            "has_text": window.has_text,
            "dataset": window.dataset,
            "action_zero": self.stats[config.name].normalize(np.zeros((1, 7)))[0],  # "no motion", for baselines
        }


class WindowDataset(IterableDataset):
    """Endless stream of training samples. Each worker (and rank) samples windows with its own RNG; a window that fails
    to fetch or decode is logged and replaced, and too many consecutive failures raise.

    windows_per_episode > 1 draws that many windows from each fetched episode; the extra ones read the local cache,
    which keeps Hub requests under its rate limit. Samples pass through a shuffle buffer of shuffle_buffer samples so
    windows of one episode spread across batches. Episodes are still drawn in proportion to their window count and
    the extra windows are uniform within the episode, so every window stays equally likely.

    With group_size = the DataLoader batch size, samples are emitted in runs of group_size that share a view count.
    A DataLoader worker builds each batch from its own consecutive samples, so every batch is all-1-view or
    all-2-view and the collator can drop the empty slot instead of computing on padding.
    """

    def __init__(self, indexes: list[EpisodeIndex], stats: dict[str, ActionStats], cache_root: str, cache_bytes: float,
                 image_size: int = 224, seed: int = 0, max_failures: int = 50, group_size: int = 1,
                 windows_per_episode: int = 1, shuffle_buffer: int = 1):  # fmt: skip
        self.indexes = indexes
        self.stats = stats
        self.cache_root = cache_root
        self.cache_bytes = cache_bytes
        self.image_size = image_size
        self.seed = seed
        self.max_failures = max_failures
        self.group_size = group_size
        self.windows_per_episode = windows_per_episode
        self.shuffle_buffer = shuffle_buffer

    def _samples(self, rng):
        sampler = WindowSampler(self.indexes)
        by_name = {ix.config.name: ix for ix in self.indexes}
        loader = WindowLoader(by_name, self.stats, HubCache(self.cache_root, self.cache_bytes), self.image_size)
        failures = 0
        while True:
            first = sampler.sample(rng)
            for i in range(self.windows_per_episode):
                window = first if i == 0 else by_name[first.dataset].resample_start(first, rng)
                try:
                    sample = loader.load(window, rng)
                except Exception as e:  # network, missing file, short or corrupt video: move to another episode
                    failures += 1
                    logger.warning("skipping %s: %r", window, e)
                    if failures > self.max_failures:
                        raise RuntimeError(f"{failures} consecutive windows failed") from e
                    break
                failures = 0
                if not by_name[first.dataset].config.train_actions:
                    sample["action_valid"] = np.zeros_like(sample["action_valid"])
                yield sample

    def __iter__(self):
        worker = get_worker_info()
        rng = np.random.default_rng([self.seed, int(os.environ.get("RANK", 0)), worker.id if worker else 0])
        buffer, groups = [], {}
        for sample in self._samples(rng):
            buffer.append(sample)
            if len(buffer) < self.shuffle_buffer:
                continue
            i = int(rng.integers(len(buffer)))
            buffer[i], buffer[-1] = buffer[-1], buffer[i]
            sample = buffer.pop()
            if self.group_size == 1:
                yield sample
                continue
            group = groups.setdefault(int(sample["view_valid"].sum()), [])
            group.append(sample)
            if len(group) == self.group_size:
                yield from group
                group.clear()


class FixedWindows(Dataset):
    """A fixed list of windows (the validation set), loaded on demand. View choice is seeded per item, so every
    evaluation sees identical samples. It uses its own cache so the training cache's evictions don't force
    re-downloads (and Hub requests) at each evaluation."""

    def __init__(self, indexes: list[EpisodeIndex], stats: dict[str, ActionStats], windows: list[Window], cache_root: str,
                 cache_bytes: float, image_size: int = 224, retries: int = 3):  # fmt: skip
        self.indexes = indexes
        self.stats = stats
        self.windows = windows
        self.cache_root = cache_root
        self.cache_bytes = cache_bytes
        self.image_size = image_size
        self.retries = retries
        self._loader = None

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, i):
        if self._loader is None:  # built inside each DataLoader worker
            self._loader = WindowLoader({ix.config.name: ix for ix in self.indexes}, self.stats,
                                        HubCache(self.cache_root, self.cache_bytes), self.image_size)  # fmt: skip
        for attempt in range(self.retries):
            try:
                return self._loader.load(self.windows[i], np.random.default_rng(i))
            except Exception as e:
                logger.warning("validation window %s failed (attempt %d): %r", self.windows[i], attempt + 1, e)
                error = e
        raise RuntimeError(f"validation window {self.windows[i]} failed {self.retries} times") from error
