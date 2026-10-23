"""Estimate per-dataset action percentiles and state-change scales from a random sample of episodes; writes
data/action_stats.json. With HF_HUB_OFFLINE=1 it reuses the episodes already in --cache without any Hub request."""

import argparse
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from sawyer.data.index import DATASETS, load_index
from sawyer.data.normalize import ActionStats, save_stats
from sawyer.data.robot_state import STATE_SPECS, to_actions, to_pose7, to_state_target


def episode_deltas(index, episode: int, cache: str) -> tuple[np.ndarray, np.ndarray]:
    """(actions (N, 7), state-target changes (N, 10)) at the 5 Hz step size."""
    path = hf_hub_download(index.config.repo, index.data_file(episode), repo_type="dataset", local_dir=f"{cache}/{index.config.name}")
    raw = np.stack(pq.read_table(path, columns=["observation.state"]).column(0).to_pylist()).astype(np.float64)
    pose = to_pose7(raw, STATE_SPECS[index.config.name])
    # Every phase of the stride, so all raw frames contribute at the 5 Hz step size.
    phases = [pose[offset :: index.stride] for offset in range(index.stride)]
    actions = np.concatenate([to_actions(p) for p in phases])
    state_changes = np.concatenate([np.diff(to_state_target(p), axis=0) for p in phases])
    return actions, state_changes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--meta", default="data/meta")
    parser.add_argument("--cache", default="data/stats_cache")
    parser.add_argument("--out", default="data/action_stats.json")
    args = parser.parse_args()

    rng = np.random.default_rng(0)
    stats = {}
    for config in DATASETS:
        index = load_index(config, args.meta)
        episodes = rng.choice(index.episode, size=min(args.episodes, len(index.episode)), replace=False)
        with ThreadPoolExecutor(16) as pool:
            parts = list(pool.map(lambda e: episode_deltas(index, int(e), args.cache), episodes))
        actions = np.concatenate([a for a, _ in parts])
        state_changes = np.concatenate([s for _, s in parts])
        stats[config.name] = ActionStats.from_actions(actions)
        stats[config.name].state_scale = np.abs(state_changes).mean(0).tolist()
        clipped = (np.abs(stats[config.name].normalize(actions)[:, :6]) >= 1).mean()
        print(f"{config.name}: {len(episodes)} episodes, {len(actions):,} steps, {clipped:.2%} of pose values clipped")
        print(f"  q01 {np.round(stats[config.name].q01, 4)}\n  q99 {np.round(stats[config.name].q99, 4)}")
        print(f"  state_scale {np.round(stats[config.name].state_scale, 4)}")
    save_stats(stats, args.out)


if __name__ == "__main__":
    main()
