"""Which camera views go into a sample."""

import numpy as np

MAX_VIEWS = 2  # deployment UR5e: 1 static + 1 wrist camera


def select_views(cameras: list[str], rng: np.random.Generator, max_views: int = MAX_VIEWS) -> list[int]:
    """Pick up to max_views of the available (non-blank) cameras; returns indices into `cameras`.

    A wrist camera is always kept when present and the remaining slots are filled with randomly chosen static views.
    Slot order is shuffled, so the RoPE view slot carries no camera identity (that comes from the camera type).
    Fewer than max_views indices means the remaining slots are padding, masked out by view_valid.
    """
    wrist = [i for i, c in enumerate(cameras) if c == "wrist"]
    static = [i for i, c in enumerate(cameras) if c != "wrist"]
    chosen = wrist[:1]
    chosen += [static[i] for i in rng.permutation(len(static))[: max_views - len(chosen)]]
    return [chosen[i] for i in rng.permutation(len(chosen))]
