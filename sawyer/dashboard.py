"""Training dashboard: a PNG of the training curves, redrawn at every log, viewable without a TensorBoard server."""

import os
import threading
import time

import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter
from scipy.signal import lfilter
from transformers import TrainerCallback

from sawyer.updates import SERIES_PANELS, columns

PANELS = [
    ("loss", "Total loss", True),
    ("loss_visual", "Visual loss: next-frame V-JEPA latents", True),
    ("loss_state", "State loss: next robot state", True),
    ("loss_action", "Action loss: 8-step chunks", True),
    ("grad_norm", "Gradient norm (clipped at 1.0)", True),
    ("learning_rate", "Learning rate", False),
    ("seq_per_sec", "Throughput (sequences / s)", False),
]  # (log key, title, log-scale y)
SERIES, SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#2a78d6", "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
# Validation: one series per dataset, categorical slots 1-4 in a fixed order (validated palette; the aqua and yellow
# slots sit under 3:1 contrast, so every line also carries a direct label).
DATASET_COLORS = {"droid": "#2a78d6", "bridge": "#eb6834", "ur5": "#1baf7a", "kuka": "#eda100"}
VAL_PANELS = [
    (lambda d: f"eval_{d}_action", None, "Validation action loss"),
    (lambda d: f"eval_{d}_action", lambda d: f"eval_{d}_action_zero", "Validation action ÷ zero-motion baseline"),
    (lambda d: f"eval_{d}_visual", lambda d: f"eval_{d}_visual_copy", "Validation visual ÷ copy-frame baseline"),
    (lambda d: f"eval_{d}_state", lambda d: f"eval_{d}_state_copy", "Validation state ÷ copy-state baseline"),
]  # (metric, baseline it is divided by, title)
# Series in each update-size panel: categorical slots 5-8 in order, so they never share a hue with a dataset.
UPDATE_COLORS = ["#e87ba4", "#008300", "#6250d6", "#e34948"]
# Sequential blue ramp (light = small update, dark = large) for the heatmap.
UPDATE_CMAP = LinearSegmentedColormap.from_list(
    "updates", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
)


def smooth(values: np.ndarray) -> np.ndarray:
    """Debiased EMA spanning ~5% of the points (at most 50 logs), so short runs are barely smoothed and the
    trend never lags behind from a cold start."""
    beta = 1 - 1 / np.clip(len(values) / 20, 1, 50)
    if beta == 0:
        return values.copy()
    # acc[i] = beta * acc[i - 1] + (1 - beta) * values[i], as one filter call: rendering runs in a thread that shares
    # the GIL with the training loop, and the history reaches ~60k points per panel.
    acc = lfilter([1 - beta], [1, -beta], values)
    return acc / (1 - beta ** np.arange(1, len(values) + 1))


def _status(rows: list[dict], max_steps: int, seq_per_step: int) -> str:
    last = rows[-1]
    step = last["step"]
    rates = [r["seq_per_sec"] for r in rows[-20:] if r.get("seq_per_sec")]
    lines = [f"Step {step:,} / {max_steps:,}  ({100 * step / max_steps:.1f}%)", f"Sequences seen: {step * seq_per_step:,}"]
    if rates:
        rate = float(np.median(rates))
        days = (max_steps - step) * seq_per_step / rate / 86400
        lines.append(f"Throughput: {rate:.2f} seq/s")
        lines.append(f"Time remaining: {days:.1f} days")
    lines.append("")
    lines += [f"{name}: {last[key]:.4f}" for key, name in
              [("loss", "Total"), ("loss_visual", "Visual"), ("loss_state", "State"), ("loss_action", "Action")] if key in last]  # fmt: skip
    return "\n".join(lines)


def _style(ax, title: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel("optimizer step", color=MUTED, fontsize=8)


def _empty(ax) -> None:
    ax.text(0.5, 0.5, "no data yet", transform=ax.transAxes, ha="center", va="center", color=MUTED)
    ax.set_xticks([])
    ax.set_yticks([])


def _validation_panel(ax, evals: list[dict], metric, baseline, title: str) -> None:
    _style(ax, title)
    drawn = False
    for name, color in DATASET_COLORS.items():
        pts = [(h["step"], h[metric(name)] / (h[baseline(name)] if baseline else 1)) for h in evals
               if metric(name) in h and (baseline is None or h.get(baseline(name)))]  # fmt: skip
        if not pts:
            continue
        pts = np.array(pts, dtype=float)
        ax.plot(pts[:, 0], pts[:, 1], color=color, linewidth=2, marker="o", markersize=4, label=name)
        ax.annotate(name, pts[-1], xytext=(4, 0), textcoords="offset points", va="center", color=INK_2, fontsize=8)
        drawn = True
    if not drawn:
        _empty(ax)
        return
    if baseline:
        ax.axhline(1.0, color=MUTED, linestyle="--", linewidth=1)
    ax.legend(loc="best", fontsize=8, frameon=False, labelcolor=INK_2)


def _dataset_panel(ax, rows: list[dict], value, title: str, reference: float | None = None, smoothed: bool = False):
    """One line per dataset (fixed colors, labelled at the line end). value(row, dataset) -> float or None."""
    _style(ax, title)
    drawn = False
    for name, color in DATASET_COLORS.items():
        pts = [(h["step"], v) for h in rows if (v := value(h, name)) is not None]
        if not pts:
            continue
        pts = np.array(pts, dtype=float)
        if smoothed:
            ax.plot(pts[:, 0], smooth(pts[:, 1]), color=color, linewidth=2, label=name)
        else:
            ax.plot(pts[:, 0], pts[:, 1], color=color, linewidth=2, marker="o", markersize=4, label=name)
        end = (pts[-1, 0], smooth(pts[:, 1])[-1] if smoothed else pts[-1, 1])
        ax.annotate(name, end, xytext=(4, 0), textcoords="offset points", va="center", color=INK_2, fontsize=8)
        drawn = True
    if not drawn:
        _empty(ax)
        return
    if reference is not None:
        ax.axhline(reference, color=MUTED, linestyle="--", linewidth=1)
    ax.legend(loc="best", fontsize=8, frameon=False, labelcolor=INK_2)


def _update_series_panels(axes: list, updates: list[dict]) -> None:
    """Relative update size over time for every trainable part, one small panel per stage of the network, all on
    one log scale so the panels compare directly."""
    panels = list(SERIES_PANELS.items())
    for ax in axes[1:]:
        ax.sharex(axes[0])
        ax.sharey(axes[0])
    ends = []  # (axes, series, last point) for the line-end labels, placed once the shared y range is known
    for i, (ax, (panel, series)) in enumerate(zip(axes, panels)):
        _style(ax, "Weight update per step, ‖Δw‖ / ‖w‖ (all trainable parts)" if i == 0 else "")
        if i < len(axes) - 1:
            ax.set_xlabel("")
            ax.tick_params(labelbottom=False)
        for name, color in zip(series, UPDATE_COLORS):
            pts = np.array([(r["step"], r["series"][name]) for r in updates if name in r.get("series", {})], dtype=float)
            if len(pts) == 0:
                continue
            ax.plot(pts[:, 0], pts[:, 1], color=color, linewidth=2, label=name)
            ends.append((ax, name, pts[-1]))
        ax.text(0.01, 0.95, panel, transform=ax.transAxes, va="top", color=INK, fontsize=8, weight="bold")
        if ax.lines:
            ax.legend(loc="lower left", ncol=len(ax.lines), fontsize=7, frameon=False, labelcolor=INK_2,
                      handlelength=1.2, columnspacing=1.0)  # fmt: skip
    if not ends:
        for ax in axes:
            _empty(ax)
        return
    axes[0].set_yscale("log")
    # Line-end labels, nudged apart in log space so lines ending close together keep readable labels.
    lo, hi = np.log10(axes[0].get_ylim())
    gap = 0.14 * (hi - lo)
    for ax in axes:
        mine = sorted((e for e in ends if e[0] is ax), key=lambda e: e[2][1])
        y_prev = -np.inf
        for _, name, (x, y) in mine:
            y_label = max(np.log10(y), y_prev + gap)
            ax.annotate(name, (x, y), xytext=(x, 10**y_label), textcoords="data", va="center", color=INK_2, fontsize=7,
                        annotation_clip=False)  # fmt: skip
            y_prev = y_label


def _update_heatmap(ax, updates: list[dict]) -> None:
    """Latest relative update for every trainable part at every position in the network (inputs, blocks, outputs):
    which layers and parts are learning the most right now."""
    latest = next((r for r in reversed(updates) if "cells" in r), None)
    if latest is None:
        _style(ax, "Relative weight update of every trainable part")
        _empty(ax)
        return
    rows = list(latest["cells"])
    grid = np.array([latest["cells"][r] for r in rows], dtype=float)
    grid = np.ma.masked_invalid(np.where(grid > 0, grid, np.nan))
    cols = columns(grid.shape[1] - 2)
    ax.set_facecolor(SURFACE)
    ax.set_title(f"Relative weight update of every trainable part (‖Δw‖ / ‖w‖ per step, at step {latest['step']:,}); "
                 "darker = changing faster", loc="left", color=INK, fontsize=10)  # fmt: skip
    image = ax.imshow(grid, aspect="auto", cmap=UPDATE_CMAP, norm=LogNorm(vmin=grid.min(), vmax=grid.max()),
                      interpolation="nearest")  # fmt: skip
    # Surface-colored gaps between the input, block and output sections, along both axes.
    for x in (0.5, len(cols) - 1.5):
        ax.axvline(x, color=SURFACE, linewidth=3)
    filled = ~np.ma.getmaskarray(grid)
    section = np.where(filled[:, 1:-1].any(1), 1, np.where(filled[:, 0], 0, 2))  # 0 inputs, 1 blocks, 2 outputs
    for i in np.flatnonzero(np.diff(section)) + 1:
        ax.axhline(i - 0.5, color=SURFACE, linewidth=3)
    ax.set_yticks(range(len(rows)), rows)
    ax.set_xticks(range(len(cols)), cols)
    ax.set_xlabel("network position: inputs, block 0 (closest to the inputs) ... block 31, outputs", color=MUTED, fontsize=8)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    bar = ax.figure.colorbar(image, ax=ax, fraction=0.02, pad=0.01)
    bar.ax.tick_params(colors=MUTED, labelsize=8)
    bar.outline.set_visible(False)


def _mean_pose_corr(h: dict, name: str) -> float | None:
    xyz, rot = h.get(f"eval_{name}_corr_xyz"), h.get(f"eval_{name}_corr_rot")
    return None if xyz is None or rot is None else (xyz + rot) / 2


def render(history: list[dict], max_steps: int, seq_per_step: int, path: str, updates: list[dict] | None = None) -> None:
    rows = [h for h in history if "loss" in h and "step" in h]
    if not rows:
        return
    evals = [h for h in history if "eval_loss" in h]
    # Figure API, not pyplot: safe to render from the callback's background thread.
    fig = Figure(figsize=(16, 25), facecolor=SURFACE, layout="constrained")
    # Rows 3-6 are one row of charts, split in four so the update-size column holds four small panels (a nested
    # gridspec would get constrained layout's full padding between each). Row 7 is the heatmap.
    grid = fig.add_gridspec(8, 4, height_ratios=[1, 1, 1, 0.35, 0.35, 0.35, 0.35, 2.0])
    axes = np.array([[fig.add_subplot(grid[r, c] if r < 3 else grid[3:7, c]) if (r, c) != (3, 3) else None
                      for c in range(4)] for r in range(4)])  # fmt: skip
    for ax, (metric, baseline, title) in zip(axes[2], VAL_PANELS):
        _validation_panel(ax, evals, metric, baseline, title)
    _dataset_panel(axes[3, 0], rows, lambda h, d: h.get(f"train_{d}_action"), "Train action loss by dataset",
                   smoothed=True)  # fmt: skip
    _dataset_panel(axes[3, 1], evals, _mean_pose_corr, "Val action correlation (6 pose axes)",
                   reference=0.0)  # fmt: skip
    _dataset_panel(axes[3, 2], evals, lambda h, d: h.get(f"eval_{d}_spread"),
                   "Val prediction spread ÷ true (1 = unshrunk)", reference=1.0)  # fmt: skip
    _update_series_panels([fig.add_subplot(grid[3 + i, 3]) for i in range(len(SERIES_PANELS))], updates or [])
    _update_heatmap(fig.add_subplot(grid[7, :]), updates or [])
    for ax, (key, title, log_y) in zip(axes[:2].flat, PANELS):
        pts = np.array([(h["step"], h[key]) for h in rows if key in h], dtype=float)
        _style(ax, title)
        if len(pts) == 0:
            _empty(ax)
            continue
        if key == "learning_rate":
            ax.plot(pts[:, 0], pts[:, 1], color=SERIES, linewidth=2)
        else:  # raw values faint, smoothed trend solid: one series, so no legend
            ax.plot(pts[:, 0], pts[:, 1], color=SERIES, linewidth=1, alpha=0.3)
            ax.plot(pts[:, 0], smooth(pts[:, 1]), color=SERIES, linewidth=2)
        if key == "grad_norm":
            ax.axhline(1.0, color=MUTED, linestyle="--", linewidth=1)
            ax.annotate("clip", (pts[-1, 0], 1.0), xytext=(0, 3), textcoords="offset points", ha="right", color=MUTED, fontsize=8)
        if log_y and (pts[:, 1] > 0).all():
            ax.set_yscale("log")
            plain = FuncFormatter(lambda v, _: f"{v:.3g}")
            ax.yaxis.set_major_formatter(plain)
            ax.yaxis.set_minor_formatter(plain)
    status = axes[1, 3]
    status.axis("off")
    status.text(0, 1, _status(rows, max_steps, seq_per_step), va="top", ha="left", color=INK_2, fontsize=11, family="monospace")
    fig.suptitle("Sawyer pretraining", x=0.01, ha="left", color=INK, fontsize=13, weight="bold")
    tmp = f"{path}.tmp.png"
    fig.savefig(tmp, dpi=100, facecolor=SURFACE)
    os.replace(tmp, path)  # viewers never see a half-written image


class DashboardCallback(TrainerCallback):
    """Redraws the dashboard after a log, at most every min_interval seconds, in a background thread so drawing
    never stalls training. Validation results are always drawn as soon as they are logged."""

    def __init__(self, path: str, seq_per_step: int, min_interval: float = 30.0, updates=None):
        self.path = path
        self.seq_per_step = seq_per_step
        self.min_interval = min_interval
        self.updates = updates  # an UpdateTracker, for the weight-update charts
        self._last = 0.0
        self._thread = None

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero or (self._thread is not None and self._thread.is_alive()):
            return
        is_eval = logs is not None and "eval_loss" in logs
        if not is_eval and time.time() - self._last < self.min_interval:
            return
        self._last = time.time()
        history = [dict(h) for h in state.log_history]  # snapshot: training keeps appending
        updates = list(self.updates.records) if self.updates is not None else None
        self._thread = threading.Thread(target=render, args=(history, state.max_steps, self.seq_per_step, self.path, updates),
                                        daemon=True)  # fmt: skip
        self._thread.start()
