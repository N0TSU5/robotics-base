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

from sawyer.updates import COMPONENT_ROWS, GROUPS

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
# Component groups in the update-size chart: categorical slots 5-8, so they never share a hue with a dataset.
GROUP_COLORS = dict(zip(GROUPS, ["#e87ba4", "#008300", "#6250d6", "#e34948"]))
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


def _update_groups_panel(ax, updates: list[dict]) -> None:
    _style(ax, "Weight update per step, ‖Δw‖ / ‖w‖")
    if not updates:
        _empty(ax)
        return
    for group, color in GROUP_COLORS.items():
        pts = np.array([(r["step"], r["groups"][group]) for r in updates if group in r.get("groups", {})], dtype=float)
        if len(pts) == 0:
            continue
        ax.plot(pts[:, 0], pts[:, 1], color=color, linewidth=2, label=group)
        ax.annotate(group, pts[-1], xytext=(4, 0), textcoords="offset points", va="center", color=INK_2, fontsize=8)
    ax.set_yscale("log")
    ax.legend(loc="best", fontsize=8, frameon=False, labelcolor=INK_2)


def _update_heatmap(ax, updates: list[dict]) -> None:
    """Latest relative update for every block x component: which layers and parts are learning the most right now."""
    if not updates or "blocks" not in updates[-1]:
        _style(ax, "Relative weight update per block and component")
        _empty(ax)
        return
    latest = updates[-1]
    grid = np.array([latest["blocks"][c] for c in COMPONENT_ROWS], dtype=float)
    grid = np.ma.masked_invalid(np.where(grid > 0, grid, np.nan))
    ax.set_facecolor(SURFACE)
    ax.set_title(f"Relative weight update per block and component (‖Δw‖ / ‖w‖ per step, at step {latest['step']:,}); "
                 "darker = changing faster", loc="left", color=INK, fontsize=10)  # fmt: skip
    if grid.count() == 0:
        _empty(ax)
        return
    image = ax.imshow(grid, aspect="auto", cmap=UPDATE_CMAP, norm=LogNorm(vmin=grid.min(), vmax=grid.max()),
                      interpolation="nearest")  # fmt: skip
    ax.set_yticks(range(len(COMPONENT_ROWS)), COMPONENT_ROWS)
    ax.set_xticks(range(grid.shape[1]), [str(i) for i in range(grid.shape[1])])
    ax.set_xlabel("block (0 = closest to the inputs)", color=MUTED, fontsize=8)
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
    fig = Figure(figsize=(16, 19), facecolor=SURFACE, layout="constrained")
    grid = fig.add_gridspec(5, 4, height_ratios=[1, 1, 1, 1, 0.9])
    axes = np.array([[fig.add_subplot(grid[r, c]) for c in range(4)] for r in range(4)])
    for ax, (metric, baseline, title) in zip(axes[2], VAL_PANELS):
        _validation_panel(ax, evals, metric, baseline, title)
    _dataset_panel(axes[3, 0], rows, lambda h, d: h.get(f"train_{d}_action"), "Train action loss by dataset",
                   smoothed=True)  # fmt: skip
    _dataset_panel(axes[3, 1], evals, _mean_pose_corr, "Val action correlation (6 pose axes)",
                   reference=0.0)  # fmt: skip
    _dataset_panel(axes[3, 2], evals, lambda h, d: h.get(f"eval_{d}_spread"),
                   "Val prediction spread ÷ true (1 = unshrunk)", reference=1.0)  # fmt: skip
    _update_groups_panel(axes[3, 3], updates or [])
    _update_heatmap(fig.add_subplot(grid[4, :]), updates or [])
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
