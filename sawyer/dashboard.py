"""Training dashboard: a PNG of the training curves, redrawn at every log, viewable without a TensorBoard server."""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402
from transformers import TrainerCallback  # noqa: E402

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


def smooth(values: np.ndarray) -> np.ndarray:
    """Debiased EMA spanning ~5% of the points (at most 50 logs), so short runs are barely smoothed and the
    trend never lags behind from a cold start."""
    beta = 1 - 1 / np.clip(len(values) / 20, 1, 50)
    out, acc = np.empty_like(values), 0.0
    for i, v in enumerate(values):
        acc = beta * acc + (1 - beta) * v
        out[i] = acc / (1 - beta ** (i + 1)) if beta > 0 else v
    return out


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


def render(history: list[dict], max_steps: int, seq_per_step: int, path: str) -> None:
    rows = [h for h in history if "loss" in h and "step" in h]
    if not rows:
        return
    evals = [h for h in history if "eval_loss" in h]
    fig, axes = plt.subplots(3, 4, figsize=(16, 11), facecolor=SURFACE, layout="constrained")
    for ax, (metric, baseline, title) in zip(axes[2], VAL_PANELS):
        _validation_panel(ax, evals, metric, baseline, title)
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
    plt.close(fig)
    os.replace(tmp, path)  # viewers never see a half-written image


class DashboardCallback(TrainerCallback):
    def __init__(self, path: str, seq_per_step: int):
        self.path = path
        self.seq_per_step = seq_per_step

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_world_process_zero:
            render(state.log_history, state.max_steps, self.seq_per_step, self.path)
