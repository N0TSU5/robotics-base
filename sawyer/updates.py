"""How much each part of the predictor is changing: relative weight-update sizes read from the AdamW state.

For a parameter w, AdamW's step is lr * (m_hat / (sqrt(v_hat) + eps) + weight_decay * w). Its norm divided by
||w|| is the fraction of the weights rewritten per step. Every trainable parameter is counted, aggregated per
heatmap cell (part x network position) and per chart series. This needs no extra forward/backward pass and no copy of the weights.
"""

import json
from collections import defaultdict
from pathlib import Path

import torch
from transformers import TrainerCallback

# Every trainable parameter -> (heatmap row, update-chart series), matched by name prefix: block parameters
# without their "blocks.<i>." prefix, the rest by their full name. Anything unmatched lands in "other", so a
# newly added parameter shows up instead of silently disappearing from the charts.
INPUT_PARTS = {
    "visual_in.adapters": ("camera adapters", "visual in"),
    "visual_in.camera_tokens": ("camera tokens", "visual in"),
    "visual_in.embodiment_tokens": ("robot tokens", "visual in"),
    "state_in": ("state encoder", "state encoder"),
    "text_in.proj": ("text projection", "text in"),
    "text_in.norm": ("text norm", "text in"),
    "text_in.null": ("text null token", "text in"),
}
BLOCK_PARTS = {
    "attn.qkv": ("self-attn qkv", "self-attn"),
    "attn.proj": ("self-attn out", "self-attn"),
    "attn.q_norm": ("self-attn QK norms", "block norms"),
    "attn.k_norm": ("self-attn QK norms", "block norms"),
    "attn.view_embed": ("camera bias table", "camera bias"),
    "attn.view_gate": ("camera bias gate", "camera bias"),
    "cross.q.": ("cross-attn q", "cross-attn"),
    "cross.kv": ("cross-attn kv", "cross-attn"),
    "cross.proj": ("cross-attn out", "cross-attn"),
    "cross.q_norm": ("cross-attn QK norms", "block norms"),
    "cross.k_norm": ("cross-attn QK norms", "block norms"),
    "mlp.gate_up": ("MLP in", "MLP"),
    "mlp.down": ("MLP out", "MLP"),
    "attn_norm": ("pre-norms", "block norms"),
    "cross_norm": ("pre-norms", "block norms"),
    "mlp_norm": ("pre-norms", "block norms"),
}
OUTPUT_PARTS = {
    "out_norm": ("final norm", "final norm"),
    "visual_out": ("visual head", "visual head"),
    "state_out": ("state head", "state head"),
    "action_head.0": ("action head in", "action head"),
    "action_head.2": ("action head out", "action head"),
}
OTHER = "other"
ROWS = [*dict.fromkeys(r for r, _ in INPUT_PARTS.values()), *dict.fromkeys(r for r, _ in BLOCK_PARTS.values()),
        *dict.fromkeys(r for r, _ in OUTPUT_PARTS.values()), OTHER]  # fmt: skip
# The update-size chart: one small panel per stage, at most four series each.
SERIES_PANELS = {
    "inputs": ["visual in", "state encoder", "text in"],
    "transformer blocks": ["self-attn", "cross-attn", "camera bias", "MLP"],
    "norms": ["block norms", "final norm"],
    "output heads": ["visual head", "state head", "action head"],
}
SERIES = [s for panel in SERIES_PANELS.values() for s in panel] + [OTHER]


def columns(num_blocks: int) -> list[str]:
    """Heatmap columns in network order: inputs, every block, outputs."""
    return ["in", *map(str, range(num_blocks)), "out"]


def _match(name: str, parts: dict) -> tuple[str, str] | None:
    return next((v for k, v in parts.items() if name.startswith(k)), None)


def classify(name: str) -> tuple[str, str, str]:
    """Parameter name (without the "predictor." prefix) -> (heatmap row, heatmap column, chart series)."""
    if name.startswith("blocks."):
        _, idx, rest = name.split(".", 2)
        row, series = _match(rest, BLOCK_PARTS) or (OTHER, OTHER)
        return row, idx, series
    for column, parts in (("in", INPUT_PARTS), ("out", OUTPUT_PARTS)):
        if match := _match(name, parts):
            return match[0], column, match[1]
    return OTHER, "out", OTHER


@torch.no_grad()
def update_ratios(model: torch.nn.Module, optimizer: torch.optim.Optimizer, num_blocks: int) -> dict:
    """{"cells": {row: [ratio per column]}, "series": {series: ratio}} for the optimizer's latest step, over every
    parameter the optimizer holds. Cells that hold no parameter are NaN; rows / series with none are left out."""
    names = {id(p): n.removeprefix("predictor.") for n, p in model.named_parameters()}
    keys, upd_sq, w_sq = [], [], []
    step = None
    for group in optimizer.param_groups:
        lr, (beta1, beta2), eps, wd = group["lr"], group["betas"], group["eps"], group["weight_decay"]
        for p in group["params"]:
            state = optimizer.state.get(p)
            if not state or "exp_avg" not in state or id(p) not in names:
                continue
            if step is None:
                step = float(state["step"])  # the single GPU sync of the whole pass
            u = state["exp_avg"] / (1 - beta1**step)
            u.div_((state["exp_avg_sq"] / (1 - beta2**step)).sqrt_().add_(eps))
            if wd:
                u.add_(p, alpha=wd)
            keys.append(classify(names[id(p)]))
            upd_sq.append((lr * lr) * u.float().square().sum())
            w_sq.append(p.float().square().sum())
    if not keys:
        return {}
    upd_sq, w_sq = torch.stack(upd_sq).tolist(), torch.stack(w_sq).tolist()

    cell_sums = defaultdict(lambda: [0.0, 0.0])
    series_sums = defaultdict(lambda: [0.0, 0.0])
    for (row, column, series), u, w in zip(keys, upd_sq, w_sq):
        for sums, key in ((cell_sums, (row, column)), (series_sums, series)):
            sums[key][0] += u
            sums[key][1] += w

    def ratio(u, w):
        return (u / w) ** 0.5 if w > 0 else float("nan")

    cols = columns(num_blocks)
    rows = {r for r, _ in cell_sums}
    return {
        "cells": {r: [ratio(*cell_sums[r, c]) if (r, c) in cell_sums else float("nan") for c in cols]
                  for r in ROWS if r in rows},  # fmt: skip
        "series": {s: ratio(*series_sums[s]) for s in SERIES if s in series_sums},
    }


class UpdateTracker(TrainerCallback):
    """Every `every` optimizer steps, records update_ratios() to <path> (one JSON line per record, so the history
    survives restarts) and keeps it in memory for the dashboard."""

    def __init__(self, path: str, num_blocks: int, every: int = 50):
        self.path = Path(path)
        self.num_blocks = num_blocks
        self.every = every
        self.records = []
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    self._add(json.loads(line))

    def _add(self, record: dict) -> None:
        # A resume replays steps after its checkpoint: the newer record replaces everything from its step on.
        self.records = [r for r in self.records if r["step"] < record["step"]] + [record]

    def on_step_end(self, args, state, control, model=None, optimizer=None, **kwargs):
        if not state.is_world_process_zero or state.global_step % self.every or model is None or optimizer is None:
            return
        record = {"step": state.global_step, **update_ratios(model, optimizer, self.num_blocks)}
        self._add(record)
        with open(self.path, "a") as f:
            f.write(json.dumps(record) + "\n")
