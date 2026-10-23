"""How much each part of the predictor is changing: relative weight-update sizes read from the AdamW state.

For a parameter w, AdamW's step is lr * (m_hat / (sqrt(v_hat) + eps) + weight_decay * w). Its norm divided by
||w|| is the fraction of the weights rewritten per step, aggregated per (block, component) and per component group.
This needs no extra forward/backward pass and no copy of the weights.
"""

import json
from collections import defaultdict
from pathlib import Path

import torch
from transformers import TrainerCallback

# Within-block parameter -> heatmap row. Everything else in a block (RMSNorm / QK-norm weights) is "norms".
BLOCK_COMPONENTS = {
    "attn.qkv.weight": "self-attn qkv",
    "attn.proj.weight": "self-attn out",
    "attn.view_embed.weight": "camera bias",
    "attn.view_gate": "camera bias",
    "cross.q.weight": "cross-attn q",
    "cross.kv.weight": "cross-attn kv",
    "cross.proj.weight": "cross-attn out",
    "mlp.gate_up.weight": "MLP in",
    "mlp.down.weight": "MLP out",
}
COMPONENT_ROWS = ["self-attn qkv", "self-attn out", "camera bias", "cross-attn q", "cross-attn kv", "cross-attn out",
                  "MLP in", "MLP out", "norms"]  # fmt: skip
GROUPS = ["inputs", "attention", "MLP", "heads"]
HEADS = ("visual_out", "state_out", "action_head")


def classify(name: str) -> tuple[int | None, str, str | None]:
    """Parameter name (without the "predictor." prefix) -> (block index or None, component, group or None)."""
    if name.startswith("blocks."):
        _, idx, rest = name.split(".", 2)
        component = BLOCK_COMPONENTS.get(rest, "norms")
        group = "MLP" if component.startswith("MLP") else None if component == "norms" else "attention"
        return int(idx), component, group
    if name.startswith(HEADS):
        return None, name.split(".")[0], "heads"
    if name.startswith(("visual_in", "state_in", "text_in")):
        return None, name.split(".")[0], "inputs"
    return None, name.split(".")[0], None  # out_norm


@torch.no_grad()
def update_ratios(model: torch.nn.Module, optimizer: torch.optim.Optimizer, num_blocks: int) -> dict:
    """{"blocks": {component: [ratio per block]}, "groups": {group: ratio}} for the optimizer's latest step."""
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

    block_sums = defaultdict(lambda: [0.0, 0.0])
    group_sums = defaultdict(lambda: [0.0, 0.0])
    for (block, component, group), u, w in zip(keys, upd_sq, w_sq):
        if block is not None:
            block_sums[component, block][0] += u
            block_sums[component, block][1] += w
        if group is not None:
            group_sums[group][0] += u
            group_sums[group][1] += w

    def ratio(u, w):
        return (u / w) ** 0.5 if w > 0 else float("nan")

    return {
        "blocks": {c: [ratio(*block_sums[c, b]) if (c, b) in block_sums else float("nan") for b in range(num_blocks)]
                   for c in COMPONENT_ROWS},  # fmt: skip
        "groups": {g: ratio(*group_sums[g]) for g in GROUPS if g in group_sums},
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
