"""Which inputs does the action policy use? Evaluates a checkpoint on the fixed validation windows with one input
swapped at a time, next to trivial baselines.

Ablations (the chunk L1 loss divided by the unablated model's, on the same samples; 1.0 = input unused):
  images: every step's frames come from another window of the same dataset (the batch rolled by one)
  text: the instruction replaced by the null token (only samples that had an instruction)
  proprio: every step's robot state comes from another window of the same dataset

Baselines (the model's loss divided by the baseline's; below 1.0 = the model beats it):
  zero motion, and constant velocity: the chunk at step t repeats the last observed delta (step t - 1 -> t), so
  it exists only for t >= 1. Step 0 has no history at all, which is what deployment (T = 1) sees.

  python scripts/probe_policy.py --checkpoint runs/sawyer/checkpoint-5250
"""

import argparse
from collections import defaultdict

import numpy as np
import torch
from safetensors.torch import load_file
from torch.utils.data import DataLoader
from transformers.trainer_utils import get_last_checkpoint

from sawyer.data.index import DATASETS, load_index, validation_windows
from sawyer.data.loader import FixedWindows
from sawyer.data.normalize import load_stats
from sawyer.encoders import FrozenQwen, FrozenVJEPA, load_tokenizer
from sawyer.model import SawyerConfig
from sawyer.training import SawyerCollator, SawyerForTraining

def chunk_error(pred, target, valid):
    """Per-sample, per-step L1 sum and count over valid chunk entries: (B, T) each."""
    err = (pred.float() - target.float()).abs().mean(-1) * valid
    return err.sum(-1), valid.sum(-1).float()


def roll(x):
    return torch.roll(x, 1, dims=0)


@torch.no_grad()
def probe_batch(model, batch):
    p = model.predictor
    frames, camera, view_valid = batch["frames"], batch["camera"], batch["view_valid"]
    B, T, V = frames.shape[:3]
    enc = model.vjepa(frames.flatten(0, 2)).unflatten(0, (B, T, V))
    feats = torch.where(view_valid[:, None, :, None, None], enc, 0)
    text = model.qwen(batch["input_ids"], batch["text_mask"])
    target, valid = batch["action_target"], batch["action_valid"]

    def actions(feats=feats, camera=camera, view_valid=view_valid, state=batch["state"], has_text=batch["has_text"]):
        h = p._trunk(feats, model.grid, camera, view_valid, state, batch["embodiment"], text, batch["text_mask"], has_text)
        return p._action_chunk(h[:, :, -1])

    out = {
        "model": actions(),
        "images": actions(feats=roll(feats), camera=roll(camera), view_valid=roll(view_valid)),
        "text": actions(has_text=torch.zeros_like(batch["has_text"])),
        "proprio": actions(state=roll(batch["state"])),
        "zero": batch["action_zero"][:, None, None].expand_as(target),
    }
    # Constant velocity: the step t - 1 -> t delta (action index t - 1 = chunk (t - 1, 0)) for every k; t = 0 has none.
    cv = torch.cat([torch.zeros_like(target[:, :1]), target[:, :-1, :1].expand(-1, -1, target.shape[2], -1)], 1)
    out["const_velocity"] = cv
    return {k: chunk_error(v, target, valid) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None, help="default: latest in runs/sawyer")
    ap.add_argument("--val-windows", type=int, default=96)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    ckpt = args.checkpoint or get_last_checkpoint("runs/sawyer")

    model = SawyerForTraining(SawyerConfig(), FrozenVJEPA().cuda(), FrozenQwen().cuda())
    model.load_state_dict(load_file(f"{ckpt}/model.safetensors"))
    model.cuda().eval()

    stats = load_stats("data/action_stats.json")
    indexes = [load_index(c, "data/meta", split="val") for c in DATASETS]
    data = FixedWindows(indexes, stats, validation_windows(indexes, args.val_windows), "data/val_cache", 5e9)
    loader = DataLoader(data, batch_size=args.batch, num_workers=args.workers, collate_fn=SawyerCollator(load_tokenizer()))

    # sums[dataset, variant, steps]: steps "0" (deployment, no history) or "1-8"; the "text" suffix keeps only
    # samples that had an instruction, where the text ablation is meaningful.
    sums = defaultdict(lambda: torch.zeros(2))
    for i, batch in enumerate(loader):
        names = batch.pop("dataset")
        assert len(set(names)) == 1, "validation batches must hold a single dataset"
        batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            errs = probe_batch(model, batch)
        has_text = batch["has_text"].float()[:, None]
        for k, (s, c) in errs.items():
            for steps, sl in (("0", slice(0, 1)), ("1-8", slice(1, None))):
                for suffix, w in (("", 1.0), (" text", has_text)):
                    sums[names[0], k, steps + suffix] += torch.stack([(s[:, sl] * w).sum(), (c[:, sl] * w).sum()]).cpu()
        print(f"\rbatch {i + 1}/{len(loader)}", end="", flush=True)
    print(f"\n\n{ckpt}, {args.val_windows} validation windows per dataset. Action chunk L1, normalized units.\n")

    def loss(d, k, steps):
        s, c = sums[d, k, steps]
        return (s / c).item() if c > 0 else float("nan")

    print("Baselines: model loss / baseline loss (below 1 = model better)")
    print(f"{'':8} | {'step 0':^23} | {'steps 1-8':^41}")
    print(f"{'dataset':8} | {'model':>7} {'zero':>7} {'÷zero':>7} | {'model':>7} {'zero':>7} {'const-v':>7} {'÷zero':>7} {'÷const-v':>9}")
    for c in DATASETS:
        d = c.name
        m0, z0 = loss(d, "model", "0"), loss(d, "zero", "0")
        m, z, cv = loss(d, "model", "1-8"), loss(d, "zero", "1-8"), loss(d, "const_velocity", "1-8")
        print(f"{d:8} | {m0:7.4f} {z0:7.4f} {m0 / z0:7.3f} | {m:7.4f} {z:7.4f} {cv:7.4f} {m / z:7.3f} {m / cv:9.3f}")

    print("\nAblations: loss with the input swapped / model loss (1.0 = input unused)")
    print(f"{'':8} | {'step 0':^26} | {'steps 1-8':^26}")
    print(f"{'dataset':8} | {'images':>8} {'text':>8} {'proprio':>8} | {'images':>8} {'text':>8} {'proprio':>8}")
    for c in DATASETS:
        d = c.name
        row = []
        for steps in ("0", "1-8"):
            row += [loss(d, "images", steps) / loss(d, "model", steps),
                    loss(d, "text", steps + " text") / loss(d, "model", steps + " text"),
                    loss(d, "proprio", steps) / loss(d, "model", steps)]  # fmt: skip
        print(f"{d:8} | {row[0]:8.3f} {row[1]:8.3f} {row[2]:8.3f} | {row[3]:8.3f} {row[4]:8.3f} {row[5]:8.3f}")

if __name__ == "__main__":
    main()
