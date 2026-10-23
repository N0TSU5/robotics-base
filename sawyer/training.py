"""Training wrapper, collator, optimizer groups, and the Trainer subclass."""

import time
from collections import defaultdict

import numpy as np
import torch
from torch import nn
from transformers import Trainer

from sawyer.model import SawyerConfig, SawyerPredictor, sawyer_loss

ARRAY_KEYS = ("frames", "camera", "view_valid", "state", "state_target", "action_target", "action_valid", "action_zero",
              "state_scale")
# Parameters kept out of weight decay besides biases and norms (ndim < 2): the RoboJEPA warm start (state encoder,
# camera / robot tokens), whose small learned scale decay would erode, and scale-free embeddings and gates.
NO_DECAY = ("state_in.weight", "state_in.bias", "visual_in.camera_tokens", "visual_in.embodiment_tokens", "text_in.null")
NO_DECAY_SUFFIXES = ("view_embed.weight", "view_gate")


class SawyerCollator:
    def __init__(self, tokenizer, max_text_tokens: int = 63):  # + the null token = 64 cross-attn keys, a multiple of 8
        self.tokenizer = tokenizer
        self.max_text_tokens = max_text_tokens

    def __call__(self, samples: list[dict]) -> dict:
        batch = {k: torch.from_numpy(np.stack([s[k] for s in samples])) for k in ARRAY_KEYS}
        # A batch with no second view anywhere (grouped by WindowDataset) drops the empty slot: half the tokens.
        views = int(batch["view_valid"].sum(1).max())
        for k in ("frames", "camera", "view_valid"):
            batch[k] = batch[k][:, :, :views] if k == "frames" else batch[k][:, :views]
        batch["embodiment"] = torch.tensor([s["embodiment"] for s in samples])
        batch["has_text"] = torch.tensor([s["has_text"] for s in samples])
        batch["dataset"] = [s["dataset"] for s in samples]  # for per-dataset validation; popped before the model
        # Fixed length, so the compiled blocks see one text shape instead of recompiling per batch.
        tokens = self.tokenizer([s["instruction"] for s in samples], padding="max_length", truncation=True,
                                max_length=self.max_text_tokens, return_tensors="pt")  # fmt: skip
        batch["input_ids"] = tokens["input_ids"]
        batch["text_mask"] = tokens["attention_mask"].bool()
        return batch


class SawyerForTraining(nn.Module):
    """Frozen encoders + trainable predictor + losses, called by the Trainer with a collated batch.

    The encoders are stored outside the module tree: they are not in parameters() (optimizer, grad clipping), not
    in state_dict() (checkpoints hold only the predictor), and Trainer's model.train() never reaches them.
    """

    def __init__(self, config: SawyerConfig, vjepa: nn.Module, qwen: nn.Module, image_size: int = 224):
        super().__init__()
        self.predictor = SawyerPredictor(config)
        object.__setattr__(self, "vjepa", vjepa.eval().requires_grad_(False))
        object.__setattr__(self, "qwen", qwen.eval().requires_grad_(False))
        self.grid = (image_size // 16, image_size // 16)

    def forward(self, frames, camera, view_valid, state, state_target, action_target, action_valid, embodiment,
                input_ids, text_mask, has_text, action_zero=None, state_scale=None):  # fmt: skip
        B, T, V = frames.shape[:3]
        # Every slot is encoded: no boolean indexing (a GPU sync) and one encoder shape per view count. Training
        # batches have no padded slots (WindowDataset groups by view count); padded eval slots are zeroed.
        enc = self.vjepa(frames.flatten(0, 2)).unflatten(0, (B, T, V))
        feats = torch.where(view_valid[:, None, :, None, None], enc, 0)
        text = self.qwen(input_ids, text_mask)
        outputs = self.predictor(feats, self.grid, camera, view_valid, state, embodiment, text, text_mask, has_text,
                                 state_now=state_target)  # fmt: skip
        losses = sawyer_loss(outputs, feats, view_valid, state_target, action_target, action_valid,
                             self.predictor.cfg.loss_weights, state_scale)  # fmt: skip
        if not self.training and action_zero is not None:
            losses.update(baselines(feats, view_valid, state_target, action_target, action_valid, action_zero, state_scale))
            # Step-0 chunks (what deployment executes), for the action-agreement diagnostics.
            losses.update(_pred_actions=outputs[2][:, 0], _target_actions=action_target[:, 0], _valid_actions=action_valid[:, 0])
        return losses


def baselines(feats, view_valid, state_target, action_target, action_valid, action_zero, state_scale=None) -> dict:
    """Trivial predictors scored with the same L1 losses: the bar a useful model has to clear.
    visual_copy: next frame = this frame; state_copy: next state = this state; action_zero: no motion."""
    err = (feats[:, :-1].float() - feats[:, 1:].float()).abs().mean(-1)
    mask = view_valid[:, None, :, None].expand_as(err)
    act_err = (action_zero[:, None, None].float() - action_target.float()).abs().mean(-1) * action_valid
    state_err = (state_target[:, :-1] - state_target[:, 1:]).abs()
    if state_scale is not None:
        state_err = state_err / state_scale[:, None]
    return {
        "visual_copy": (err * mask).sum() / mask.sum(),
        "state_copy": state_err.mean(),
        "action_zero": act_err.sum() / action_valid.sum().clamp(min=1),
    }


def action_agreement(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> dict:
    """How well predicted action chunks track the true ones, beyond the L1 loss.

    pred / target (N, ACTION_HORIZON, 7), valid (N, ACTION_HORIZON). Per action axis over the valid entries:
    Pearson correlation (does the prediction move with the truth?) and the prediction's spread relative to the
    target's (below 1: predictions shrunk toward the mean). Returned as means over xyz, rotation and the gripper.
    """
    p, t = pred[valid].double(), target[valid].double()  # (M, 7)
    p_std, t_std = p.std(0, unbiased=False), t.std(0, unbiased=False)
    corr = ((p - p.mean(0)) * (t - t.mean(0))).mean(0) / (p_std * t_std).clamp(min=1e-12)
    spread = p_std / t_std.clamp(min=1e-12)
    return {
        "corr_xyz": corr[:3].mean().item(),
        "corr_rot": corr[3:6].mean().item(),
        "corr_grip": corr[6].item(),
        "spread": spread[:6].mean().item(),  # gripper excluded: its target spread is ~0 for most datasets
    }


def param_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        name = name.removeprefix("predictor.")
        if p.ndim < 2 or name in NO_DECAY or name.endswith(NO_DECAY_SUFFIXES):
            no_decay.append(p)
        else:
            decay.append(p)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


PER_SAMPLE_KEYS = ("_visual_sum", "_visual_count", "_action_sum", "_action_count")


class SawyerTrainer(Trainer):
    """Logs the visual / state / action losses, per-dataset training losses and throughput next to the total loss.

    Per-dataset sums accumulate on the GPU (index_add_ by dataset id) and are read only when logging, so they add no
    per-micro-batch GPU sync."""

    def __init__(self, *args, dataset_names: list[str] = (), **kwargs):
        super().__init__(*args, **kwargs)
        self._parts = {"visual": 0.0, "state": 0.0, "action": 0.0}
        self._num_parts = 0
        self._samples = 0
        self._last_log = time.time()
        self._dataset_names = list(dataset_names)
        self._dataset_ids = {name: i for i, name in enumerate(self._dataset_names)}
        self._per_dataset = None  # {key: (num_datasets,) tensor}, created on the loss's device

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        names = inputs.pop("dataset", None)
        losses = model(**inputs)
        for k in self._parts:
            self._parts[k] += losses[k].detach().float()
        self._num_parts += 1
        self._samples += len(inputs["frames"])
        if names is not None and self._dataset_ids:
            device = losses["loss"].device
            ids = torch.tensor([self._dataset_ids[n] for n in names]).to(device, non_blocking=True)
            if self._per_dataset is None:
                self._per_dataset = {k: torch.zeros(len(self._dataset_ids), device=device) for k in PER_SAMPLE_KEYS}
            for k in PER_SAMPLE_KEYS:
                self._per_dataset[k].index_add_(0, ids, losses[k].detach().float())
        return (losses["loss"], losses) if return_outputs else losses["loss"]

    @torch.no_grad()
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """Losses and baselines per dataset on the fixed validation windows. Validation batches hold one dataset
        each (windows are listed dataset by dataset, a multiple of the batch size per dataset)."""
        model = self.model
        model.eval()
        sums, counts = defaultdict(float), defaultdict(int)
        chunks = defaultdict(lambda: ([], [], []))  # dataset -> predicted, target, valid step-0 action chunks
        for batch in self.get_eval_dataloader(eval_dataset):
            names = batch.pop("dataset")
            assert len(set(names)) == 1, "validation batches must hold a single dataset"
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(**self._prepare_inputs(batch))
            for part, key in zip(chunks[names[0]], ("_pred_actions", "_target_actions", "_valid_actions")):
                part.append(out[key].float().cpu())
            for k, v in out.items():
                if not k.startswith("_"):
                    sums[names[0], k] += v.item()
                    counts[names[0], k] += 1
        model.train()
        metrics = {f"{metric_key_prefix}_{d}_{k}": round(sums[d, k] / counts[d, k], 4) for d, k in sums}
        for d, (pred, target, valid) in chunks.items():
            agreement = action_agreement(torch.cat(pred), torch.cat(target), torch.cat(valid).bool())
            metrics.update({f"{metric_key_prefix}_{d}_{k}": round(v, 4) for k, v in agreement.items()})
        datasets = sorted({d for d, _ in sums})
        metrics[f"{metric_key_prefix}_loss"] = round(np.mean([metrics[f"{metric_key_prefix}_{d}_loss"] for d in datasets]), 4)
        self.log(metrics)
        return metrics

    def log(self, logs, *args, **kwargs):
        if self._num_parts and "loss" in logs:
            logs.update({f"loss_{k}": round((v / self._num_parts).item(), 4) for k, v in self._parts.items()})
            now = time.time()
            logs["seq_per_sec"] = round(self._samples / (now - self._last_log), 3)
            self._parts = dict.fromkeys(self._parts, 0.0)
            self._num_parts = self._samples = 0
            self._last_log = now
            if self._per_dataset is not None:
                sums = {k: v.tolist() for k, v in self._per_dataset.items()}  # the one sync per log
                for i, name in enumerate(self._dataset_names):
                    for part in ("visual", "action"):
                        if sums[f"_{part}_count"][i] > 0:
                            logs[f"train_{name}_{part}"] = round(sums[f"_{part}_sum"][i] / sums[f"_{part}_count"][i], 4)
                for v in self._per_dataset.values():
                    v.zero_()
        super().log(logs, *args, **kwargs)
