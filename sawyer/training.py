"""Training wrapper, collator, optimizer groups, and the Trainer subclass."""

import time
from collections import defaultdict

import numpy as np
import torch
from torch import nn
from transformers import Trainer

from sawyer.model import SawyerConfig, SawyerPredictor, sawyer_loss

ARRAY_KEYS = ("frames", "camera", "view_valid", "state", "state_target", "action_target", "action_valid", "action_zero")
# Parameters kept out of weight decay besides biases and norms (ndim < 2): the RoboJEPA warm start (state encoder,
# camera / robot tokens), whose small learned scale decay would erode, and scale-free embeddings and gates.
NO_DECAY = ("state_in.weight", "state_in.bias", "visual_in.camera_tokens", "visual_in.embodiment_tokens", "text_in.null")
NO_DECAY_SUFFIXES = ("view_embed.weight", "view_gate")


class SawyerCollator:
    def __init__(self, tokenizer, max_text_tokens: int = 64):
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
                input_ids, text_mask, has_text, action_zero=None):  # fmt: skip
        B, T, V = frames.shape[:3]
        # Every slot is encoded: no boolean indexing (a GPU sync) and one encoder shape per view count. Training
        # batches have no padded slots (WindowDataset groups by view count); padded eval slots are zeroed.
        enc = self.vjepa(frames.flatten(0, 2)).unflatten(0, (B, T, V))
        feats = torch.where(view_valid[:, None, :, None, None], enc, 0)
        text = self.qwen(input_ids, text_mask)
        outputs = self.predictor(feats, self.grid, camera, view_valid, state, embodiment, text, text_mask, has_text,
                                 state_now=state_target)  # fmt: skip
        losses = sawyer_loss(outputs, feats, view_valid, state_target, action_target, action_valid,
                             self.predictor.cfg.loss_weights)  # fmt: skip
        if not self.training and action_zero is not None:
            losses.update(baselines(feats, view_valid, state_target, action_target, action_valid, action_zero))
        return losses


def baselines(feats, view_valid, state_target, action_target, action_valid, action_zero) -> dict:
    """Trivial predictors scored with the same L1 losses: the bar a useful model has to clear.
    visual_copy: next frame = this frame; state_copy: next state = this state; action_zero: no motion."""
    err = (feats[:, :-1].float() - feats[:, 1:].float()).abs().mean(-1)
    mask = view_valid[:, None, :, None].expand_as(err)
    act_err = (action_zero[:, None, None].float() - action_target.float()).abs().mean(-1) * action_valid
    return {
        "visual_copy": (err * mask).sum() / mask.sum(),
        "state_copy": (state_target[:, :-1] - state_target[:, 1:]).abs().mean(),
        "action_zero": act_err.sum() / action_valid.sum().clamp(min=1),
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


class SawyerTrainer(Trainer):
    """Logs the visual / state / action losses and training throughput next to the total loss."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._parts = {"visual": 0.0, "state": 0.0, "action": 0.0}
        self._num_parts = 0
        self._samples = 0
        self._last_log = time.time()

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        inputs.pop("dataset", None)
        losses = model(**inputs)
        for k in self._parts:
            self._parts[k] += losses[k].detach().float()
        self._num_parts += 1
        self._samples += len(inputs["frames"])
        return (losses["loss"], losses) if return_outputs else losses["loss"]

    @torch.no_grad()
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """Losses and baselines per dataset on the fixed validation windows. Validation batches hold one dataset
        each (windows are listed dataset by dataset, a multiple of the batch size per dataset)."""
        model = self.model
        model.eval()
        sums, counts = defaultdict(float), defaultdict(int)
        for batch in self.get_eval_dataloader(eval_dataset):
            names = batch.pop("dataset")
            assert len(set(names)) == 1, "validation batches must hold a single dataset"
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(**self._prepare_inputs(batch))
            for k, v in out.items():
                sums[names[0], k] += v.item()
                counts[names[0], k] += 1
        model.train()
        metrics = {f"{metric_key_prefix}_{d}_{k}": round(sums[d, k] / counts[d, k], 4) for d, k in sums}
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
        super().log(logs, *args, **kwargs)
