"""Pretrain the Sawyer predictor with the HF Trainer, streaming windows from the Hub."""

import argparse
import logging
import math
import sys

import torch
from transformers import TrainingArguments
from transformers.trainer_utils import get_last_checkpoint

from sawyer.data.index import DATASETS, WINDOW_STEPS, load_index, validation_windows
from sawyer.data.loader import FixedWindows, WindowDataset
from sawyer.data.normalize import load_stats
from sawyer.dashboard import DashboardCallback
from sawyer.data.views import MAX_VIEWS
from sawyer.encoders import FrozenQwen, FrozenVJEPA, load_tokenizer
from sawyer.model import SawyerConfig
from sawyer.training import SawyerCollator, SawyerForTraining, SawyerTrainer, param_groups

ROBOJEPA_CKPT = "checkpoints/robojepa/cooldown_300m_300k.pth.tar"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="runs/sawyer")
    p.add_argument("--max-steps", type=int, default=None, help="default: 20 tokens / parameter")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=1.5e-4)
    p.add_argument("--min-lr", type=float, default=1.5e-5)
    p.add_argument("--warmup", type=int, default=2000)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--cache-gb", type=float, default=30)
    p.add_argument("--save-steps", type=int, default=250, help="checkpoint every N steps (~55 min)")
    p.add_argument("--eval-steps", type=int, default=250, help="validation every N steps (~45 min)")
    p.add_argument("--val-windows", type=int, default=32, help="held-out windows per dataset")
    p.add_argument("--logging-steps", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", default=None, help="checkpoint dir, or 'latest'")
    p.add_argument("--checkpoint-layers", type=int, default=None, help="blocks using activation checkpointing (default all)")
    p.add_argument("--compile-mode", default="max-autotune-no-cudagraphs",
                   help="torch.compile mode for the predictor blocks (autotuned: 6%% faster on the L40S, ~4 min extra compile per shape, cached)")
    p.add_argument("--no-compile", action="store_true", help="eager blocks with SDPA and eager V-JEPA (debugging)")
    args = p.parse_args()
    sys.stdout.reconfigure(line_buffering=True)  # Trainer's loss lines show up live in log files
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("httpx2").setLevel(logging.WARNING)  # one line per Hub request otherwise

    resume = get_last_checkpoint(args.output) if args.resume == "latest" else args.resume
    resume_step = int(resume.rsplit("-", 1)[1]) if resume else 0

    vjepa, qwen = FrozenVJEPA().cuda(), FrozenQwen().cuda()
    model = SawyerForTraining(SawyerConfig(checkpoint_layers=args.checkpoint_layers), vjepa, qwen, args.image_size)
    model.predictor.load_robojepa(ROBOJEPA_CKPT)
    model.cuda()
    if not args.no_compile:
        model.predictor.compile_blocks(args.compile_mode)
        vjepa.compile_encoder()
    num_params = sum(p.numel() for p in model.parameters())

    seq_tokens = WINDOW_STEPS * (MAX_VIEWS * (args.image_size // 16) ** 2 + 1)
    max_steps = args.max_steps or math.ceil(20 * num_params / seq_tokens / (args.batch * args.accum))
    logging.info("predictor %.1fM params, %d tokens/sequence, %d optimizer steps", num_params / 1e6, seq_tokens, max_steps)

    stats = load_stats("data/action_stats.json")
    # The stream is random and endless, so a resumed run draws fresh windows instead of replaying the first ones.
    dataset = WindowDataset([load_index(c, "data/meta", split="train") for c in DATASETS], stats, "data/cache",
                            args.cache_gb * 1e9, args.image_size, seed=args.seed * 10**9 + resume_step,
                            group_size=args.batch, windows_per_episode=4, shuffle_buffer=64)  # fmt: skip
    # Fixed held-out windows, a multiple of the batch size per dataset so each validation batch is one dataset.
    assert args.val_windows % args.batch == 0
    val_indexes = [load_index(c, "data/meta", split="val") for c in DATASETS]
    val_dataset = FixedWindows(val_indexes, stats, validation_windows(val_indexes, args.val_windows), "data/val_cache",
                               5e9, args.image_size)  # fmt: skip
    optimizer = torch.optim.AdamW(param_groups(model, args.weight_decay), lr=args.lr, betas=(0.9, 0.95), fused=True)
    training_args = TrainingArguments(
        output_dir=args.output,
        max_steps=max_steps,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        gradient_accumulation_steps=args.accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr": args.min_lr},
        warmup_steps=args.warmup,
        max_grad_norm=1.0,
        bf16=True,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        dataloader_num_workers=args.workers,
        dataloader_prefetch_factor=4,
        dataloader_persistent_workers=True,
        remove_unused_columns=False,
        logging_nan_inf_filter=False,  # its isnan check syncs the GPU every micro-batch, only to filter the logged loss
        accelerator_config={"non_blocking": True},  # async host-to-GPU copies (batches are already pinned)
        ignore_data_skip=True,  # don't re-read and discard every batch already trained on when resuming
        report_to="none",  # curves go to dashboard.png (DashboardCallback)
        seed=args.seed,
    )
    trainer = SawyerTrainer(model=model, args=training_args, train_dataset=dataset, eval_dataset=val_dataset,
                            data_collator=SawyerCollator(load_tokenizer()), optimizers=(optimizer, None))  # fmt: skip
    trainer.add_callback(DashboardCallback(f"{args.output}/dashboard.png", args.batch * args.accum))
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(f"{args.output}/final")


if __name__ == "__main__":
    main()
