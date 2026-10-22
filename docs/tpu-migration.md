# TPU v6e migration: status and handoff

Work in progress on branch `tpu-v6e`: porting Sawyer pretraining from one L40S to a v6e-4 TPU VM (single host,
4 chips, 32 GiB HBM each), with checkpoints saved to GCS every 15 minutes because the VM is spot.
Last updated 2026-10-22.

## Decisions so far

- **Stack: TorchTPU (`torch_tpu==0.2.0`), not PyTorch/XLA.** PyTorch/XLA is deprecated (its README says it will be
  archived on 2026-11-15). Its last release, 2.9.0, only supports torch 2.9, while TorchTPU supports
  torch 2.11 to 2.15, so the torch 2.14.1 pin stays. TorchTPU is a native PrivateUse1 backend: `device="tpu"`,
  `torch.compile(backend="tpu")`, one process per chip, and DDP over the `tpu_dist` process group.
- **Replace the HF Trainer with an explicit training loop.** accelerate 1.15.0 (the latest) has no TorchTPU support.
  Under `torchrun` it falls back to `DistributedType.NO` with a CPU default device, so the Trainer would neither wrap
  the model in DDP nor shard evaluation, and autocast would target the wrong device. The upstream PR
  (huggingface/accelerate#4278) was closed unmerged and only covered device maps anyway. The new loop must keep the
  existing behaviour: AdamW (0.9, 0.95) with the same decay groups, `get_cosine_with_min_lr_schedule_with_warmup`,
  gradient clipping at 1.0, bf16 autocast, the same log keys (so `dashboard.render` keeps working), per-dataset
  evaluation, and the same checkpoint layout (`checkpoint-N/model.safetensors` with `predictor.*` keys).
- **Attention on TPU: SDPA with the dense boolean block-causal mask.** TorchTPU patches `flex_attention` to build a
  dense additive bias from the BlockMask and call SDPA anyway, so a BlockMask gains nothing. TorchTPU lowers SDPA with
  a mask to its flash-attention kernel. `torch.tpu.ops.splash_sdpa` only does causal, sliding-window or full masks
  (`segment_ids` is hard-coded to None), so it can't express block-causal. A custom JAX splash kernel with a static
  block-causal `NumpyMask` plus segment ids (via `torch.tpu.pallas.jax_op` and a hand-registered backward) is the
  upgrade path if attention turns out to dominate.

## Done (committed)

- `sawyer/encoders.py`: V-JEPA LayerNorms run in fp32, as CUDA autocast does. TorchTPU's `layer_norm` fails on a
  bf16 input with fp32 weights ("a required argument is in a failed state"). `rms_norm` with mixed dtypes works.
- `sawyer/inputs.py`: `VisualInput` gathers each view's adapter weight instead of looping over `camera.unique()` with
  boolean-mask writes, so shapes are static and there is no host sync. Parameter names are unchanged.
- `sawyer/model.py`: `compile_blocks()` picks the backend from the parameters' device: `tpu` with the SDPA mask path,
  or inductor with flex_attention on CUDA.
- The CPU test suite still passes (61 passed, 3 GPU-only skipped).

## Measured on the v6e-4

| What | Result |
|---|---|
| 4-rank DDP over `tpu_dist` (torchrun) | gradients averaged correctly; a gloo side group for CPU control messages works |
| V-JEPA, 288 frames of 224 px (one batch of 16 seq × 9 steps × 2 views) | eager 987 ms, compiled 442 ms (11 s compile), peak 9.7 GiB |
| Qwen3-Embedding, 16 × 64 tokens | 145 ms eager |
| Encoder parity vs CPU fp32 | V-JEPA relative L2 error 1.47% (CPU bf16 autocast: 1.36%), Qwen 2.04% (CPU bf16: 2.02%), cosine ≥ 0.9998 |
| Small predictor (dim 128, depth 2), fp32, eager and compiled | finite, matches CPU (max abs diff 0.017) |
| **Full predictor, compiled blocks, bf16 autocast, grad checkpointing, V=2, per-chip batch 4** | **loss NaN**, 1215 ms/step (3.3 seq/s), peak 28.7 GiB, 128 s first step |
| **Same at per-chip batch 16** | **loss NaN**, 5700 ms/step (2.8 seq/s), peak 31.2 GiB, 328 s first step |

For comparison, the L40S does about 5 seq/s end to end. The full-size result is the open problem: NaN plus memory near
the 32 GiB limit. Next steps for it:
1. Run full size in eager mode, and in bf16 without backward, to isolate the NaN: compiled vs eager, forward vs
   backward, attention vs the rest. Check for the (B, 1, N, N) bool mask being materialized per layer, or the
   flash kernel falling back to a math path because N = 9 × 393 = 3537 isn't a multiple of 128.
2. Inspect memory: grad checkpointing around compiled blocks, the mask, and per-layer attention intermediates.
   Padding N up to a multiple of 128 may be needed for the flash kernel.
3. Try `TORCH_TPU_DEFER_AND_FUSE=1` for the eager parts (encoders, input assembly, losses).

Benchmarks to reproduce these numbers: `scripts/tpu_bench/encoders.py` and `scripts/tpu_bench/predictor.py B [eager|compiled] [V]`.

## Still to do

1. Fix the full-size NaN and memory and reach good throughput (see above).
2. `SawyerForTraining.forward`: encode all B·T·V frames and zero the padded views. That replaces `frames[valid]`,
   which has a data-dependent shape. Training batches are grouped by view count, so this wastes nothing there.
   Update `test_padded_views_are_not_encoded` to match.
3. Training loop (`sawyer/training.py` and `scripts/train.py`): torchrun, one process per chip, DDP over `tpu_dist`.
   Per-chip batch 16 with no accumulation keeps the global batch at 64. `max_steps` must divide by the world size.
   Evaluation shards batch-sized chunks across ranks so each batch stays single-dataset, then all-reduces the sums.
   Check whether `fused=True` AdamW works on TPU; if not, use foreach or a compiled step.
4. Checkpointing (`sawyer/checkpointing.py`):
   - Save every 15 minutes (`--save-minutes`), plus at every `--save-steps` evaluation. Rank 0 decides and broadcasts
     the decision over the gloo group.
   - Each save copies the state to host RAM, then a background thread writes `checkpoint-N/` locally and uploads it to
     `gs://project-metadata-11-26/<run>/checkpoint-N/`, so training only waits for the copy to host.
   - `trainer_state.json` is uploaded last, as the completion marker, and only the last K checkpoints are kept.
   - `--resume latest` takes the newest complete checkpoint, local or GCS, downloading it if the VM is new.
   - Use `google-cloud-storage` with Application Default Credentials.
5. A launch script that exports the TorchTPU topology (`eval $(python -m torch.tpu.distributed | sed 's/^/export /')`),
   sets `TORCH_TPU_DEVICES_PER_HOST=4`, optionally sets `TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT=gs://...` so restarts
   skip recompilation, and runs `torchrun --nproc_per_node=4 scripts/train.py ... --resume latest`. Also a VM bootstrap
   script, because a preempted spot VM comes back with an empty disk.
6. Data throughput: with 4× the compute, the loader needs many more workers (the VM has 180 vCPUs). Hub requests
   scale with throughput, and the README already reports HTTP 429 at about 5 requests/s. Expose
   `--windows-per-episode`, measure, and log in to HF.
7. Update the tests (GPU-only flex tests should cover any accelerator), the README and `pyproject.toml`
   (`torch_tpu`, `google-cloud-storage`, CPU torch wheels).

## Environment on a new VM

```bash
git clone --recurse-submodules https://github.com/N0TSU5/robotics-base.git && cd robotics-base && git checkout tpu-v6e
pip install --user 'uv>=0.12,<0.13'
uv venv .venv --python 3.12 && source .venv/bin/activate
uv pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu   # CPU wheels: no CUDA libs
uv pip install -e '.[test]' && uv pip install torch_tpu==0.2.0
# weights: same commands as README step 4
python -m pytest -q tests/   # 61 passed, 3 skipped
```

The VM's service account needs write access to the bucket: `roles/storage.objectAdmin` on
`project-metadata-11-26`, plus the TPU created with `--scopes=https://www.googleapis.com/auth/cloud-platform`.
OAuth scopes can't be changed after creation. The old VM only had `devstorage.read_only`.
