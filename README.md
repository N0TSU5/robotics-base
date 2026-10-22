# Sawyer

Pretraining code for **Sawyer**, a robot policy that maps camera frames, robot state and a language instruction
to a chunk of 8 end-effector actions (1.6 s at 5 Hz).

- **Encoders (frozen):** V-JEPA 2.1 ViT-L encodes camera frames, and Qwen3-Embedding-0.6B encodes the instruction.
- **Predictor (trained):** a 684M-parameter block-causal transformer, warm-started from parts of RoboJEPA-300M.
  Besides the actions, it predicts the next frame's V-JEPA features and the next robot state, but only as
  training losses.
- **Training data:** DROID, Bridge, kuka and Berkeley's UR5 dataset, streamed from the Hugging Face Hub.

This README covers rebuilding the environment, checking that it works, and launching training.

## Requirements

| | Tested with | Minimum |
|---|---|---|
| GPU | NVIDIA L40S, 48 GB | 48 GB (batch 16 peaks at ~23 GB; evaluation and compilation need headroom) |
| NVIDIA driver | 615.71 (CUDA 13.4) | one that supports CUDA 13.0, which the PyTorch wheels are built against |
| CPU / RAM | 8 cores / 32 GB | the data loader runs 14 worker processes and peaks at ~13 GB |
| Disk | — | ~100 GB free: 7 GB weights, 30 GB video cache, up to 23 GB of checkpoints while saving |
| Network | ~19 MB/s to huggingface.co | training streams ~10 MB/s of video |
| OS / Python | Linux, Python 3.12.3 | Python ≥ 3.12 |
| Tools | [uv](https://docs.astral.sh/uv/) 0.12, git | |

You also need a Hugging Face account. All four datasets stream from the Hub, and logged-in users get a higher
request rate limit.

## 1. Get the code

```bash
git clone --recurse-submodules <this repo> robotics-base
cd robotics-base
```

`third_party/vjepa2` and `third_party/robo_jepa` are git submodules pinned to the commits this code was built
against. The V-JEPA encoder is imported straight from `third_party/vjepa2`, so the submodules are required.
If you cloned without `--recurse-submodules`, run `git submodule update --init`.

## 2. Create the Python environment

```bash
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[test]"
```

`pyproject.toml` pins every dependency to the version used for development and benchmarking: torch 2.14.1
(CUDA 13.0 wheels), transformers 5.20.2 and huggingface-hub 2.2.0. Newer versions may work, but the throughput
numbers below were measured with these.

Check that PyTorch sees the GPU:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
# 2.14.1+cu130 13.0 True
```

## 3. Log in to Hugging Face

```bash
hf auth login
```

## 4. Download the model weights (~7 GB)

The weights aren't in the repository. They go in `checkpoints/`, which is git-ignored:

```bash
mkdir -p checkpoints/vjepa2_1 checkpoints/robojepa

# V-JEPA 2.1 ViT-L/16, 384 px (4.9 GB; the file also holds the predictor and optimizer state, only the encoder is used)
curl -L -o checkpoints/vjepa2_1/vjepa2_1_vitl_dist_vitG_384.pt \
    https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt

# RoboJEPA-300M world model (1.2 GB; only its state encoder and camera/robot tokens are copied)
curl -L -o checkpoints/robojepa/cooldown_300m_300k.pth.tar \
    https://dl.fbaipublicfiles.com/robojepa/wm/wm-300m/cooldown_300m_300k.pth.tar

# Qwen3-Embedding-0.6B (1.2 GB)
hf download Qwen/Qwen3-Embedding-0.6B --local-dir checkpoints/qwen3-embedding-0.6b
```

The code expects exactly these paths: see `sawyer/encoders.py` and `ROBOJEPA_CKPT` in `scripts/train.py`.

## 5. Check the installation

```bash
python -m pytest -q tests/
# 64 passed
```

The tests use the small sample episodes in `sample_data/` and need no network. Without the weights from step 4,
or without a GPU, the tests that load the real encoders or compare the compiled GPU path are **skipped** rather
than failed. If you see skips, recheck steps 2 and 4.

## 6. Data

Nothing needs to be downloaded in advance. Training streams these LeRobot-format ports from the Hub:

| Dataset | Hub repo | Robot | Share of samples |
|---|---|---|---|
| DROID | `IPEC-COMMUNITY/droid_lerobot` | Franka | 0.56 |
| Bridge | `IPEC-COMMUNITY/bridge_orig_lerobot` | WidowX | 0.36 |
| kuka | `IPEC-COMMUNITY/kuka_lerobot` | Kuka iiwa | 0.05 |
| UR5 | `IPEC-COMMUNITY/berkeley_autolab_ur5_lerobot` | UR5 | 0.03 |

On first use, each dataset's episode list (`meta/episodes.jsonl`, ~5 MB in total) is downloaded to `data/meta/`.
Episode files go to an LRU cache in `data/cache/`, capped by `--cache-gb`. The fixed validation episodes use a
separate cache in `data/val_cache/`. Every 50th episode (2%) is held out for validation, picked by a fixed hash,
so the split is identical on every machine.

`data/action_stats.json` (committed) holds the per-dataset 1st/99th action percentiles used to normalize actions.
Deployment also needs it, to convert the model's output back into metres and radians. To recompute it (about
6,900 small downloads):

```bash
python scripts/compute_action_stats.py
```

## 7. Train

**Smoke test**, about 10 minutes: checks streaming, both encoders, compilation, checkpointing and validation.

```bash
python scripts/train.py --output runs/check --max-steps 20 --batch 16 --accum 1 \
    --warmup 5 --logging-steps 5 --save-steps 10 --val-windows 16 --workers 14
```

You should see losses printed every 5 steps and per-dataset `eval_*` metrics at steps 10 and 20. The first
step takes about 30 s longer while the transformer blocks compile.

**Full run**: 60,437 optimizer steps (3.9M sequences, 20 tokens per parameter). That's about 9 days on one
L40S.

```bash
mkdir -p runs
python scripts/train.py --output runs/sawyer --workers 14 > runs/sawyer.log 2>&1
```

The defaults are:
- Batch 16 with 4 gradient-accumulation steps, so 64 sequences per optimizer step.
- AdamW with β = (0.9, 0.95), learning rate peaking at 1.5e-4 after 2,000 warmup steps, then cosine decay to 1.5e-5.
- Weight decay 0.1, gradient clipping at 1.0, bf16.
- A checkpoint plus validation every 250 steps (about 50 minutes). The last 2 checkpoints are kept, about 7.7 GB each.

Run `python scripts/train.py --help` for every option.

**Resume** after an interruption. The data stream is reseeded, so no batches are replayed:

```bash
python scripts/train.py --output runs/sawyer --workers 14 --resume latest >> runs/sawyer.log 2>&1
```

**Monitoring.** `runs/<output>/dashboard.png` is redrawn at every log step, so open it in any image viewer.
It shows:
- The training losses, gradient norm, learning rate and throughput.
- A status panel with the time remaining.
- Per-dataset validation curves. Three of them are divided by a trivial baseline (copy the frame, copy the state,
  don't move), so values below 1 mean the model beats it.

`runs/<output>/checkpoint-*/trainer_state.json` holds the full log history.

## Repository layout

| Path | Contents |
|---|---|
| `sawyer/model.py` | the predictor (`SawyerPredictor`), its config, `predict_actions` for deployment, and the losses |
| `sawyer/modules.py`, `sawyer/rope.py` | the transformer block, 4D RoPE, block-causal masks (SDPA and flex_attention), and text context |
| `sawyer/inputs.py` | camera adapters, the per-robot state encoder, and the RoboJEPA warm start |
| `sawyer/encoders.py` | the frozen V-JEPA and Qwen encoders, and the tokenizer |
| `sawyer/training.py` | the training wrapper, collator, weight-decay groups, validation baselines, and the Trainer subclass |
| `sawyer/dashboard.py` | the PNG training dashboard |
| `sawyer/data/` | dataset registry and episode index, Hub cache, window loader, state and action conversion, view selection, normalization |
| `scripts/train.py` | the training entry point |
| `scripts/compute_action_stats.py` | rebuilds `data/action_stats.json` |
| `tests/` | unit and integration tests |
| `sample_data/` | a few real episodes used by the tests |

## Known issues

- **Hub rate limit.** At full speed the loader makes ~5 Hub requests per second. Earlier, without episode reuse,
  it hit HTTP 429 ("Too Many Requests"). `huggingface_hub` waits and retries, so no data is lost, but throughput
  dips. Fewer `--workers` lowers the request rate, at the cost of speed.
- **Disk.** Peak use is roughly `--cache-gb` + 3 × 7.7 GB (a new checkpoint is written before the oldest is
  deleted). Lower `--cache-gb` if space is tight.
- **Throughput** on one L40S is ~4.5–5.3 sequences/s end to end. Single-camera batches (kuka, Bridge episodes
  with one camera) run about twice as fast as two-camera ones.

## Licenses

This repository's own code is separate from the third-party components it uses, which keep their own licenses:

- **RoboJEPA:** the code in `third_party/robo_jepa` and the checkpoint are **CC BY-NC-SA 4.0** (non-commercial,
  share-alike). Sawyer copies weights from that checkpoint, so check what that means for your use before
  distributing trained models.
- **V-JEPA 2:** the repository is MIT-licensed. Its README doesn't state a separate license for the 2.1 weights.
- **Qwen3-Embedding-0.6B:** Apache 2.0.
- **Datasets:** the dataset ports carry their own licenses. Check each Hub dataset card.
