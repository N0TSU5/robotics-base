"""Time the frozen encoders on one TPU chip: eager vs torch.compile(backend="tpu")."""

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[2]))
from sawyer.encoders import FrozenQwen, FrozenVJEPA, load_tokenizer  # noqa: E402

dev = torch.device("tpu")


def timed(fn, n=3):
    out = fn()
    torch.accelerator.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        out = fn()
    torch.accelerator.synchronize()
    return out, (time.perf_counter() - t) / n


t0 = time.perf_counter()
vjepa = FrozenVJEPA().to(dev)
print(f"vjepa load {time.perf_counter() - t0:.1f}s", flush=True)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 288  # 16 seqs x 9 steps x 2 views
frames = torch.randint(0, 255, (N, 224, 224, 3), dtype=torch.uint8, device=dev)

t0 = time.perf_counter()
out = vjepa(frames)
torch.accelerator.synchronize()
print(f"vjepa eager first call {time.perf_counter() - t0:.1f}s shape {tuple(out.shape)} {out.dtype}", flush=True)
ref, dt = timed(lambda: vjepa(frames))
print(f"vjepa eager steady {dt * 1e3:.0f} ms for {N} frames", flush=True)

cvjepa = torch.compile(vjepa, backend="tpu", dynamic=False)
t0 = time.perf_counter()
out = cvjepa(frames)
torch.accelerator.synchronize()
print(f"vjepa compiled first call {time.perf_counter() - t0:.1f}s", flush=True)
out, dt = timed(lambda: cvjepa(frames))
print(f"vjepa compiled steady {dt * 1e3:.0f} ms; max |compiled - eager| {(out.float() - ref.float()).abs().max().item():.4f}, "
      f"ref scale {ref.float().abs().mean().item():.4f}", flush=True)
print("mem", torch.accelerator.max_memory_allocated() / 2**30, "GiB", flush=True)

tok = load_tokenizer()
qwen = FrozenQwen().to(dev)
t = tok(["put the cup in the bowl"] * 16, padding="max_length", truncation=True, max_length=64, return_tensors="pt")
ids, mask = t["input_ids"].to(dev), t["attention_mask"].bool().to(dev)
t0 = time.perf_counter()
h = qwen(ids, mask)
torch.accelerator.synchronize()
print(f"qwen eager first {time.perf_counter() - t0:.1f}s shape {tuple(h.shape)} {h.dtype}", flush=True)
h, dt = timed(lambda: qwen(ids, mask))
print(f"qwen eager steady {dt * 1e3:.0f} ms", flush=True)
qwen_cpu = FrozenQwen()
h_cpu = qwen_cpu(t["input_ids"], t["attention_mask"].bool())
print(f"qwen tpu vs cpu: max abs diff {(h.cpu().float() - h_cpu.float()).abs().max().item():.3f}, scale {h_cpu.float().abs().mean().item():.3f}", flush=True)
