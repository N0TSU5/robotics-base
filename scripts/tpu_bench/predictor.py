"""Full-size predictor fwd + bwd on one TPU chip. Usage: scripts/tpu_bench/predictor.py B [eager|compiled] [V]"""

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[2]))
from sawyer.inputs import CAMERAS  # noqa: E402
from sawyer.model import SawyerConfig, SawyerPredictor, sawyer_loss  # noqa: E402

B = int(sys.argv[1])
mode = sys.argv[2] if len(sys.argv) > 2 else "compiled"
V = int(sys.argv[3]) if len(sys.argv) > 3 else 2
dev = torch.device("tpu")
T, P, L = 9, 196, 64

torch.manual_seed(0)
model = SawyerPredictor(SawyerConfig()).to(dev).train()
if mode == "compiled":
    model.compile_blocks()
cams = torch.tensor([CAMERAS["left"], CAMERAS["wrist"]])[:V]
batch = dict(
    feats=torch.randn(B, T, V, P, 1024, device=dev), grid=(14, 14), camera=cams.expand(B, V).contiguous().to(dev),
    view_valid=torch.ones(B, V, dtype=torch.bool, device=dev), state=torch.randn(B, T, 64, device=dev),
    embodiment=torch.zeros(B, dtype=torch.long, device=dev), text=torch.randn(B, L, 1024, device=dev),
    text_mask=torch.ones(B, L, dtype=torch.bool, device=dev), has_text=torch.ones(B, dtype=torch.bool, device=dev),
)
state_target = torch.randn(B, T, 10, device=dev)
action_target = torch.randn(B, T, 8, 7, device=dev)
action_valid = torch.ones(B, T, 8, dtype=torch.bool, device=dev)


def step():
    with torch.autocast("tpu", dtype=torch.bfloat16):
        out = model(**batch)
        losses = sawyer_loss(out, batch["feats"], batch["view_valid"], state_target, action_target, action_valid,
                             model.cfg.loss_weights)
    losses["loss"].backward()
    return losses["loss"].detach()


t0 = time.perf_counter()
loss = step()
torch.accelerator.synchronize()
print(f"B={B} V={V} {mode}: first step {time.perf_counter() - t0:.1f}s loss {loss.item():.4f}", flush=True)
step()
torch.accelerator.synchronize()
n = 3
t0 = time.perf_counter()
for _ in range(n):
    step()
torch.accelerator.synchronize()
dt = (time.perf_counter() - t0) / n
tokens = B * T * (V * P + 1)
flops = 8 * 684e6 * tokens  # fwd + bwd + checkpoint recompute, linear layers only
print(f"B={B} V={V} {mode}: {dt * 1e3:.0f} ms/step, {B / dt:.1f} seq/s, ~{flops / dt / 1e12:.0f} TFLOP/s linear, "
      f"peak mem {torch.accelerator.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
