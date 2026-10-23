from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from sawyer.encoders import IMAGENET_MEAN, IMAGENET_STD, QWEN_DIR, VJEPA_CKPT, load_tokenizer
from sawyer.model import SawyerConfig
from sawyer.training import SawyerCollator, SawyerForTraining, param_groups

SMALL = dict(dim=64, depth=2, num_heads=4, ffn_dim=128, enc_dim=32, text_dim=48, grad_checkpointing=False)


class FakeVJEPA(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(3, 32)

    def train(self, mode=True):
        return super().train(False)

    @torch.no_grad()
    def forward(self, frames):  # (N, 32, 32, 3) -> (N, 4, 32): 2x2 grid of 16px patches
        x = frames.float().unflatten(1, (2, 16)).unflatten(3, (2, 16)).mean((2, 4)).flatten(1, 2)
        return self.proj(x / 255)


class FakeQwen(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(200000, 48)

    def train(self, mode=True):
        return super().train(False)

    @torch.no_grad()
    def forward(self, input_ids, attention_mask):
        return self.emb(input_ids)


def _sample(rng, n_views=2):
    view_valid = np.array([True, n_views == 2])
    frames = rng.integers(0, 255, (9, 2, 32, 32, 3), dtype=np.uint8)
    frames[:, ~view_valid] = 0
    return {
        "frames": frames, "camera": np.array([0, 2]), "view_valid": view_valid,
        "state": rng.normal(size=(9, 64)).astype(np.float32), "state_target": rng.normal(size=(9, 10)).astype(np.float32),
        "action_target": rng.normal(size=(9, 8, 7)).astype(np.float32), "action_valid": np.ones((9, 8), bool),
        "embodiment": 26, "instruction": "put the cup in the bowl" if n_views == 2 else "", "has_text": n_views == 2,
        "dataset": "ur5", "action_zero": np.zeros(7, np.float32), "state_scale": np.ones(10, np.float32),
    }  # fmt: skip


@pytest.fixture(scope="module")
def tokenizer():
    if not QWEN_DIR.exists():
        pytest.skip("Qwen checkpoint not downloaded")
    return load_tokenizer()


def test_collator_pads_text_on_the_right(tokenizer):
    rng = np.random.default_rng(0)
    batch = SawyerCollator(tokenizer)([_sample(rng), _sample(rng, n_views=1)])
    assert batch["frames"].shape == (2, 9, 2, 32, 32, 3) and batch["frames"].dtype == torch.uint8
    assert batch["input_ids"].shape == (2, 63)  # fixed length for the compiled blocks (64 keys with the null token)
    n = int(batch["text_mask"][0].sum())
    assert batch["text_mask"][0, :n].all() and not batch["text_mask"][0, n:].any()  # right padding
    assert batch["text_mask"][1, 0] and not batch["text_mask"][1, 1:].any()  # "" -> one EOS, then padding
    assert batch["has_text"].tolist() == [True, False] and batch["embodiment"].tolist() == [26, 26]


def test_wrapper_trains_predictor_only_and_keeps_encoders_frozen(tokenizer):
    rng = np.random.default_rng(0)
    vjepa, qwen = FakeVJEPA(), FakeQwen()
    model = SawyerForTraining(SawyerConfig(**SMALL), vjepa, qwen, image_size=32)
    model.train()
    assert not vjepa.training and not qwen.training
    assert all(k.startswith("predictor.") for k in model.state_dict())
    assert {id(p) for p in model.parameters()}.isdisjoint({id(p) for p in [*vjepa.parameters(), *qwen.parameters()]})

    batch = SawyerCollator(tokenizer)([_sample(rng), _sample(rng, n_views=1)])
    batch.pop("dataset")  # the Trainer removes it before calling the model
    losses = model(**batch)
    assert all(torch.isfinite(v).all() for v in losses.values())
    losses["loss"].backward()
    assert model.predictor.visual_in.adapters[0].weight.grad is not None
    assert vjepa.proj.weight.grad is None and qwen.emb.weight.grad is None


def test_every_slot_is_encoded_in_one_call_and_padded_views_change_nothing(tokenizer):
    calls = []

    class Recording(FakeVJEPA):
        def forward(self, frames):
            calls.append(len(frames))
            return super().forward(frames)

    torch.manual_seed(0)
    model = SawyerForTraining(SawyerConfig(**SMALL), Recording(), FakeQwen(), image_size=32).eval()
    rng = np.random.default_rng(0)
    batch = SawyerCollator(tokenizer)([_sample(rng), _sample(rng, n_views=1)])
    batch.pop("dataset")
    ref = model(**batch)
    assert calls == [2 * 9 * 2]  # fixed shape: padded slots included
    noisy = {**batch, "frames": batch["frames"].clone()}
    noisy["frames"][1, :, 1] = 255  # whatever sits in the padded slot is zeroed after encoding
    out = model(**noisy)
    for k in ref:
        torch.testing.assert_close(out[k], ref[k])


def test_weight_decay_groups():
    model = SawyerForTraining(SawyerConfig(**SMALL), FakeVJEPA(), FakeQwen(), image_size=32)
    decay, no_decay = param_groups(model, 0.1)
    names = {id(p): n.removeprefix("predictor.") for n, p in model.named_parameters()}
    no_decay_names = {names[id(p)] for p in no_decay["params"]}
    decay_names = {names[id(p)] for p in decay["params"]}
    assert decay["weight_decay"] == 0.1 and no_decay["weight_decay"] == 0.0
    assert len(decay_names) + len(no_decay_names) == len(names)
    for n in ["state_in.weight", "state_in.bias", "visual_in.camera_tokens", "visual_in.embodiment_tokens",
              "text_in.null", "blocks.0.attn.view_embed.weight", "blocks.0.attn.view_gate", "blocks.0.attn_norm.weight",
              "blocks.0.attn.q_norm.weight", "out_norm.weight"]:  # fmt: skip
        assert n in no_decay_names, n
    for n in ["blocks.0.attn.qkv.weight", "blocks.0.mlp.gate_up.weight", "visual_in.adapters.0.weight", "action_head.0.weight", "state_out"]:
        assert n in decay_names, n


@pytest.mark.skipif(not (torch.cuda.is_available() and Path(VJEPA_CKPT).exists()), reason="needs GPU and V-JEPA checkpoint")
def test_real_vjepa_normalizes_on_gpu_and_stays_frozen():
    from sawyer.encoders import FrozenVJEPA

    vjepa = FrozenVJEPA().cuda()
    vjepa.train()
    assert not vjepa.training and not any(p.requires_grad for p in vjepa.parameters())
    frames = torch.randint(0, 255, (3, 224, 224, 3), dtype=torch.uint8, device="cuda")
    feats = vjepa(frames)
    assert feats.shape == (3, 196, 1024) and feats.device.type == "cuda"
    # Same result as normalizing by hand and calling the raw encoder.
    x = frames.permute(0, 3, 1, 2).float() / 255
    x = (x - torch.tensor(IMAGENET_MEAN, device="cuda").view(1, 3, 1, 1)) / torch.tensor(IMAGENET_STD, device="cuda").view(1, 3, 1, 1)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        ref = vjepa.encoder(x.unsqueeze(2))
    torch.testing.assert_close(feats, ref)


@pytest.mark.skipif(not (torch.cuda.is_available() and Path(VJEPA_CKPT).exists()), reason="needs GPU and V-JEPA checkpoint")
def test_compiled_vjepa_matches_eager():
    from sawyer.encoders import FrozenVJEPA

    vjepa = FrozenVJEPA().cuda()
    frames = torch.randint(0, 255, (4, 224, 224, 3), dtype=torch.uint8, device="cuda")
    with torch.no_grad():  # fp32 reference: eager and compiled bf16 both sit ~1.5% (relative) away from it
        x = (frames.permute(0, 3, 1, 2).unsqueeze(2).float() / 255 - vjepa.mean) / vjepa.std
        fp32 = vjepa.encoder(x).float()
    eager = vjepa(frames)
    vjepa.compile_encoder()  # fullgraph=True: raises if the encoder doesn't compile as one graph
    compiled = vjepa(frames)

    def rel(a):
        return ((a.float() - fp32).norm() / fp32.norm()).item()

    assert compiled.shape == eager.shape and compiled.dtype == eager.dtype
    assert rel(compiled) <= 1.25 * rel(eager)  # fused kernels round differently, but no further from fp32


def test_collator_drops_the_empty_view_slot_only_when_no_sample_uses_it(tokenizer):
    rng = np.random.default_rng(0)
    collate = SawyerCollator(tokenizer)
    single = collate([_sample(rng, n_views=1), _sample(rng, n_views=1)])
    assert single["frames"].shape[2] == 1 and single["camera"].shape == (2, 1) and single["view_valid"].all()
    mixed = collate([_sample(rng), _sample(rng, n_views=1)])
    assert mixed["frames"].shape[2] == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")
def test_flex_attention_path_matches_sdpa():
    from sawyer.model import SawyerPredictor

    torch.manual_seed(0)
    cfg = SawyerConfig(dim=128, depth=2, num_heads=4, ffn_dim=256, enc_dim=32, text_dim=48, grad_checkpointing=False)
    model = SawyerPredictor(cfg).cuda().eval()
    B, T, V, P = 2, 3, 2, 4
    batch = dict(
        feats=torch.randn(B, T, V, P, 32, device="cuda"), grid=(2, 2), camera=torch.tensor([[0, 2], [3, 0]], device="cuda"),
        view_valid=torch.tensor([[True, True], [True, False]], device="cuda"), state=torch.randn(B, T, 64, device="cuda"),
        embodiment=torch.tensor([0, 26], device="cuda"), text=torch.randn(B, 5, 48, device="cuda"),
        text_mask=torch.ones(B, 5, dtype=torch.bool, device="cuda"), has_text=torch.tensor([True, False], device="cuda"),
        state_now=torch.randn(B, T, 10, device="cuda"),
    )  # fmt: skip
    with torch.no_grad():
        ref = model(**batch)
        model.compile_blocks()
        out = model(**batch)
    for a, b in zip(ref, out):
        torch.testing.assert_close(a, b, rtol=2e-3, atol=2e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")
def test_flex_attention_path_matches_sdpa_gradients():
    from sawyer.model import SawyerPredictor

    torch.manual_seed(0)
    cfg = SawyerConfig(dim=128, depth=2, num_heads=4, ffn_dim=256, enc_dim=32, text_dim=48, grad_checkpointing=True)
    model = SawyerPredictor(cfg).cuda().train()
    torch.nn.init.normal_(model.blocks[0].attn.view_gate)  # make the view-bias path carry gradient too
    B, T, V, P = 2, 3, 2, 4
    batch = dict(
        feats=torch.randn(B, T, V, P, 32, device="cuda"), grid=(2, 2), camera=torch.tensor([[0, 2], [3, 0]], device="cuda"),
        view_valid=torch.tensor([[True, True], [True, False]], device="cuda"), state=torch.randn(B, T, 64, device="cuda"),
        embodiment=torch.tensor([0, 26], device="cuda"), text=torch.randn(B, 5, 48, device="cuda"),
        text_mask=torch.ones(B, 5, dtype=torch.bool, device="cuda"), has_text=torch.tensor([True, False], device="cuda"),
        state_now=torch.randn(B, T, 10, device="cuda"),
    )  # fmt: skip

    def grads():
        model.zero_grad()
        visual, state, actions = model(**batch)
        (visual.square().mean() + state.square().mean() + actions.square().mean()).backward()
        return {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}

    ref = grads()
    model.compile_blocks()
    out = grads()
    assert ref.keys() == out.keys()
    for name in ref:
        torch.testing.assert_close(out[name], ref[name], rtol=5e-3, atol=5e-4, msg=name)


def test_baselines_score_trivial_predictors():
    from sawyer.training import baselines

    feats = torch.zeros(1, 3, 2, 4, 8)
    feats[:, 1:] = 1.0  # frame changes from step 0 to 1, then stays
    view_valid = torch.tensor([[True, False]])
    state = torch.tensor([[[0.0] * 10, [1.0] * 10, [1.0] * 10]])
    action_target = torch.full((1, 3, 8, 7), 0.5)
    action_valid = torch.ones(1, 3, 8, dtype=torch.bool)
    out = baselines(feats, view_valid, state, action_target, action_valid, torch.full((1, 7), -0.5))
    torch.testing.assert_close(out["visual_copy"], torch.tensor(0.5))  # wrong by 1 on half the transitions
    torch.testing.assert_close(out["state_copy"], torch.tensor(0.5))
    torch.testing.assert_close(out["action_zero"], torch.tensor(1.0))


def test_eval_mode_adds_baselines_training_mode_does_not(tokenizer):
    rng = np.random.default_rng(0)
    model = SawyerForTraining(SawyerConfig(**SMALL), FakeVJEPA(), FakeQwen(), image_size=32)
    batch = SawyerCollator(tokenizer)([_sample(rng), _sample(rng)])
    batch.pop("dataset")
    assert "action_zero" not in model.train()(**batch)
    assert {"visual_copy", "state_copy", "action_zero"} <= model.eval()(**batch).keys()


def test_state_loss_is_measured_in_units_of_each_datasets_typical_change():
    from sawyer.model import sawyer_loss

    B, T = 2, 3
    feats = torch.zeros(B, T, 1, 4, 8)
    state_target = torch.zeros(B, T, 10)
    state_next = torch.zeros(B, T, 10)
    state_next[0] += 0.01  # sample 0 off by 0.01 everywhere, sample 1 exact
    outputs = (feats.clone(), state_next, torch.zeros(B, T, 8, 7))
    args = (feats, torch.ones(B, 1, dtype=torch.bool), state_target, torch.zeros(B, T, 8, 7), torch.ones(B, T, 8, dtype=torch.bool))
    weights = {"visual": 1.0, "state": 1.0, "action": 1.0}
    raw = sawyer_loss(outputs, *args, weights)["state"]
    scaled = sawyer_loss(outputs, *args, weights, state_scale=torch.full((B, 10), 0.01))["state"]
    torch.testing.assert_close(raw, torch.tensor(0.005))
    torch.testing.assert_close(scaled, torch.tensor(0.5))  # 0.01 error = one typical step, averaged with an exact sample


def test_per_sample_sums_reproduce_the_batch_losses():
    from sawyer.model import sawyer_loss

    torch.manual_seed(0)
    B, T = 3, 4
    feats, visual = torch.randn(B, T, 2, 5, 8), torch.randn(B, T, 2, 5, 8)
    view_valid = torch.tensor([[True, True], [True, False], [True, True]])
    action_valid = torch.rand(B, T, 8) > 0.3
    out = sawyer_loss((visual, torch.zeros(B, T, 10), torch.randn(B, T, 8, 7)), feats, view_valid, torch.zeros(B, T, 10),
                      torch.randn(B, T, 8, 7), action_valid, {"visual": 1.0, "state": 1.0, "action": 1.0})  # fmt: skip
    torch.testing.assert_close(out["_action_sum"].sum() / out["_action_count"].sum(), out["action"])
    torch.testing.assert_close(out["_visual_sum"].sum() / out["_visual_count"].sum(), out["visual"])


def test_action_agreement_detects_shrinkage_and_correlation():
    from sawyer.training import action_agreement

    torch.manual_seed(0)
    target = torch.randn(64, 8, 7)
    valid = torch.ones(64, 8, dtype=torch.bool)
    shrunk = action_agreement(0.25 * target, target, valid)
    assert shrunk["corr_xyz"] > 0.99 and shrunk["corr_rot"] > 0.99 and abs(shrunk["spread"] - 0.25) < 1e-6
    noise = action_agreement(torch.randn(64, 8, 7), target, valid)
    assert abs(noise["corr_xyz"]) < 0.1 and abs(noise["spread"] - 1) < 0.15
    valid[:, 4:] = False
    target[:, 4:] = 1e6  # entries past the episode end must not count
    assert action_agreement(0.5 * target, target, valid)["spread"] == pytest.approx(0.5)


def test_trainer_accumulates_losses_per_dataset_without_mixing(tokenizer):
    import types

    from sawyer.training import PER_SAMPLE_KEYS, SawyerTrainer

    rng = np.random.default_rng(0)
    model = SawyerForTraining(SawyerConfig(**SMALL), FakeVJEPA(), FakeQwen(), image_size=32).train()
    a, b = _sample(rng), _sample(rng)
    b["dataset"] = "droid"
    batch = SawyerCollator(tokenizer)([a, b])
    fake = types.SimpleNamespace(_parts={"visual": 0.0, "state": 0.0, "action": 0.0}, _num_parts=0, _samples=0,
                                 _dataset_ids={"droid": 0, "ur5": 1}, _per_dataset=None)  # fmt: skip
    SawyerTrainer.compute_loss(fake, model, dict(batch))
    per_sample = model.eval()(**{k: v for k, v in batch.items() if k != "dataset"})
    for k in PER_SAMPLE_KEYS:
        # row 0 = droid = sample b, row 1 = ur5 = sample a (eval and train passes match: no dropout)
        torch.testing.assert_close(fake._per_dataset[k], per_sample[k].flip(0), rtol=1e-4, atol=1e-5)


def test_update_ratios_cover_every_trainable_part():
    from sawyer.updates import ROWS, SERIES, columns, update_ratios

    torch.manual_seed(0)
    model = SawyerForTraining(SawyerConfig(**SMALL), FakeVJEPA(), FakeQwen(), image_size=32)
    for p in model.parameters():
        p.grad = torch.randn_like(p)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    opt.step()
    out = update_ratios(model, opt, num_blocks=2)
    assert list(out["cells"]) == [r for r in ROWS if r != "other"]  # every part present, nothing unclassified
    assert list(out["series"]) == [s for s in SERIES if s != "other"]
    assert all(len(v) == len(columns(2)) for v in out["cells"].values())
    # After one AdamW step m_hat / sqrt(v_hat) = sign(g), so the update norm is lr * sqrt(numel).
    w = model.predictor.blocks[0].attn.qkv.weight
    expected = 1e-3 * w.numel() ** 0.5 / w.norm().item()
    assert out["cells"]["self-attn qkv"][1] == pytest.approx(expected, rel=1e-3)  # column 0 is the inputs
    w = model.predictor.action_head[0].weight
    expected = 1e-3 * w.numel() ** 0.5 / w.norm().item()
    assert out["cells"]["action head in"][-1] == pytest.approx(expected, rel=1e-3)


def test_full_size_predictor_has_no_unclassified_parameters():
    from sawyer.model import SawyerPredictor
    from sawyer.updates import OTHER, classify

    with torch.device("meta"):
        model = SawyerPredictor(SawyerConfig())
    assert [n for n, _ in model.named_parameters() if classify(n)[0] == OTHER] == []


def test_update_tracker_history_survives_restarts(tmp_path):
    from types import SimpleNamespace

    from sawyer.updates import UpdateTracker

    path = tmp_path / "updates.jsonl"
    path.write_text("".join(f'{{"step": {s}, "groups": {{}}}}\n' for s in [50, 100, 150, 100]))  # resumed at 100
    tracker = UpdateTracker(str(path), num_blocks=2, every=50)
    assert [r["step"] for r in tracker.records] == [50, 100]
    model = SawyerForTraining(SawyerConfig(**SMALL), FakeVJEPA(), FakeQwen(), image_size=32)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    tracker.on_step_end(None, SimpleNamespace(is_world_process_zero=True, global_step=149), None, model=model, optimizer=opt)
    assert [r["step"] for r in tracker.records] == [50, 100]  # not a multiple of `every`
