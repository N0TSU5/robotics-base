from pathlib import Path

import pytest
import torch

from sawyer.inputs import CAMERAS
from sawyer.model import SawyerConfig, SawyerPredictor, sawyer_loss

ROBOJEPA = Path(__file__).parents[1] / "checkpoints/robojepa/cooldown_300m_300k.pth.tar"
SMALL = dict(dim=64, depth=2, num_heads=4, ffn_dim=128, enc_dim=32, text_dim=48, grad_checkpointing=False)


def _batch(B=2, T=3, V=2, grid=(2, 3), L=4, enc_dim=32, text_dim=48):
    P = grid[0] * grid[1]
    return dict(
        feats=torch.randn(B, T, V, P, enc_dim),
        grid=grid,
        camera=torch.tensor([[CAMERAS["left"], CAMERAS["wrist"]], [CAMERAS["back"], CAMERAS["left"]]])[:B],
        view_valid=torch.tensor([[True, True], [True, False]])[:B],  # sample 1 has a padded second view
        state=torch.randn(B, T, 64),
        embodiment=torch.tensor([0, 10])[:B],
        text=torch.randn(B, L, text_dim),
        text_mask=torch.ones(B, L, dtype=torch.bool),
        has_text=torch.tensor([True, False])[:B],
    )


def test_output_shapes():
    model = SawyerPredictor(SawyerConfig(**SMALL)).eval()
    visual, state, actions = model(**_batch())
    assert visual.shape == (2, 3, 2, 6, 32) and state.shape == (2, 3, 10) and actions.shape == (2, 3, 8, 7)


def test_predict_actions_is_one_pass_from_the_latest_step():
    torch.manual_seed(0)
    model = SawyerPredictor(SawyerConfig(**SMALL)).eval()
    batch = _batch()
    _, _, chunks = model(**batch)
    first = {**batch, "feats": batch["feats"][:, :1], "state": batch["state"][:, :1]}  # deployment: only step 0
    torch.testing.assert_close(model.predict_actions(**first), chunks[:, 0])
    torch.testing.assert_close(model.predict_actions(**batch), chunks[:, -1])
    assert model.predict_actions(**first).shape == (2, 8, 7)


def test_state_token_shares_no_position_with_any_patch():
    model = SawyerPredictor(SawyerConfig(**SMALL))
    b = _batch()
    _, pos, camera_type, _ = model._assemble(b["feats"], b["grid"], b["camera"], b["view_valid"], b["state"], b["embodiment"])
    is_state = camera_type == 0
    assert (pos[is_state][:, 1:] == 0).all()  # state: time axis only
    assert (pos[~is_state][:, 1:] >= 1).all()  # patches: every spatial axis starts at 1
    assert pos[~is_state][:, 1].max() == b["grid"][0] and pos[~is_state][:, 2].max() == b["grid"][1]


def test_same_type_cameras_get_distinct_positions():
    model = SawyerPredictor(SawyerConfig(**SMALL))
    b = _batch(B=1)
    b["camera"] = torch.tensor([[CAMERAS["side"], CAMERAS["side"]]])  # Bridge camera1 + camera2
    _, pos, camera_type, _ = model._assemble(b["feats"], b["grid"], b["camera"], b["view_valid"], b["state"], b["embodiment"])
    patches = pos[0][camera_type[0] != 0]
    assert len(torch.unique(patches, dim=0)) == len(patches)  # every patch has its own (t, row, col, view)


def test_no_leak_from_future_steps():
    torch.manual_seed(0)
    model = SawyerPredictor(SawyerConfig(**SMALL)).eval()
    batch = _batch()
    out = model(**batch)
    batch["feats"] = batch["feats"].clone()
    batch["state"] = batch["state"].clone()
    batch["feats"][:, 2] += 5.0
    batch["state"][:, 2] += 5.0
    out2 = model(**batch)
    torch.testing.assert_close(out[0][:, :2], out2[0][:, :2])  # visual predictions from steps 0, 1
    torch.testing.assert_close(out[1][:, :2], out2[1][:, :2])  # state predictions from steps 0, 1
    torch.testing.assert_close(out[2][:, :2], out2[2][:, :2])  # chunks from steps 0, 1 never see step 2
    assert not torch.allclose(out[2][:, 2], out2[2][:, 2])


def test_padded_view_is_ignored():
    torch.manual_seed(0)
    model = SawyerPredictor(SawyerConfig(**SMALL)).eval()
    batch = _batch()
    out = model(**batch)
    batch["feats"] = batch["feats"].clone()
    batch["feats"][1, :, 1] = 100.0  # sample 1's padded view
    out2 = model(**batch)
    torch.testing.assert_close(out[0][:, :, 0], out2[0][:, :, 0])
    torch.testing.assert_close(out[1], out2[1])
    torch.testing.assert_close(out[2], out2[2])


def test_loss_backprops_into_copied_inputs():
    torch.manual_seed(0)
    model = SawyerPredictor(SawyerConfig(**SMALL)).train()
    batch = _batch()
    out = model(**batch)
    action_valid = torch.ones(2, 3, 8, dtype=torch.bool)
    action_valid[:, 2, 6:] = False  # last step's chunk runs past the episode end
    losses = sawyer_loss(out, batch["feats"], batch["view_valid"], torch.randn(2, 3, 10), torch.randn(2, 3, 8, 7),
                         action_valid, model.cfg.loss_weights)  # fmt: skip
    assert all(torch.isfinite(v) for v in losses.values())
    losses["loss"].backward()
    assert model.state_in.weight.grad[[0, 10]].abs().sum() > 0  # only the two robots in the batch
    assert model.state_in.weight.grad[1].abs().sum() == 0
    assert model.state_in.bias.grad[[0, 10]].abs().sum() > 0
    assert model.visual_in.camera_tokens.grad[[CAMERAS["left"], CAMERAS["wrist"]]].abs().sum() > 0
    assert model.visual_in.embodiment_tokens.grad[[0, 10]].abs().sum() > 0


@pytest.mark.skipif(not ROBOJEPA.exists(), reason="RoboJEPA checkpoint not downloaded")
def test_robojepa_warm_start_stays_trainable():
    model = SawyerPredictor(SawyerConfig(**{**SMALL, "dim": 1024, "num_heads": 16, "enc_dim": 1024, "text_dim": 1024}))
    model.load_robojepa(str(ROBOJEPA))
    copied = [model.state_in.weight, model.state_in.bias, model.visual_in.camera_tokens, model.visual_in.embodiment_tokens]
    assert all(p.requires_grad for p in copied)
    params = {id(p) for p in model.parameters()}
    assert all(id(p) in params for p in copied)  # registered, so the optimizer will update them


def test_full_size_param_count():
    with torch.device("meta"):
        model = SawyerPredictor(SawyerConfig())
    blocks = sum(p.numel() for p in model.blocks.parameters())
    total = sum(p.numel() for p in model.parameters())
    assert 671e6 < blocks < 672e6
    assert total < 690e6


def test_action_loss_ignores_steps_past_episode_end():
    torch.manual_seed(0)
    model = SawyerPredictor(SawyerConfig(**SMALL)).eval()
    batch = _batch()
    out = model(**batch)
    target = torch.randn(2, 3, 8, 7)
    valid = torch.ones(2, 3, 8, dtype=torch.bool)
    valid[:, :, 5:] = False
    args = (batch["feats"], batch["view_valid"], torch.randn(2, 3, 10))
    base = sawyer_loss(out, *args, target, valid, model.cfg.loss_weights)["action"]
    target[:, :, 5:] = 1e6  # garbage beyond the episode must not matter
    torch.testing.assert_close(sawyer_loss(out, *args, target, valid, model.cfg.loss_weights)["action"], base)
