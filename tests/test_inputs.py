from pathlib import Path

import pytest
import torch

from sawyer.inputs import CAMERAS, EMBODIMENTS, ROBOJEPA_EMBODIMENTS, StateEncoder, VisualInput, load_robojepa_inputs

ROBOJEPA = Path(__file__).parents[1] / "checkpoints/robojepa/cooldown_300m_300k.pth.tar"


def test_state_encoder_picks_each_samples_embodiment():
    torch.manual_seed(0)
    enc = StateEncoder(dim=16)
    state = torch.randn(2, 3, 64)
    emb = torch.tensor([EMBODIMENTS.index("franka"), EMBODIMENTS.index("bridge")])
    out = enc(state, emb)
    for b in range(2):
        torch.testing.assert_close(out[b], state[b] @ enc.weight[emb[b]] + enc.bias[emb[b]])


def test_visual_input_routes_views_to_their_camera_adapter():
    torch.manual_seed(0)
    vis = VisualInput(enc_dim=8, dim=16)
    torch.nn.init.normal_(vis.camera_tokens)
    torch.nn.init.normal_(vis.embodiment_tokens)
    feats = torch.randn(2, 3, 2, 5, 8)  # B, T, V, P, enc_dim
    camera = torch.tensor([[CAMERAS["left"], CAMERAS["wrist"]], [CAMERAS["wrist"], CAMERAS["back"]]])
    emb = torch.tensor([0, 7])
    out = vis(feats, camera, emb)
    assert out.shape == (2, 3, 2, 5, 16)
    for b in range(2):
        for v in range(2):
            c = camera[b, v]
            expected = vis.adapters[c](feats[b, :, v]) + vis.camera_tokens[c] + vis.embodiment_tokens[emb[b]]
            torch.testing.assert_close(out[b, :, v], expected)


@pytest.mark.skipif(not ROBOJEPA.exists(), reason="RoboJEPA checkpoint not downloaded")
def test_load_robojepa_inputs_matches_checkpoint():
    enc, vis = StateEncoder(dim=1024), VisualInput(enc_dim=1024, dim=1024)
    load_robojepa_inputs(enc, vis, str(ROBOJEPA))
    sd = torch.load(ROBOJEPA, map_location="cpu", weights_only=True)["model"]
    # Same math as RoboJEPA's MultiEmbodimentLinear: einsum("btp,be,phe->bth") + onehot @ b
    state = torch.randn(2, 4, 64)
    emb = torch.tensor([EMBODIMENTS.index("franka"), EMBODIMENTS.index("kuka")])
    onehot = torch.eye(len(ROBOJEPA_EMBODIMENTS))[emb]
    ref = torch.einsum("btp,be,phe->bth", state, onehot, sd["state_encoder.patch_embed.w"])
    ref = ref + (onehot @ sd["state_encoder.patch_embed.b"]).unsqueeze(1)
    torch.testing.assert_close(enc(state, emb), ref, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(vis.camera_tokens, sd["view_assignment_tokens"][0])


@pytest.mark.skipif(not ROBOJEPA.exists(), reason="RoboJEPA checkpoint not downloaded")
def test_ur5_is_a_new_robot_warm_started_from_franka():
    enc, vis = StateEncoder(dim=1024), VisualInput(enc_dim=1024, dim=1024)
    load_robojepa_inputs(enc, vis, str(ROBOJEPA))
    ur5, franka = EMBODIMENTS.index("ur5"), EMBODIMENTS.index("franka")
    assert ur5 == len(ROBOJEPA_EMBODIMENTS)  # appended after RoboJEPA's 26
    torch.testing.assert_close(enc.weight[ur5], enc.weight[franka])
    torch.testing.assert_close(enc.bias[ur5], enc.bias[franka])
    assert not vis.embodiment_tokens[ur5].any()
    assert vis.embodiment_tokens[franka].abs().sum() > 0
