"""Input adapters: V-JEPA patch features and robot state -> predictor tokens, with RoboJEPA-300M weights."""

import torch
from torch import nn

# RoboJEPA's embodiment list, in its order, so per-embodiment weights copy over index for index.
ROBOJEPA_EMBODIMENTS = [
    "franka", "franka_joints", "franka_omron", "franka_omron_state_delta", "so-101", "agibot_robotic",
    "franka_roboset", "bridge", "agilex", "robomind_franka", "google", "kuka", "franka_robonet", "baxter",
    "sawyer", "widowx_robonet", "language_table", "franka_fmb", "kuka_robonet", "agibot_humanoid", "1x_robot",
    "franka_cart_action_key", "franka_joint_action_key", "franka_cart_vel_action_key",
    "franka_joint_vel_action_key", "so-101_joints",
]  # fmt: skip
# Robots RoboJEPA never saw, appended after its list. Value: the RoboJEPA robot whose state encoder warm-starts it.
NEW_EMBODIMENTS = {"ur5": "franka"}  # same 7-d xyz + euler + closedness layout, single arm
EMBODIMENTS = ROBOJEPA_EMBODIMENTS + list(NEW_EMBODIMENTS)

# RoboJEPA's semantic camera ids (configs/scaling/wm_4b.yaml view_map). "side" is Bridge's camera1/camera2.
CAMERAS = {"left": 0, "right": 1, "wrist": 2, "back": 3, "side": 4, "top": 5, "front": 6}

STATE_DIM = 64  # RoboJEPA zero-pads every robot's state to 64


class StateEncoder(nn.Module):
    """Per-embodiment linear map from a zero-padded 64-d state to one predictor token (RoboJEPA's state encoder).

    The copied weights expect RoboJEPA's raw state layout: xyz (m), euler xyz (rad), gripper closedness in [0, 1],
    zero-padded to 64. RoboJEPA normalizes nothing, so neither may we.
    """

    def __init__(self, dim: int, num_embodiments: int = len(EMBODIMENTS), state_dim: int = STATE_DIM):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_embodiments, state_dim, dim))
        self.bias = nn.Parameter(torch.zeros(num_embodiments, dim))
        nn.init.normal_(self.weight, std=state_dim**-0.5)

    def forward(self, state, embodiment):
        """state (B, T, 64); embodiment (B,) long -> (B, T, dim)."""
        return torch.einsum("btp,bpd->btd", state, self.weight[embodiment]) + self.bias[embodiment].unsqueeze(1)


class VisualInput(nn.Module):
    """V-JEPA ViT-L patch features -> predictor tokens.

    A fresh linear adapter per camera type (RoboJEPA's 1664-d projection can't take ViT-L's 1024-d features), then
    RoboJEPA's learned camera token and robot token added to every patch, as in its predictor.
    """

    def __init__(self, enc_dim: int, dim: int, num_cameras: int = len(CAMERAS), num_embodiments: int = len(EMBODIMENTS)):
        super().__init__()
        self.adapters = nn.ModuleList(nn.Linear(enc_dim, dim, bias=False) for _ in range(num_cameras))
        self.camera_tokens = nn.Parameter(torch.zeros(num_cameras, dim))
        self.embodiment_tokens = nn.Parameter(torch.zeros(num_embodiments, dim))

    def forward(self, feats, camera, embodiment):
        """feats (B, T, V, P, enc_dim); camera (B, V) RoboJEPA camera id; embodiment (B,) -> (B, T, V, P, dim)."""
        # Each view's adapter is gathered by camera id and applied in one batched matmul: no per-camera loop over
        # camera.unique() and boolean masks, each of which would sync the GPU.
        weight = torch.stack([a.weight for a in self.adapters])[camera]  # (B, V, dim, enc_dim)
        out = torch.einsum("btvpe,bvde->btvpd", feats, weight)
        tokens = self.camera_tokens[camera][:, None, :, None] + self.embodiment_tokens[embodiment][:, None, None, None]
        return out + tokens


@torch.no_grad()
def load_robojepa_inputs(state_encoder: StateEncoder, visual: VisualInput, path: str) -> None:
    """Warm-start the state encoder, camera tokens and robot tokens from RoboJEPA-300M.

    Only values are copied; the parameters stay trainable and are updated like any other weight.
    New robots get the state encoder of the RoboJEPA robot named in NEW_EMBODIMENTS and a zero robot token, which is
    RoboJEPA's own init for these tokens. Call after any global weight init.
    """
    sd = torch.load(path, map_location="cpu", weights_only=True)["model"]
    n = len(ROBOJEPA_EMBODIMENTS)
    state_encoder.weight[:n].copy_(sd["state_encoder.patch_embed.w"].permute(2, 0, 1))  # (64, D, E) -> (E, 64, D)
    state_encoder.bias[:n].copy_(sd["state_encoder.patch_embed.b"])
    visual.camera_tokens.copy_(sd["view_assignment_tokens"][0])
    visual.embodiment_tokens[:n].copy_(sd["embodiment_assignment_tokens"][0])
    for i, source in enumerate(NEW_EMBODIMENTS.values(), start=n):
        src = ROBOJEPA_EMBODIMENTS.index(source)
        state_encoder.weight[i].copy_(state_encoder.weight[src])
        state_encoder.bias[i].copy_(state_encoder.bias[src])
        visual.embodiment_tokens[i].zero_()
