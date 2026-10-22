"""Sawyer predictor: assembles vision + proprio tokens per step and runs the block-causal transformer.

Each step's state token predicts a chunk of the next ACTION_HORIZON relative delta actions in one pass; that chunk is
the only deployment output (predict_actions). The next-step V-JEPA latent and robot state heads are training-only
auxiliary losses that shape the trunk into a world model."""

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers import PreTrainedConfig

from sawyer.inputs import CAMERAS, EMBODIMENTS, StateEncoder, VisualInput, load_robojepa_inputs
from sawyer.modules import Block, TextContext, block_causal_block_mask, block_causal_mask
from sawyer.rope import COL, ROW, VIEW, RoPE4D
from sawyer.rope import T as TIME

STATE_TARGET_DIM = 10  # xyz + 6D rotation (first two rotation-matrix columns) + gripper: no euler wrap at ±pi
ACTION_DIM = 7  # RoboJEPA's state-delta action: dxyz, relative rotation R_{t+1} R_t^T as euler xyz, dgripper
ACTION_HORIZON = 8  # 1.6 s at 5 Hz


class SawyerConfig(PreTrainedConfig):
    model_type = "sawyer"

    def __init__(
        self,
        dim: int = 1024,
        depth: int = 32,
        num_heads: int = 16,
        ffn_dim: int = 4096,
        enc_dim: int = 1024,
        text_dim: int = 1024,
        init_std: float = 0.02,
        grad_checkpointing: bool = True,
        loss_weights: dict | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dim = dim
        self.depth = depth
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        self.enc_dim = enc_dim
        self.text_dim = text_dim
        self.init_std = init_std
        self.grad_checkpointing = grad_checkpointing
        self.loss_weights = loss_weights or {"visual": 1.0, "state": 1.0, "action": 1.0}


class SawyerPredictor(nn.Module):
    def __init__(self, cfg: SawyerConfig):
        super().__init__()
        self.cfg = cfg
        self.visual_in = VisualInput(cfg.enc_dim, cfg.dim)
        self.state_in = StateEncoder(cfg.dim)
        self.text_in = TextContext(cfg.text_dim, cfg.dim)
        self.rope = RoPE4D(cfg.dim // cfg.num_heads)
        self.blocks = nn.ModuleList(
            Block(cfg.dim, cfg.num_heads, cfg.ffn_dim, num_camera_types=len(CAMERAS)) for _ in range(cfg.depth)
        )
        self.out_norm = nn.RMSNorm(cfg.dim, eps=1e-6)
        self.visual_out = nn.Linear(cfg.dim, cfg.enc_dim, bias=False)
        self.state_out = nn.Parameter(torch.empty(len(EMBODIMENTS), cfg.dim, STATE_TARGET_DIM))  # per-robot decoder
        self.action_head = nn.Sequential(
            nn.Linear(cfg.dim, cfg.dim, bias=False), nn.SiLU(), nn.Linear(cfg.dim, ACTION_HORIZON * ACTION_DIM, bias=False)
        )
        self.use_flex = False
        self.init_weights()

    def compile_blocks(self):
        """Fast training path: torch.compile each block (fuses norms, RoPE, view bias, SwiGLU) and switch the
        block-causal mask to flex_attention, which only compiles efficiently inside torch.compile."""
        self.use_flex = True
        for blk in self.blocks:
            blk.compile(dynamic=False)

    @torch.no_grad()
    def init_weights(self):
        std = self.cfg.init_std
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=std)
        # Scale every residual-branch output by the number of branches it sums with (3 per block).
        for blk in self.blocks:
            for out_proj in (blk.attn.proj, blk.cross.proj, blk.mlp.down):
                nn.init.trunc_normal_(out_proj.weight, std=std / (3 * self.cfg.depth) ** 0.5)
        nn.init.trunc_normal_(self.state_out, std=std)

    def load_robojepa(self, path: str):
        """Copy RoboJEPA-300M's state encoder, camera tokens and robot tokens over the fresh init."""
        load_robojepa_inputs(self.state_in, self.visual_in, path)

    def _assemble(self, feats, grid, camera, view_valid, state, embodiment):
        """Lay out each step as [view 0 patches, view 1 patches, ..., state token] and build per-token metadata."""
        B, T, V, P, _ = feats.shape
        S = V * P + 1  # tokens per step
        dev = feats.device
        x = torch.cat([self.visual_in(feats, camera, embodiment).flatten(2, 3), self.state_in(state, embodiment)[:, :, None]], 2)

        # Spatial coordinates start at 1 for patches, so 0 on row/col/view means "not a visual token": the state
        # token at (t, 0, 0, 0) shares no position with any patch. RoPE is shift-invariant, so patch geometry is unchanged.
        # The view axis is the view's slot in the sample (as in RoboJEPA), so two cameras of the same type never share
        # positions; the semantic camera id reaches the model through the adapter, camera token and attention bias.
        patch = torch.arange(P, device=dev)
        pos = torch.zeros(B, T, S, 4, dtype=torch.long, device=dev)
        pos[..., TIME] = torch.arange(T, device=dev)[None, :, None]
        pos[:, :, :-1, ROW] = (patch // grid[1]).repeat(V) + 1
        pos[:, :, :-1, COL] = (patch % grid[1]).repeat(V) + 1
        pos[:, :, :-1, VIEW] = torch.arange(1, V + 1, device=dev).repeat_interleave(P)

        camera_type = F.pad(camera.repeat_interleave(P, dim=1) + 1, (0, 1))  # +1: index 0 is the state token
        valid = F.pad(view_valid.repeat_interleave(P, dim=1), (0, 1), value=True)
        camera_type = camera_type[:, None].expand(B, T, S).reshape(B, T * S)
        valid = valid[:, None].expand(B, T, S).reshape(B, T * S)
        return x.reshape(B, T * S, -1), pos.reshape(B, T * S, 4), camera_type, valid

    def _trunk(self, feats, grid, camera, view_valid, state, embodiment, text, text_mask, has_text):
        """Shared inputs of forward / predict_actions (see forward). Returns normed hidden states (B, T, V*P + 1, D)."""
        B, T, V, P, _ = feats.shape
        x, pos, camera_type, valid = self._assemble(feats, grid, camera, view_valid, state, embodiment)
        mask_fn = block_causal_block_mask if self.use_flex else block_causal_mask
        attn_mask = mask_fn(pos[..., TIME], valid)
        rope = self.rope(pos)
        context, ctx_mask = self.text_in(text, text_mask, has_text)

        for blk in self.blocks:
            if self.cfg.grad_checkpointing and self.training:
                x = checkpoint(blk, x, context, rope, camera_type, attn_mask, ctx_mask, use_reentrant=False)
            else:
                x = blk(x, context, rope, camera_type, attn_mask, ctx_mask)
        return self.out_norm(x).view(B, T, V * P + 1, -1)

    def _action_chunk(self, h_state):
        """(B, T, D) state-token outputs -> (B, T, ACTION_HORIZON, 7) relative delta actions starting at each step."""
        return self.action_head(h_state).unflatten(-1, (ACTION_HORIZON, ACTION_DIM))

    def forward(self, feats, grid, camera, view_valid, state, embodiment, text, text_mask, has_text):
        """Training pass.

        feats (B, T, V, P, enc_dim) V-JEPA patch features per step and view, P = grid[0] * grid[1]
        camera (B, V) RoboJEPA camera ids; view_valid (B, V) False for padded views
        state (B, T, 64) RoboJEPA raw state layout; embodiment (B,)
        text (B, L, text_dim) Qwen states; text_mask (B, L); has_text (B,)

        Returns visual (B, T, V, P, enc_dim) and state (B, T, 10), where step t predicts step t + 1 (auxiliary), and
        actions (B, T, ACTION_HORIZON, 7), where actions[:, t, k] goes from step t + k to t + k + 1.
        """
        B, T, V, P, _ = feats.shape
        h = self._trunk(feats, grid, camera, view_valid, state, embodiment, text, text_mask, has_text)
        visual = self.visual_out(h[:, :, :-1]).view(B, T, V, P, -1)
        h_state = h[:, :, -1]
        state_next = torch.einsum("btd,bdk->btk", h_state, self.state_out[embodiment])
        return visual, state_next, self._action_chunk(h_state)

    @torch.no_grad()
    def predict_actions(self, feats, grid, camera, view_valid, state, embodiment, text, text_mask, has_text):
        """Deployment: one forward pass over the observed steps (usually T = 1), returns only the action chunk
        (B, ACTION_HORIZON, 7) from the latest step. Visual and state heads are not run."""
        h = self._trunk(feats, grid, camera, view_valid, state, embodiment, text, text_mask, has_text)
        return self._action_chunk(h[:, -1:, -1])[:, 0]


def sawyer_loss(outputs, feats, view_valid, state_target, action_target, action_valid, weights):
    """Teacher-forced L1 losses.

    state_target (B, T, 10); action_target (B, T, ACTION_HORIZON, 7), normalized per dataset;
    action_valid (B, T, ACTION_HORIZON) False where the chunk runs past the end of the episode.
    """
    visual, state_next, actions = outputs
    err = (visual[:, :-1].float() - feats[:, 1:].float()).abs().mean(-1)  # (B, T - 1, V, P)
    mask = view_valid[:, None, :, None].expand_as(err)
    act_err = (actions.float() - action_target.float()).abs().mean(-1) * action_valid
    losses = {
        "visual": (err * mask).sum() / mask.sum(),
        "state": F.l1_loss(state_next[:, :-1].float(), state_target[:, 1:].float()),
        "action": act_err.sum() / action_valid.sum().clamp(min=1),
    }
    losses["loss"] = sum(weights[k] * losses[k] for k in weights)
    return losses
