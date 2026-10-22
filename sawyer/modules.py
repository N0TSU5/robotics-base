"""Predictor block: self-attn over vision + proprio tokens, cross-attn to text, SwiGLU MLP."""

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention.flex_attention import BlockMask, create_block_mask, flex_attention

from sawyer.rope import apply_rope

# Compiled, create_block_mask evaluates mask_mod per tile instead of materializing the dense (B, N, N) mask eagerly.
_create_block_mask = torch.compile(create_block_mask)


def block_causal_mask(timestep: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """timestep (B, N) int, valid (B, N) bool -> (B, 1, N, N) bool, True = may attend.

    A query at step t sees every valid key at step <= t: all views and the state token of its own step and
    all earlier steps. Padded queries still see step 0, so no row is fully masked.
    """
    allowed = timestep.unsqueeze(-1) >= timestep.unsqueeze(-2)
    return (allowed & valid.unsqueeze(-2)).unsqueeze(1)


def block_causal_block_mask(timestep: torch.Tensor, valid: torch.Tensor) -> BlockMask:
    """Same mask as block_causal_mask, as a flex_attention BlockMask: fully masked 128x128 tiles (future steps)
    are skipped instead of computed and thrown away."""

    def mask_mod(b, h, q, kv):
        return (timestep[b, q] >= timestep[b, kv]) & valid[b, kv]

    B, N = timestep.shape
    return _create_block_mask(mask_mod, B=B, H=None, Q_LEN=N, KV_LEN=N, device=timestep.device)


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_camera_types: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-6)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=1e-6)
        # Camera-type bias added to q/k after RoPE, one 64-d vector per head (RoboJEPA shares one across heads).
        # Index 0 marks non-visual tokens and stays zero. The bias is RMS-normalized and scaled by tanh(gate), so
        # its per-channel RMS never exceeds that of the QK-normed q/k; gate=0 means training starts with no bias.
        self.view_embed = nn.Embedding(num_camera_types + 1, dim, padding_idx=0)
        self.view_gate = nn.Parameter(torch.zeros(num_heads, 1, 1))

    def forward(self, x, rope, camera_type, attn_mask):
        """x (B, N, D); rope = (cos, sin) from RoPE4D; camera_type (B, N) long;
        attn_mask: (B, 1, N, N) bool for SDPA, or a BlockMask for flex_attention (compiled training path)."""
        B, N, _ = x.shape
        q, k, v = self.qkv(x).view(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k = apply_rope(self.q_norm(q), *rope), apply_rope(self.k_norm(k), *rope)
        view_bias = self.view_embed(camera_type).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        view_bias = (F.rms_norm(view_bias, (self.head_dim,), eps=1e-6) * torch.tanh(self.view_gate)).to(q.dtype)
        q, k = (q + view_bias).to(v.dtype), (k + view_bias).to(v.dtype)
        if isinstance(attn_mask, BlockMask):
            out = flex_attention(q, k, v, block_mask=attn_mask)
        else:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        return self.proj(out.transpose(1, 2).reshape(B, N, -1))


class CrossAttention(nn.Module):
    """Queries from the vision/proprio stream, keys/values from text tokens. No RoPE: Qwen already encodes order."""

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q = nn.Linear(dim, dim, bias=False)
        self.kv = nn.Linear(dim, 2 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-6)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=1e-6)

    def forward(self, x, text, text_mask):
        """x (B, N, D); text (B, L, D); text_mask (B, 1, 1, L) bool with at least one True per row (null token)."""
        B, N, _ = x.shape
        L = text.shape[1]
        q = self.q(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k, v = self.kv(text).view(B, L, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        out = F.scaled_dot_product_attention(self.q_norm(q), self.k_norm(k), v, attn_mask=text_mask)
        return self.proj(out.transpose(1, 2).reshape(B, N, -1))


class TextContext(nn.Module):
    """Qwen3 token states -> cross-attn keys/values, with a learned null token at index 0.

    The null token is always visible, so a sample with no text (empty instruction or dropped for
    unconditional training) attends only to it and the cross-attn softmax never sees a fully masked row.
    The 9:1 text / no-text split is decided by the sampler and arrives through has_text.
    """

    def __init__(self, text_dim: int, dim: int):
        super().__init__()
        self.norm = nn.RMSNorm(text_dim, eps=1e-6)  # Qwen token norms are ~100
        self.proj = nn.Linear(text_dim, dim, bias=False)
        self.null = nn.Parameter(torch.randn(1, 1, dim) * 0.02)

    def forward(self, hidden, mask, has_text):
        """hidden (B, L, text_dim) Qwen last_hidden_state; mask (B, L) bool tokenizer mask; has_text (B,) bool.

        Returns context (B, 1 + L, dim) and text_mask (B, 1, 1, 1 + L). Qwen tokenizes "" to one EOS token, so
        empty instructions must be flagged through has_text to land on the null token like dropped ones.
        """
        B = hidden.shape[0]
        mask = mask.bool() & has_text[:, None]
        context = torch.cat([self.null.expand(B, -1, -1), self.proj(self.norm(hidden))], dim=1)
        text_mask = torch.cat([mask.new_ones(B, 1), mask], dim=1)
        return context, text_mask[:, None, None, :]


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.gate_up = nn.Linear(dim, 2 * hidden_dim, bias=False)
        self.down = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        return self.down(F.silu(gate) * up)


class Block(nn.Module):
    """Pre-RMSNorm residual block with three branches: self-attn, cross-attn, MLP."""

    def __init__(self, dim: int, num_heads: int, ffn_dim: int, num_camera_types: int):
        super().__init__()
        self.attn_norm = nn.RMSNorm(dim, eps=1e-6)
        self.attn = SelfAttention(dim, num_heads, num_camera_types)
        self.cross_norm = nn.RMSNorm(dim, eps=1e-6)
        self.cross = CrossAttention(dim, num_heads)
        self.mlp_norm = nn.RMSNorm(dim, eps=1e-6)
        self.mlp = SwiGLU(dim, ffn_dim)

    def forward(self, x, text, rope, camera_type, attn_mask, text_mask):
        x = x + self.attn(self.attn_norm(x), rope, camera_type, attn_mask)
        x = x + self.cross(self.cross_norm(x), text, text_mask)
        return x + self.mlp(self.mlp_norm(x))
