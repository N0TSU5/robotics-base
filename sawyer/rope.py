"""4D rotary position embedding over (timestep, row, col, view)."""

import torch
from torch import nn

# Order of the position columns and of the channel slabs inside each head.
T, ROW, COL, VIEW = range(4)


class RoPE4D(nn.Module):
    """Rotates each head's channels in 4 equal slabs, one per axis.

    With head_dim=64 every axis gets 16 channels (8 frequencies). Positions are
    integer coordinates of shape (B, N, 4) ordered (t, row, col, view). Proprio
    tokens use (t, 0, 0, 0), so only their time slab rotates.
    """

    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__()
        if head_dim % 8:
            raise ValueError(f"head_dim={head_dim} must split into 4 slabs of even width")
        self.slab = head_dim // 4
        inv_freq = 1.0 / base ** (torch.arange(0, self.slab, 2, dtype=torch.float32) / self.slab)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """pos (B, N, 4) -> cos, sin of shape (B, 1, N, head_dim), shared by every layer."""
        ang = pos.float().unsqueeze(-1) * self.inv_freq  # (B, N, 4, slab/2)
        ang = torch.cat([ang, ang], dim=-1).flatten(-2)  # (B, N, head_dim)
        return ang.cos().unsqueeze(1), ang.sin().unsqueeze(1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x (B, H, N, head_dim); rotate-half applied independently within each axis slab."""
    xf = x.float()
    x1, x2 = xf.unflatten(-1, (4, 2, -1)).unbind(-2)  # each (B, H, N, 4, slab/2)
    rotated = torch.stack((-x2, x1), dim=-2).flatten(-3)
    return (xf * cos + rotated * sin).type_as(x)
