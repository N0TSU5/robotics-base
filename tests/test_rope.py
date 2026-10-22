import torch

from sawyer.rope import RoPE4D, apply_rope


def _scores(rope, q, k, pos):
    cos, sin = rope(pos)
    return apply_rope(q, cos, sin) @ apply_rope(k, cos, sin).transpose(-1, -2)


def test_scores_depend_only_on_relative_position():
    torch.manual_seed(0)
    rope = RoPE4D(head_dim=64)
    q, k = torch.randn(2, 1, 2, 5, 64).unbind(0)
    pos = torch.randint(0, 10, (1, 5, 4))
    shift = torch.tensor([3, 1, 4, 1])
    torch.testing.assert_close(_scores(rope, q, k, pos), _scores(rope, q, k, pos + shift))


def test_each_axis_only_touches_its_own_slab():
    rope = RoPE4D(head_dim=64)
    x = torch.randn(1, 1, 1, 64)
    for axis in range(4):
        pos = torch.zeros(1, 1, 4, dtype=torch.long)
        pos[..., axis] = 7
        out = apply_rope(x, *rope(pos))
        changed = (out - x).abs().squeeze() > 1e-6
        assert changed[axis * 16 : (axis + 1) * 16].any()
        assert not torch.cat([changed[: axis * 16], changed[(axis + 1) * 16 :]]).any()


def test_zero_position_is_identity():
    rope = RoPE4D(head_dim=64)
    x = torch.randn(2, 16, 9, 64)
    torch.testing.assert_close(apply_rope(x, *rope(torch.zeros(2, 9, 4))), x)
