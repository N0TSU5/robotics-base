import torch

from sawyer.modules import Block, TextContext, block_causal_mask
from sawyer.rope import RoPE4D


def _inputs(B=2, steps=3, patches=4, dim=64, heads=4, L=5):
    # Per step: `patches` visual tokens (camera type 1) then one state token (camera type 0).
    per_step = patches + 1
    N = steps * per_step
    timestep = torch.arange(steps).repeat_interleave(per_step).expand(B, N)
    pos = torch.zeros(B, N, 4, dtype=torch.long)
    pos[..., 0] = timestep
    pos[..., 1] = torch.arange(N) % per_step  # stand-in row index; state token gets a nonzero value here too
    camera_type = (torch.arange(N) % per_step < patches).long().expand(B, N)
    valid = torch.ones(B, N, dtype=torch.bool)
    text_mask = torch.ones(B, 1, 1, L, dtype=torch.bool)
    return dict(
        x=torch.randn(B, N, dim),
        text=torch.randn(B, L, dim),
        rope=RoPE4D(dim // heads)(pos),
        camera_type=camera_type,
        attn_mask=block_causal_mask(timestep, valid),
        text_mask=text_mask,
    ), per_step


def test_block_param_count_matches_spec():
    block = Block(dim=1024, num_heads=16, ffn_dim=4096, num_camera_types=7)
    linear = sum(m.weight.numel() for m in block.modules() if isinstance(m, torch.nn.Linear))
    assert linear == 20_971_520  # 4d^2 self + 4d^2 cross + 3*d*4096 SwiGLU; x32 layers = 671.1M
    assert all(m.bias is None for m in block.modules() if isinstance(m, torch.nn.Linear))


def test_block_causal_future_steps_do_not_leak():
    torch.manual_seed(0)
    block = Block(dim=64, num_heads=4, ffn_dim=128, num_camera_types=7).eval()
    inp, per_step = _inputs()
    out = block(**inp)
    inp["x"] = inp["x"].clone()
    inp["x"][:, 2 * per_step :] += 10.0  # perturb only the last step
    out2 = block(**inp)
    torch.testing.assert_close(out[:, : 2 * per_step], out2[:, : 2 * per_step])
    assert not torch.allclose(out[:, 2 * per_step :], out2[:, 2 * per_step :])


def test_view_bias_is_per_head():
    block = Block(dim=64, num_heads=4, ffn_dim=128, num_camera_types=7)
    torch.nn.init.normal_(block.attn.view_embed.weight)
    bias = block.attn.view_embed(torch.tensor([[1]])).view(4, 16)
    assert not torch.allclose(bias[0], bias[1])
    block.attn.view_embed(torch.tensor([[0, 1]])).sum().backward()
    assert (block.attn.view_embed.weight.grad[0] == 0).all()  # non-visual tokens never learn a bias


def test_view_bias_is_bounded_and_starts_off():
    torch.manual_seed(0)
    block = Block(dim=64, num_heads=4, ffn_dim=128, num_camera_types=7).eval()
    inp, _ = _inputs()
    out = block(**inp)
    with torch.no_grad():
        block.attn.view_embed.weight[1:] = 0
    torch.testing.assert_close(block(**inp), out)  # gate=0: the bias has no effect at init

    with torch.no_grad():
        block.attn.view_embed.weight.mul_(0).add_(1e4)
        block.attn.view_gate.fill_(1e4)
    attn = block.attn
    raw = attn.view_embed(torch.tensor([[1]])).view(1, 1, 4, 16).transpose(1, 2)
    bias = torch.nn.functional.rms_norm(raw, (16,), eps=1e-6) * torch.tanh(attn.view_gate)
    assert bias.pow(2).mean(-1).sqrt().max() <= 1 + 1e-4  # per-channel RMS capped at 1
    assert torch.isfinite(block(**inp)).all()


def test_null_text_only_is_finite():
    block = Block(dim=64, num_heads=4, ffn_dim=128, num_camera_types=7)
    inp, _ = _inputs()
    inp["text_mask"] = torch.zeros_like(inp["text_mask"])
    inp["text_mask"][..., 0] = True  # only the null token is visible
    assert torch.isfinite(block(**inp)).all()


def test_padding_keys_are_ignored():
    torch.manual_seed(0)
    block = Block(dim=64, num_heads=4, ffn_dim=128, num_camera_types=7).eval()
    inp, per_step = _inputs()
    timestep = torch.arange(3).repeat_interleave(per_step).expand(2, -1)
    valid = torch.ones_like(timestep, dtype=torch.bool)
    valid[:, 1] = False  # pad one visual token of step 0
    inp["attn_mask"] = block_causal_mask(timestep, valid)
    out = block(**inp)
    inp["x"] = inp["x"].clone()
    inp["x"][:, 1] = 1e3
    out2 = block(**inp)
    keep = torch.arange(inp["x"].shape[1]) != 1
    torch.testing.assert_close(out[:, keep], out2[:, keep])


def test_text_context_never_fully_masks_and_routes_no_text_to_null():
    torch.manual_seed(0)
    ctx_mod = TextContext(text_dim=32, dim=64)
    block = Block(dim=64, num_heads=4, ffn_dim=128, num_camera_types=7).eval()
    hidden = torch.randn(3, 5, 32)
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 0, 0, 0, 0], [0, 0, 0, 0, 0]])  # row 2: all padding
    has_text = torch.tensor([True, False, True])  # row 1: "" tokenized to a lone EOS

    context, text_mask = ctx_mod(hidden, mask, has_text)
    assert context.shape == (3, 6, 64) and text_mask.shape == (3, 1, 1, 6)
    assert text_mask[..., 0].all()
    assert text_mask[1, ..., 1:].sum() == 0  # empty instruction sees only the null token
    assert text_mask[0, ..., 1:4].all()

    inp, _ = _inputs(B=3, L=6)
    inp["text"], inp["text_mask"] = context, text_mask
    out = block(**inp)
    assert torch.isfinite(out).all()
    out.sum().backward()
    assert ctx_mod.null.grad is not None and torch.isfinite(ctx_mod.null.grad).all()
