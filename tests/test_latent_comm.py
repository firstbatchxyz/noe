"""Tests for latent communication module."""

import torch
import pytest

from noe_train.comm.latent import LatentChannel, LatentConfig, LatentProjector, LatentReceiver


@pytest.fixture
def config():
    return LatentConfig(hidden_dim=64, latent_dim=16, num_virtual_tokens=2, gate_init=-5.0)


def test_projector_shapes(config):
    proj = LatentProjector(config)
    hidden = torch.randn(2, 10, 64)  # batch=2, seq=10, hidden=64
    latent = proj(hidden)
    assert latent.shape == (2, 16)


def test_projector_with_mask(config):
    proj = LatentProjector(config)
    hidden = torch.randn(2, 10, 64)
    mask = torch.ones(2, 10)
    mask[0, 7:] = 0  # first sample has 7 real tokens
    latent = proj(hidden, attention_mask=mask)
    assert latent.shape == (2, 16)


def test_projector_last_pooling():
    cfg = LatentConfig(hidden_dim=64, latent_dim=16, pooling="last")
    proj = LatentProjector(cfg)
    hidden = torch.randn(1, 10, 64)
    latent = proj(hidden)
    assert latent.shape == (1, 16)


def test_receiver_shapes(config):
    recv = LatentReceiver(config)
    latent = torch.randn(2, 16)
    embeds = torch.randn(2, 10, 64)
    aug_embeds, aug_mask = recv(latent, embeds)
    # seq_len should grow by num_virtual_tokens
    assert aug_embeds.shape == (2, 12, 64)
    assert aug_mask is None  # no mask provided


def test_receiver_with_mask(config):
    recv = LatentReceiver(config)
    latent = torch.randn(2, 16)
    embeds = torch.randn(2, 10, 64)
    mask = torch.ones(2, 10)
    aug_embeds, aug_mask = recv(latent, embeds, attention_mask=mask)
    assert aug_embeds.shape == (2, 12, 64)
    assert aug_mask.shape == (2, 12)
    # Virtual tokens should always be attended to
    assert aug_mask[:, :2].all()


def test_gate_starts_closed(config):
    recv = LatentReceiver(config)
    assert recv.gate_value < 0.01  # sigmoid(-5) ≈ 0.007


def test_gate_controls_magnitude(config):
    recv = LatentReceiver(config)
    latent = torch.randn(1, 16)
    embeds = torch.randn(1, 5, 64)

    # Gate nearly closed — virtual tokens should have tiny magnitude
    aug, _ = recv(latent, embeds)
    virtual_norm = aug[:, :2, :].norm().item()
    original_norm = aug[:, 2:, :].norm().item()
    assert virtual_norm < original_norm * 0.1  # virtual tokens are ~1% of signal

    # Open the gate manually
    recv.gate.data = torch.tensor(5.0)  # sigmoid(5) ≈ 0.993
    assert recv.gate_value > 0.99
    aug_open, _ = recv(latent, embeds)
    virtual_norm_open = aug_open[:, :2, :].norm().item()
    assert virtual_norm_open > virtual_norm * 10  # much larger now


def test_channel_roundtrip(config):
    channel_a = LatentChannel(config)
    channel_b = LatentChannel(config)

    # Expert A produces hidden states → project to latent
    hidden_a = torch.randn(1, 20, 64)
    latent = channel_a.project(hidden_a)
    assert latent.shape == (1, 16)

    # Expert B receives the latent
    embeds_b = torch.randn(1, 15, 64)
    aug_embeds, aug_mask = channel_b.receive(latent, embeds_b)
    assert aug_embeds.shape == (1, 17, 64)  # 15 + 2 virtual tokens


def test_channel_param_count(config):
    channel = LatentChannel(config)
    # Projector: Linear(64→16) + LayerNorm(16) = 64*16+16 + 16+16
    # Receiver: Linear(16→128) + LayerNorm(64) + gate = 16*128+128 + 64+64 + 1
    count = channel.param_count()
    assert count > 0
    assert count < 100_000  # should be small


def test_channel_repr(config):
    channel = LatentChannel(config)
    r = channel.extra_repr()
    assert "latent_dim=16" in r
    assert "virtual_tokens=2" in r
    assert "gate=" in r


def test_default_config_params():
    """Test with default config (Qwen3.5-4B dimensions)."""
    config = LatentConfig()
    channel = LatentChannel(config)
    # Projector: Linear(2560→256) + LN = 2560*256+256 + 256+256 ≈ 656K
    # Receiver: Linear(256→10240) + LN + gate = 256*10240+10240 + 2560+2560 + 1 ≈ 2.6M
    count = channel.param_count()
    assert 2_500_000 < count < 4_000_000  # ~3.3M total
