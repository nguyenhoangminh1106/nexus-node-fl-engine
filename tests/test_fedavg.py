"""Tests for the FedAvg aggregation algorithm."""

import pytest
import torch

from nexus.core.fedavg import weighted_fedavg


def _make_sd(val: float, keys: list[str] = None) -> dict:
    """Create a simple state_dict with uniform values."""
    keys = keys or ["layer.weight", "layer.bias"]
    return {k: torch.full((4,), val) for k in keys}


def test_single_submission():
    """FedAvg with one node should return that node's weights unchanged."""
    sd = _make_sd(1.0)
    result = weighted_fedavg([(sd, 100)])
    for key in sd:
        assert torch.allclose(result[key], sd[key])


def test_equal_weights():
    """Two nodes with equal samples should produce the average."""
    sd1 = _make_sd(2.0)
    sd2 = _make_sd(4.0)
    result = weighted_fedavg([(sd1, 100), (sd2, 100)])
    for key in result:
        assert torch.allclose(result[key], torch.full((4,), 3.0))


def test_weighted_average():
    """Weights should be proportional to sample counts."""
    sd1 = _make_sd(0.0)
    sd2 = _make_sd(10.0)
    # sd1 has 900 samples, sd2 has 100 → result should be close to 1.0
    result = weighted_fedavg([(sd1, 900), (sd2, 100)])
    for key in result:
        assert torch.allclose(result[key], torch.full((4,), 1.0))


def test_three_nodes():
    """FedAvg with three nodes."""
    sd1 = _make_sd(3.0)
    sd2 = _make_sd(6.0)
    sd3 = _make_sd(9.0)
    result = weighted_fedavg([(sd1, 100), (sd2, 100), (sd3, 100)])
    for key in result:
        assert torch.allclose(result[key], torch.full((4,), 6.0))


def test_empty_raises():
    with pytest.raises(ValueError):
        weighted_fedavg([])
