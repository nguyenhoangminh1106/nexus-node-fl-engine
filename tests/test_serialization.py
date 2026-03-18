"""Tests for model weight serialization."""

import torch

from nexus.core.serialization import (
    bytes_to_state_dict,
    deserialize_state_dict,
    serialize_state_dict,
    state_dict_to_bytes,
)


def test_base64_roundtrip():
    """Serialize → deserialize should produce identical state_dict."""
    original = {"weight": torch.randn(10, 5), "bias": torch.randn(5)}
    encoded = serialize_state_dict(original)
    decoded = deserialize_state_dict(encoded)
    for key in original:
        assert torch.equal(original[key], decoded[key])


def test_bytes_roundtrip():
    """state_dict → bytes → state_dict should be identical."""
    original = {"w": torch.randn(3, 3)}
    data = state_dict_to_bytes(original)
    assert isinstance(data, bytes)
    restored = bytes_to_state_dict(data)
    assert torch.equal(original["w"], restored["w"])
