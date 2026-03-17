"""Model weight serialization for network transport.

Extracted from the original server.py / client.py — base64-encoded PyTorch
state_dict over JSON. This is the same format used by compute nodes.
"""

import base64
import io

import torch


def serialize_state_dict(state_dict: dict) -> str:
    """Encode a PyTorch state_dict to a base64 string for JSON transport."""
    buf = io.BytesIO()
    torch.save(state_dict, buf)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def deserialize_state_dict(encoded: str) -> dict:
    """Decode a base64 string back to a PyTorch state_dict."""
    buf = io.BytesIO(base64.b64decode(encoded.encode("utf-8")))
    return torch.load(buf, map_location="cpu", weights_only=True)


def state_dict_to_bytes(state_dict: dict) -> bytes:
    """Serialize state_dict to raw bytes (for S3 storage)."""
    buf = io.BytesIO()
    torch.save(state_dict, buf)
    buf.seek(0)
    return buf.read()


def bytes_to_state_dict(data: bytes) -> dict:
    """Deserialize raw bytes back to a state_dict."""
    buf = io.BytesIO(data)
    return torch.load(buf, map_location="cpu", weights_only=True)
