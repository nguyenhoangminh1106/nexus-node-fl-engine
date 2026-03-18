"""Model format conversion — PyTorch to ONNX.

ONNX is the universal interchange format. Mobile apps convert from ONNX to
their native format:
  - Android: ONNX → TFLite (via onnx-tf or ai.onnxruntime)
  - iOS: ONNX → Core ML (via coremltools)
  - Both: can also run ONNX directly via ONNX Runtime Mobile

The server exports ONNX. The mobile app handles the last-mile conversion.
"""

import io
import uuid

import structlog
import torch

from nexus.core import model_registry
from nexus.core.serialization import bytes_to_state_dict
from nexus.storage import s3

logger = structlog.get_logger()


def state_dict_to_onnx(
    model_type: str,
    num_classes: int,
    state_dict: dict,
) -> bytes:
    """Convert a PyTorch state_dict to ONNX format.

    Returns raw ONNX bytes ready for download.
    """
    model_def = model_registry.get(model_type)
    model = model_def.build(num_classes=num_classes, pretrained=False)
    model.load_state_dict(state_dict)
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224)
    buf = io.BytesIO()

    torch.onnx.export(
        model,
        dummy_input,
        buf,
        opset_version=13,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )

    buf.seek(0)
    return buf.read()


def _onnx_cache_key(job_id: uuid.UUID, round_num: int) -> str:
    return f"checkpoints/{job_id}/round_{round_num}.onnx"


def get_or_create_onnx(
    job_id: uuid.UUID,
    round_num: int,
    model_type: str,
    num_classes: int,
    pytorch_checkpoint_path: str,
) -> bytes:
    """Get cached ONNX model or convert from PyTorch checkpoint.

    Caches the ONNX version in S3 so conversion only happens once per round.
    """
    onnx_key = _onnx_cache_key(job_id, round_num)

    # Try cache first
    try:
        return s3.download_bytes(onnx_key)
    except Exception:
        pass

    # Convert from PyTorch
    pytorch_bytes = s3.download_bytes(pytorch_checkpoint_path)
    sd = bytes_to_state_dict(pytorch_bytes)
    onnx_bytes = state_dict_to_onnx(model_type, num_classes, sd)

    # Cache for future requests
    s3.upload_bytes(onnx_key, onnx_bytes)
    logger.info("onnx_converted", job_id=str(job_id), round=round_num, size=len(onnx_bytes))

    return onnx_bytes
