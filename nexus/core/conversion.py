"""Model format conversion — PyTorch to ONNX and Core ML.

ONNX is the universal interchange format. Mobile apps convert from ONNX to
their native format:
  - Android: ONNX → TFLite (via onnx-tf or ai.onnxruntime)
  - iOS: ONNX → Core ML (via coremltools)
  - Both: can also run ONNX directly via ONNX Runtime Mobile

The server also exports Core ML updatable models for on-device training (iOS).
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


# ── Core ML (updatable) ─────────────────────────────────────────────────────


def state_dict_to_coreml(
    model_type: str,
    num_classes: int,
    state_dict: dict,
    updatable: bool = True,
    tier: int = 2,
) -> bytes:
    """Convert a PyTorch state_dict to Core ML format.

    If updatable=True, marks the model as updatable for on-device training:
      - Tier 2: only the classifier (last FC layer) is updatable
      - Tier 3: all layers are updatable

    Returns raw .mlmodel bytes ready for download.
    """
    import tempfile
    import coremltools as ct
    from coremltools.models.neural_network import NeuralNetworkBuilder, SgdParams

    # Build PyTorch model and load weights
    model_def = model_registry.get(model_type)
    model = model_def.build(num_classes=num_classes, pretrained=False)
    model.load_state_dict(state_dict)
    model.eval()

    # Convert PyTorch → Core ML via tracing
    # Must use neuralnetwork format (not mlprogram) for updatable model support
    dummy_input = torch.randn(1, 3, 224, 224)
    traced = torch.jit.trace(model, dummy_input)

    ml_model = ct.convert(
        traced,
        inputs=[ct.ImageType(name="image", shape=(1, 3, 224, 224), scale=1.0 / 255.0)],
        classifier_config=ct.ClassifierConfig(
            class_labels=[str(i) for i in range(num_classes)]
        ),
        convert_to="neuralnetwork",
    )

    if updatable:
        spec = ml_model.get_spec()
        builder = NeuralNetworkBuilder(spec=spec)

        # Determine which layers to make updatable
        # neuralNetworkClassifier for classifier models, neuralNetwork otherwise
        nn_spec = spec.neuralNetworkClassifier if spec.HasField("neuralNetworkClassifier") else spec.neuralNetwork
        layer_names = [layer.name for layer in nn_spec.layers]

        if tier == 2:
            # Tier 2: only the last FC/classifier layer
            updatable_layers = [name for name in layer_names if "classifier" in name.lower() or "fc" in name.lower()]
            if not updatable_layers:
                # Fallback: make last layer updatable
                updatable_layers = [layer_names[-1]] if layer_names else []
        else:
            # Tier 3: all layers
            updatable_layers = layer_names

        if updatable_layers:
            builder.make_updatable(updatable_layers)

            # Add softmax layer before loss if not present
            # Find the last layer's output to connect softmax
            last_layer = nn_spec.layers[-1]
            last_output = last_layer.output[0] if last_layer.output else "output"

            # Add softmax
            builder.add_softmax(
                name="softmax_output",
                input_name=last_output,
                output_name="classProbabilities",
            )

            # Set loss function — must reference softmax output
            builder.set_categorical_cross_entropy_loss(
                name="lossLayer",
                input="classProbabilities",
            )

            # Set optimizer
            builder.set_sgd_optimizer(SgdParams(lr=0.001, batch=8, momentum=0.9))
            builder.set_epochs(10, allowed_set=[1, 2, 5, 10, 20])

            spec = builder.spec

        # Save to temp file and read bytes
        with tempfile.NamedTemporaryFile(suffix=".mlmodel", delete=True) as f:
            ct.utils.save_spec(spec, f.name)
            f.seek(0)
            return f.read()
    else:
        # Non-updatable: just save directly
        with tempfile.NamedTemporaryFile(suffix=".mlmodel", delete=True) as f:
            ml_model.save(f.name)
            f.seek(0)
            return f.read()


def _coreml_cache_key(job_id: uuid.UUID, round_num: int, tier: int) -> str:
    return f"checkpoints/{job_id}/round_{round_num}_tier{tier}.mlmodel"


def get_or_create_coreml(
    job_id: uuid.UUID,
    round_num: int,
    model_type: str,
    num_classes: int,
    pytorch_checkpoint_path: str,
    tier: int = 2,
) -> bytes:
    """Get cached Core ML updatable model or convert from PyTorch checkpoint.

    Caches the Core ML version in S3 so conversion only happens once per round+tier.
    """
    coreml_key = _coreml_cache_key(job_id, round_num, tier)

    # Try cache first
    try:
        return s3.download_bytes(coreml_key)
    except Exception:
        pass

    # Convert from PyTorch
    pytorch_bytes = s3.download_bytes(pytorch_checkpoint_path)
    sd = bytes_to_state_dict(pytorch_bytes)
    coreml_bytes = state_dict_to_coreml(
        model_type, num_classes, sd, updatable=True, tier=tier,
    )

    # Cache for future requests
    s3.upload_bytes(coreml_key, coreml_bytes)
    logger.info(
        "coreml_converted",
        job_id=str(job_id),
        round=round_num,
        tier=tier,
        size=len(coreml_bytes),
    )

    return coreml_bytes
