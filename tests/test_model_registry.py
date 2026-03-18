"""Tests for the model registry."""

import pytest
import torch.nn as nn

from nexus.core import model_registry


def test_plantnet_registered():
    assert "plantnet" in model_registry.list_models()


def test_build_plantnet():
    model_def = model_registry.get("plantnet")
    model = model_def.build(num_classes=10, pretrained=False)
    assert isinstance(model, nn.Module)


def test_unknown_model_raises():
    with pytest.raises(KeyError):
        model_registry.get("nonexistent_model")


def test_freeze_unfreeze():
    model_def = model_registry.get("plantnet")
    model = model_def.build(num_classes=10, pretrained=False)

    total_params = sum(p.numel() for p in model.parameters())

    model_def.freeze_for_client(model)
    frozen_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert frozen_trainable < total_params

    model_def.unfreeze_all(model)
    unfrozen_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert unfrozen_trainable == total_params
