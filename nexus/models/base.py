"""Abstract base for ML model definitions in the Nexus model registry.

Every model that can be trained via federated learning must implement this
interface so the orchestrator knows how to build, freeze, and transform data.
"""

from abc import ABC, abstractmethod

import torch.nn as nn
from torchvision import transforms


class BaseModelDef(ABC):
    """Abstract definition for a federated-learning-compatible model."""

    @abstractmethod
    def build(self, num_classes: int, pretrained: bool = True) -> nn.Module:
        """Construct and return a fresh nn.Module instance."""

    @abstractmethod
    def freeze_for_client(self, model: nn.Module) -> None:
        """Freeze heavy layers for fast CPU training on compute nodes."""

    @abstractmethod
    def unfreeze_all(self, model: nn.Module) -> None:
        """Unfreeze all parameters (full training on server/GPU nodes)."""

    @abstractmethod
    def train_transform(self) -> transforms.Compose:
        """Return the data augmentation pipeline for training."""

    @abstractmethod
    def val_transform(self) -> transforms.Compose:
        """Return the transform pipeline for validation/inference."""
