"""PlantNet — MobileNetV2 for plant disease classification.

Preserved from the original model.py. This is the first model in the registry
and serves as the reference implementation for the BaseModelDef interface.
"""

import torch.nn as nn
from torchvision import models, transforms

from nexus.models.base import BaseModelDef


class PlantNet(nn.Module):
    """MobileNetV2 classifier for plant disease detection.

    pretrained=True  -> ImageNet weights (recommended for server / first round).
    pretrained=False -> random init (used when loading a saved checkpoint).

    For CPU-only nodes call freeze_for_client() after loading weights.
    Freezes the heavy backbone and only keeps the last few blocks + classifier
    trainable, giving ~10x speedup with minimal accuracy loss.
    """

    def __init__(self, num_classes: int = 38, pretrained: bool = True):
        super().__init__()
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        backbone = models.mobilenet_v2(weights=weights)

        self.features = backbone.features  # 19-block Sequential
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        in_features = backbone.classifier[1].in_features  # 1280
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return self.classifier(x)

    def freeze_for_client(self, train_last_n_blocks: int = 3):
        """Freeze all except the last N feature blocks + classifier.

        Default: freeze features[0:16], train features[16:18] + classifier.
        Trainable param count drops from ~3.4M to ~0.3M.
        """
        for p in self.parameters():
            p.requires_grad = False
        children = list(self.features.children())
        for block in children[-train_last_n_blocks:]:
            for p in block.parameters():
                p.requires_grad = True
        for p in self.classifier.parameters():
            p.requires_grad = True

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad = True

    def count_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class PlantNetDef(BaseModelDef):
    """Registry entry for PlantNet (MobileNetV2)."""

    def build(self, num_classes: int, pretrained: bool = True) -> nn.Module:
        return PlantNet(num_classes=num_classes, pretrained=pretrained)

    def freeze_for_client(self, model: nn.Module) -> None:
        model.freeze_for_client()

    def unfreeze_all(self, model: nn.Module) -> None:
        model.unfreeze_all()

    def train_transform(self) -> transforms.Compose:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def val_transform(self) -> transforms.Compose:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
