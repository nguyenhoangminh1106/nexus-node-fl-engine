"""Model registry — maps string model names to their definitions.

When creating a training job, clients specify a model_type string (e.g.
"plantnet"). The orchestrator uses this registry to build the model,
get transforms, and configure freezing.

To add a new model:
  1. Create a class implementing BaseModelDef in nexus/models/
  2. Register it here with register()
"""

from nexus.models.base import BaseModelDef
from nexus.models.plantnet import PlantNetDef

_registry: dict[str, BaseModelDef] = {}


def register(name: str, model_def: BaseModelDef) -> None:
    _registry[name] = model_def


def get(name: str) -> BaseModelDef:
    if name not in _registry:
        available = ", ".join(sorted(_registry.keys()))
        raise KeyError(f"Unknown model type '{name}'. Available: {available}")
    return _registry[name]


def list_models() -> list[str]:
    return sorted(_registry.keys())


# Register built-in models
register("plantnet", PlantNetDef())
