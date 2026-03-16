"""Model registry for creating models by name."""
from __future__ import annotations

from typing import Any, Dict, Type

from src.models.base import BaseModel


class ModelRegistry:
    """Registry for model architectures."""

    _models: Dict[str, Type[BaseModel]] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register a model class."""
        def decorator(model_cls: Type[BaseModel]):
            cls._models[name] = model_cls
            return model_cls
        return decorator

    @classmethod
    def create(cls, name: str, **kwargs) -> BaseModel:
        """Create a model instance by name."""
        if name not in cls._models:
            # Lazy import known models
            cls._import_defaults()

        if name not in cls._models:
            raise ValueError(
                f"Unknown model: {name}. "
                f"Available: {list(cls._models.keys())}"
            )
        return cls._models[name](**kwargs)

    @classmethod
    def list_models(cls) -> list[str]:
        """List all registered model names."""
        cls._import_defaults()
        return list(cls._models.keys())

    @classmethod
    def _import_defaults(cls) -> None:
        """Import default model modules to trigger registration."""
        if cls._models:
            return
        try:
            from src.models.resnet import ResNetClassifier
            cls._models["resnet18"] = lambda **kw: ResNetClassifier(backbone="resnet18", **kw)
            cls._models["resnet50"] = lambda **kw: ResNetClassifier(backbone="resnet50", **kw)
            cls._models["resnet101"] = lambda **kw: ResNetClassifier(backbone="resnet101", **kw)
        except ImportError:
            pass
        try:
            from src.models.transformer import VisionTransformer
            cls._models["vit_base"] = lambda **kw: VisionTransformer(
                embed_dim=768, depth=12, num_heads=12, **kw
            )
            cls._models["vit_small"] = lambda **kw: VisionTransformer(
                embed_dim=384, depth=12, num_heads=6, **kw
            )
        except ImportError:
            pass
