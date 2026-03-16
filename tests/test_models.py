"""Tests for model architectures."""
import pytest


class TestResNetClassifier:
    def test_resnet18_forward(self, sample_image_batch, sample_config):
        try:
            import torch
            from src.models.resnet import ResNetClassifier

            model = ResNetClassifier(
                num_classes=sample_config["model"]["num_classes"],
                backbone="resnet18",
                pretrained=False,
            )
            model.eval()
            with torch.no_grad():
                output = model(sample_image_batch)

            assert output.shape == (4, sample_config["model"]["num_classes"])
        except ImportError:
            pytest.skip("torch not available")

    def test_freeze_backbone(self):
        try:
            from src.models.resnet import ResNetClassifier

            model = ResNetClassifier(num_classes=10, backbone="resnet18", pretrained=False)
            model.freeze_backbone()

            trainable = sum(1 for p in model.parameters() if p.requires_grad)
            total = sum(1 for p in model.parameters())
            assert trainable < total
        except ImportError:
            pytest.skip("torch not available")

    def test_invalid_backbone(self):
        try:
            from src.models.resnet import ResNetClassifier
            with pytest.raises(ValueError, match="Unknown backbone"):
                ResNetClassifier(num_classes=10, backbone="invalid_model")
        except ImportError:
            pytest.skip("torch not available")


class TestVisionTransformer:
    def test_vit_forward(self, sample_image_batch):
        try:
            import torch
            from src.models.transformer import VisionTransformer

            model = VisionTransformer(
                num_classes=10, img_size=32, patch_size=8,
                embed_dim=64, depth=2, num_heads=4,
            )
            model.eval()
            with torch.no_grad():
                output = model(sample_image_batch)

            assert output.shape == (4, 10)
        except ImportError:
            pytest.skip("torch not available")


class TestModelRegistry:
    def test_list_models(self):
        from src.models.registry import ModelRegistry
        models = ModelRegistry.list_models()
        assert isinstance(models, list)

    def test_create_unknown_model(self):
        from src.models.registry import ModelRegistry
        with pytest.raises(ValueError, match="Unknown model"):
            ModelRegistry.create("nonexistent_model_xyz")
