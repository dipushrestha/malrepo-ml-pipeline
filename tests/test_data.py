"""Tests for data processing pipeline."""
import pytest
from pathlib import Path


class TestImageClassificationDataset:
    def test_nonexistent_dir_raises(self):
        from src.data.dataset import ImageClassificationDataset
        with pytest.raises(FileNotFoundError):
            ImageClassificationDataset("/nonexistent/path")

    def test_dataset_with_mock_data(self, tmp_data_dir):
        """Test dataset loading with mock directory structure."""
        # Create mock class dirs with dummy images
        from PIL import Image
        import numpy as np

        for cls in ["cat", "dog"]:
            cls_dir = tmp_data_dir / "train" / cls
            cls_dir.mkdir(parents=True, exist_ok=True)
            for i in range(3):
                img = Image.fromarray(
                    np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
                )
                img.save(cls_dir / f"img_{i}.jpg")

        from src.data.dataset import ImageClassificationDataset
        dataset = ImageClassificationDataset(tmp_data_dir / "train")

        assert len(dataset) == 6
        assert dataset.num_classes == 2
        assert "cat" in dataset.classes
        assert "dog" in dataset.classes


class TestDataValidator:
    def test_validate_missing_dir(self):
        from src.data.validation import DataValidator
        validator = DataValidator("/nonexistent")
        report = validator.validate()
        assert not report["valid"]

    def test_validate_empty_dir(self, tmp_path):
        from src.data.validation import DataValidator
        validator = DataValidator(tmp_path)
        report = validator.validate()
        assert not report["valid"]


class TestTransforms:
    def test_train_transforms(self):
        from src.data.transforms import get_train_transforms
        t = get_train_transforms(224)
        # May be None if torchvision not installed
        if t is not None:
            from PIL import Image
            import numpy as np
            img = Image.fromarray(
                np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            )
            result = t(img)
            assert result.shape == (3, 224, 224)
