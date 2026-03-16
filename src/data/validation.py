"""Data validation utilities."""
from __future__ import annotations

from pathlib import Path

from PIL import Image

from src.utils.logging import get_logger

logger = get_logger(__name__)


class DataValidator:
    """Validate image datasets for training readiness."""

    VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def validate(self) -> dict:
        self.errors.clear()
        self.warnings.clear()
        self._check_directory_structure()
        self._check_image_integrity()
        self._check_class_balance()
        return {"valid": len(self.errors) == 0, "errors": self.errors.copy(),
                "warnings": self.warnings.copy()}

    def _check_directory_structure(self) -> None:
        if not self.data_dir.exists():
            self.errors.append(f"Data directory does not exist: {self.data_dir}")
            return
        splits = [d for d in self.data_dir.iterdir() if d.is_dir()]
        if not splits:
            self.errors.append("No split directories found")

    def _check_image_integrity(self) -> None:
        for img_path in self.data_dir.rglob("*"):
            if img_path.suffix.lower() not in self.VALID_EXTENSIONS:
                continue
            try:
                with Image.open(img_path) as img:
                    img.verify()
            except Exception as e:
                self.errors.append(f"Corrupt image {img_path}: {e}")

    def _check_class_balance(self, threshold: float = 10.0) -> None:
        for split_dir in self.data_dir.iterdir():
            if not split_dir.is_dir():
                continue
            counts = {}
            for cls_dir in split_dir.iterdir():
                if cls_dir.is_dir():
                    counts[cls_dir.name] = sum(
                        1 for f in cls_dir.iterdir()
                        if f.suffix.lower() in self.VALID_EXTENSIONS
                    )
            if counts:
                mx, mn = max(counts.values()), min(counts.values())
                if mn > 0 and mx / mn > threshold:
                    self.warnings.append(f"Class imbalance in {split_dir.name}/: {mx/mn:.1f}x")
