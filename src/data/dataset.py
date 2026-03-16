"""Dataset classes for image classification."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Tuple

from PIL import Image


class ImageClassificationDataset:
    """Generic image classification dataset.

    Expects directory structure: root/class_name/image_file.jpg
    """

    VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

    def __init__(self, root: str | Path, transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None):
        self.root = Path(root)
        self.transform = transform
        self.target_transform = target_transform
        self.samples: list[Tuple[Path, int]] = []
        self.class_to_idx: dict[str, int] = {}
        self._scan_directory()

    def _scan_directory(self) -> None:
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.root}")
        classes = sorted(d.name for d in self.root.iterdir() if d.is_dir())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        for cls_name, cls_idx in self.class_to_idx.items():
            cls_dir = self.root / cls_name
            for img_path in sorted(cls_dir.iterdir()):
                if img_path.suffix.lower() in self.VALID_EXTENSIONS:
                    self.samples.append((img_path, cls_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple:
        img_path, target = self.samples[index]
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return image, target

    @property
    def num_classes(self) -> int:
        return len(self.class_to_idx)

    @property
    def classes(self) -> list[str]:
        return list(self.class_to_idx.keys())
