"""Inference pipeline for model predictions."""
from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image


class InferencePipeline:
    """End-to-end inference pipeline: preprocess → predict → postprocess."""

    def __init__(self, model_path: str | Path, device: str = "auto"):
        self.model_path = Path(model_path)
        self.model = None
        self.model_name = None
        self.device = device
        self._load_model()

    def _load_model(self) -> None:
        """Load model checkpoint."""
        try:
            import torch
            if self.device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"

            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.model_name = checkpoint.get("architecture", "unknown")

            from src.models.registry import ModelRegistry
            self.model = ModelRegistry.create(
                self.model_name.lower(),
                num_classes=checkpoint.get("num_classes", 1000),
            )
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {self.model_path}: {e}")

    def predict(self, image_input, top_k: int = 5) -> Dict[str, Any]:
        """Run prediction on an image.

        Args:
            image_input: PIL Image, file path, or BytesIO object.
            top_k: Number of top predictions to return.

        Returns:
            Dictionary with prediction results.
        """
        import torch

        image = self._preprocess(image_input)
        image = image.unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(image)
            probs = torch.softmax(logits, dim=-1)
            top_probs, top_indices = probs.topk(top_k, dim=-1)

        top_probs = top_probs.squeeze().cpu().numpy()
        top_indices = top_indices.squeeze().cpu().numpy()

        return {
            "class_index": int(top_indices[0]),
            "class_name": f"class_{top_indices[0]}",
            "confidence": float(top_probs[0]),
            "top_k": [
                {"class_index": int(idx), "class_name": f"class_{idx}",
                 "confidence": float(prob)}
                for idx, prob in zip(top_indices, top_probs)
            ],
        }

    def _preprocess(self, image_input) -> "torch.Tensor":
        """Preprocess image for inference."""
        from src.data.transforms import get_inference_transforms

        if isinstance(image_input, (str, Path)):
            image = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, io.BytesIO):
            image = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            image = image_input.convert("RGB")
        else:
            raise ValueError(f"Unsupported input type: {type(image_input)}")

        transform = get_inference_transforms(image_size=224)
        if transform is not None:
            return transform(image)

        # Fallback: manual numpy preprocessing
        import torch
        image = image.resize((224, 224))
        arr = np.array(image).astype(np.float32) / 255.0
        arr = (arr - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        return torch.from_numpy(arr.transpose(2, 0, 1)).float()
