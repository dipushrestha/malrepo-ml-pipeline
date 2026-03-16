"""Tests for training components."""
import pytest


class TestTrainConfig:
    def test_default_config(self):
        from src.training.trainer import TrainConfig
        config = TrainConfig()
        assert config.epochs == 100
        assert config.learning_rate == 1e-3
        assert config.optimizer == "adamw"


class TestLossFunctions:
    def test_label_smoothing(self):
        try:
            import torch
            from src.training.losses import LabelSmoothingLoss

            loss_fn = LabelSmoothingLoss(num_classes=10, smoothing=0.1)
            pred = torch.randn(4, 10)
            target = torch.randint(0, 10, (4,))
            loss = loss_fn(pred, target)
            assert loss.item() > 0
        except ImportError:
            pytest.skip("torch not available")

    def test_focal_loss(self):
        try:
            import torch
            from src.training.losses import FocalLoss

            loss_fn = FocalLoss(alpha=1.0, gamma=2.0)
            pred = torch.randn(4, 10)
            target = torch.randint(0, 10, (4,))
            loss = loss_fn(pred, target)
            assert loss.item() > 0
        except ImportError:
            pytest.skip("torch not available")

    def test_build_loss_unknown(self):
        from src.training.losses import build_loss
        with pytest.raises(ValueError, match="Unknown loss"):
            build_loss("unknown_loss")


class TestSchedulers:
    def test_warmup_cosine(self):
        try:
            import torch
            from src.training.schedulers import WarmupCosineScheduler

            model = torch.nn.Linear(10, 10)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
            scheduler = WarmupCosineScheduler(optimizer, warmup_epochs=5, total_epochs=100)

            lrs = []
            for _ in range(20):
                lrs.append(optimizer.param_groups[0]["lr"])
                scheduler.step()

            # LR should increase during warmup then decrease
            assert lrs[0] < lrs[4]  # warmup phase
            assert lrs[5] >= lrs[19]  # cosine decay
        except ImportError:
            pytest.skip("torch not available")


class TestCallbacks:
    def test_print_callback(self, capsys):
        from src.training.callbacks import PrintCallback
        cb = PrintCallback()
        history = {"train_loss": [0.5], "train_acc": [0.8], "val_loss": [0.6], "val_acc": [0.75]}
        cb(0, history)
        captured = capsys.readouterr()
        assert "Epoch" in captured.out
        assert "0.5" in captured.out
