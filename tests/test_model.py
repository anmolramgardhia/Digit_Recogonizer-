"""
test_model.py — Unit tests for model, preprocessing, and API
MNIST Digit Recognizer Project

Run with:
    pytest tests/ -v
"""

import sys
import pytest
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from model import DigitCNN, DigitCNNBatchNorm, count_parameters
from dataset import get_transforms, preprocess_image_tensor


# ── Model architecture tests ──────────────────────────────────────────────────

class TestDigitCNN:
    @pytest.fixture
    def model(self):
        return DigitCNN(dropout=0.5).eval()

    @pytest.fixture
    def batch(self):
        return torch.zeros(8, 1, 28, 28)

    def test_output_shape(self, model, batch):
        out = model(batch)
        assert out.shape == (8, 10), f"Expected (8, 10), got {out.shape}"

    def test_output_is_log_probs(self, model, batch):
        out = model(batch)
        probs = torch.exp(out)
        sums = probs.sum(dim=1)
        assert torch.allclose(sums, torch.ones(8), atol=1e-5), "Softmax probs must sum to 1"

    def test_output_values_negative(self, model, batch):
        out = model(batch)
        assert (out <= 0).all(), "log_softmax output must be <= 0"

    def test_parameter_count(self, model):
        params = count_parameters(model)
        assert 200_000 < params < 300_000, f"Unexpected param count: {params}"

    def test_single_image(self, model):
        img = torch.randn(1, 1, 28, 28)
        out = model(img)
        assert out.shape == (1, 10)

    def test_prediction_is_valid_digit(self, model, batch):
        out = model(batch)
        preds = out.argmax(dim=1)
        assert all(0 <= p <= 9 for p in preds.tolist())

    def test_eval_mode_no_dropout_effect(self, model, batch):
        out1 = model(batch)
        out2 = model(batch)
        assert torch.allclose(out1, out2), "Eval mode should be deterministic"

    def test_train_mode_dropout_varies(self, batch):
        model = DigitCNN(dropout=0.5).train()
        out1 = model(batch)
        out2 = model(batch)
        # With dropout in train mode, outputs should differ
        assert not torch.allclose(out1, out2), "Train mode with dropout should be stochastic"


class TestDigitCNNBatchNorm:
    def test_output_shape(self):
        model = DigitCNNBatchNorm().eval()
        batch = torch.zeros(4, 1, 28, 28)
        out = model(batch)
        assert out.shape == (4, 10)

    def test_batchnorm_layers_present(self):
        model = DigitCNNBatchNorm()
        import torch.nn as nn
        bn_layers = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
        assert len(bn_layers) == 2, "Expected 2 BatchNorm2d layers"


# ── Preprocessing tests ───────────────────────────────────────────────────────

class TestTransforms:
    def test_base_transform_normalizes(self):
        from PIL import Image
        import numpy as np
        img = Image.fromarray(np.full((28, 28), 128, dtype=np.uint8))
        transform = get_transforms(augment=False)
        tensor = transform(img)
        assert tensor.shape == (1, 28, 28)
        # After normalize, values should not be in [0,1] raw range
        assert tensor.min() < 0 or tensor.max() > 1, "Normalize should shift pixel range"

    def test_augment_transform_returns_same_shape(self):
        from PIL import Image
        import numpy as np
        img = Image.fromarray(np.zeros((28, 28), dtype=np.uint8))
        transform = get_transforms(augment=True)
        tensor = transform(img)
        assert tensor.shape == (1, 28, 28)

    def test_preprocess_image_tensor_shape(self):
        from PIL import Image
        import numpy as np
        img = Image.fromarray(np.full((28, 28), 200, dtype=np.uint8))
        tensor = preprocess_image_tensor(img)
        assert tensor.shape == (1, 1, 28, 28), f"Expected (1,1,28,28), got {tensor.shape}"

    def test_preprocess_resizes_larger_image(self):
        from PIL import Image
        import numpy as np
        img = Image.fromarray(np.zeros((100, 100), dtype=np.uint8))
        tensor = preprocess_image_tensor(img)
        assert tensor.shape == (1, 1, 28, 28)


# ── Integration test (requires trained model) ─────────────────────────────────

class TestIntegration:
    @pytest.fixture(scope="class")
    def model(self):
        try:
            from train import load_model
            return load_model()
        except FileNotFoundError:
            pytest.skip("Trained model not found. Run train.py first.")

    def test_predicts_without_error(self, model):
        batch = torch.zeros(1, 1, 28, 28)
        out = model(batch)
        assert out.shape == (1, 10)

    def test_prediction_in_valid_range(self, model):
        batch = torch.randn(4, 1, 28, 28)
        out = model(batch)
        preds = out.argmax(dim=1)
        assert all(0 <= p <= 9 for p in preds.tolist())

    def test_batch_inference(self, model):
        batch = torch.randn(64, 1, 28, 28)
        out = model(batch)
        assert out.shape == (64, 10)
