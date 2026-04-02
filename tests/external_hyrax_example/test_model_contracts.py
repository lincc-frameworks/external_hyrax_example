"""Tests verifying that VGG11 satisfies the hyrax_model interface contracts.

These tests use synthetic random tensors so they run without any real dataset.
They serve as a template: replace VGG11 with your own converted model class
to verify it meets the same contracts.
"""

import pytest
import torch

from external_hyrax_example.models.vgg11 import VGG11

BATCH_SIZE = 2
N_CHANNELS = 3
IMG_SIZE = 64
NUM_CLASSES = 10


@pytest.fixture
def config():
    """Minimal config dict matching the structure Hyrax injects at runtime.

    The top-level 'optimizer' and 'criterion' keys are read by the @hyrax_model
    decorator during __init__ to inject self.optimizer and self.criterion.
    """
    return {
        "external_hyrax_example": {
            "VGG11": {
                "dropout": 0.0,  # deterministic for testing
                "num_classes": NUM_CLASSES,
                "batch_norm": False,
            }
        },
        "optimizer": {"name": "torch.optim.SGD"},
        "torch.optim.SGD": {"lr": 0.01},
        "criterion": {"name": "torch.nn.CrossEntropyLoss"},
        "scheduler": {"name": None},
    }


@pytest.fixture
def data_dict():
    """Synthetic data_dict in the shape the DataProvider emits."""
    return {
        "data": {
            "image": torch.rand(BATCH_SIZE, N_CHANNELS, IMG_SIZE, IMG_SIZE),
            "label": torch.randint(0, NUM_CLASSES, (BATCH_SIZE,)),
        },
        "object_id": [str(i) for i in range(BATCH_SIZE)],
    }


@pytest.fixture
def batch(data_dict):
    """Output of prepare_inputs — the tuple that forward/train_batch receive."""
    return VGG11.prepare_inputs(data_dict)


@pytest.fixture
def model(config, batch):
    """Instantiated VGG11; @hyrax_model injects self.optimizer and self.criterion from config."""
    m = VGG11(config, data_sample=batch)
    m.eval()
    return m


# ---------------------------------------------------------------------------
# prepare_inputs
# ---------------------------------------------------------------------------


def test_prepare_inputs_returns_tuple(data_dict):
    """prepare_inputs must return a tuple."""
    result = VGG11.prepare_inputs(data_dict)
    assert isinstance(result, tuple)


def test_prepare_inputs_is_staticmethod():
    """prepare_inputs must be a staticmethod (callable without an instance)."""
    assert isinstance(inspect_staticmethod(VGG11, "prepare_inputs"), staticmethod)


def inspect_staticmethod(cls, name):
    """Helper to inspect the raw descriptor of a class attribute."""
    return cls.__dict__.get(name)


def test_prepare_inputs_first_element_is_image(data_dict):
    """First element of the returned tuple should be the image tensor."""
    images, _ = VGG11.prepare_inputs(data_dict)
    assert images.shape == (BATCH_SIZE, N_CHANNELS, IMG_SIZE, IMG_SIZE)


def test_prepare_inputs_second_element_is_label(data_dict):
    """Second element of the returned tuple should be the label tensor."""
    _, labels = VGG11.prepare_inputs(data_dict)
    assert labels.shape == (BATCH_SIZE,)


def test_prepare_inputs_missing_data_key_raises():
    """prepare_inputs must raise if the 'data' key is absent."""
    with pytest.raises(RuntimeError):
        VGG11.prepare_inputs({"object_id": ["0"]})


def test_prepare_inputs_no_label(data_dict):
    """prepare_inputs must tolerate a data_dict with no label key."""
    del data_dict["data"]["label"]
    images, label = VGG11.prepare_inputs(data_dict)
    assert label is None


# ---------------------------------------------------------------------------
# forward
# ---------------------------------------------------------------------------


def test_forward_output_shape(model, batch):
    """forward must return a tensor of shape (batch_size, num_classes)."""
    with torch.no_grad():
        output = model(batch)
    assert output.shape == (BATCH_SIZE, NUM_CLASSES)


def test_forward_returns_tensor(model, batch):
    """forward must return a torch.Tensor."""
    with torch.no_grad():
        output = model(batch)
    assert isinstance(output, torch.Tensor)


# ---------------------------------------------------------------------------
# train_batch
# ---------------------------------------------------------------------------


def test_train_batch_returns_dict(model, batch):
    """train_batch must return a dict."""
    model.train()
    result = model.train_batch(batch)
    assert isinstance(result, dict)


def test_train_batch_has_loss_key(model, batch):
    """train_batch return dict must contain a 'loss' key."""
    model.train()
    result = model.train_batch(batch)
    assert "loss" in result


def test_train_batch_loss_is_finite(model, batch):
    """Loss value must be a finite float."""
    model.train()
    result = model.train_batch(batch)
    assert isinstance(result["loss"], float)
    assert torch.isfinite(torch.tensor(result["loss"]))


# ---------------------------------------------------------------------------
# validate_batch
# ---------------------------------------------------------------------------


def test_validate_batch_returns_dict(model, batch):
    """validate_batch must return a dict."""
    result = model.validate_batch(batch)
    assert isinstance(result, dict)


def test_validate_batch_has_loss_key(model, batch):
    """validate_batch return dict must contain a 'loss' key."""
    result = model.validate_batch(batch)
    assert "loss" in result


# ---------------------------------------------------------------------------
# test_batch
# ---------------------------------------------------------------------------


def test_test_batch_returns_dict(model, batch):
    """test_batch must return a dict."""
    result = model.test_batch(batch)
    assert isinstance(result, dict)


def test_test_batch_has_loss_key(model, batch):
    """test_batch return dict must contain a 'loss' key."""
    result = model.test_batch(batch)
    assert "loss" in result


# ---------------------------------------------------------------------------
# infer_batch
# ---------------------------------------------------------------------------


def test_infer_batch_returns_tensor(model, batch):
    """infer_batch must return a tensor."""
    with torch.no_grad():
        result = model.infer_batch(batch)
    assert isinstance(result, torch.Tensor)


def test_infer_batch_output_shape(model, batch):
    """infer_batch output shape should match (batch_size, num_classes)."""
    with torch.no_grad():
        result = model.infer_batch(batch)
    assert result.shape == (BATCH_SIZE, NUM_CLASSES)
