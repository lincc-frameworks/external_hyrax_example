"""Tests verifying that Galaxy10Dataset satisfies the HyraxDataset interface contracts.

These tests use a mocked h5py.File so they run without the real Galaxy10.h5 file.
They serve as a template: replace Galaxy10Dataset with your own converted dataset
class to verify it meets the same contracts.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from hyrax.datasets import HyraxDataset

from external_hyrax_example.datasets.galaxy10_dataset import Galaxy10Dataset

N_SAMPLES = 8
IMG_H, IMG_W, IMG_C = 32, 32, 3


def _make_mock_h5(images, labels):
    """Return a context-manager mock that serves images and labels like h5py.File."""
    mock_f = {"images": images, "ans": labels}
    mock_ctx = MagicMock()
    mock_ctx.__enter__ = MagicMock(return_value=mock_f)
    mock_ctx.__exit__ = MagicMock(return_value=False)
    return mock_ctx


@pytest.fixture
def config():
    """Minimal config dict matching the structure Hyrax injects at runtime."""
    return {"external_hyrax_example": {"galaxy10_dataset": {"channels_first": False}}}


@pytest.fixture
def dataset(config, tmp_path):
    """Galaxy10Dataset backed by synthetic in-memory arrays instead of a real HDF5 file."""
    images = np.random.randint(0, 256, (N_SAMPLES, IMG_H, IMG_W, IMG_C), dtype=np.uint8)
    labels = np.random.randint(0, 10, (N_SAMPLES,))
    with patch("h5py.File", return_value=_make_mock_h5(images, labels)):
        return Galaxy10Dataset(config, tmp_path)


# ---------------------------------------------------------------------------
# Structural contracts
# ---------------------------------------------------------------------------


def test_inherits_from_hyrax_dataset(dataset):
    """super().__init__(config) must have been called for isinstance to pass."""
    assert isinstance(dataset, HyraxDataset)


def test_no_custom_getitem():
    """__getitem__ must not be overridden; HyraxDataset provides it.

    Defining __getitem__ on a subclass bypasses the DataProvider pipeline.
    """
    assert "__getitem__" not in Galaxy10Dataset.__dict__


# ---------------------------------------------------------------------------
# __len__
# ---------------------------------------------------------------------------


def test_len_returns_int(dataset):
    """__len__ must return an int."""
    assert isinstance(len(dataset), int)


def test_len_matches_sample_count(dataset):
    """__len__ must match the number of samples loaded."""
    assert len(dataset) == N_SAMPLES


# ---------------------------------------------------------------------------
# get_object_id
# ---------------------------------------------------------------------------


def test_get_object_id_returns_str(dataset):
    """get_object_id must return a str, not an int or None."""
    assert isinstance(dataset.get_object_id(0), str)


def test_get_object_id_unique(dataset):
    """Each index must produce a distinct object ID."""
    ids = [dataset.get_object_id(i) for i in range(N_SAMPLES)]
    assert len(set(ids)) == N_SAMPLES


# ---------------------------------------------------------------------------
# get_image
# ---------------------------------------------------------------------------


def test_get_image_returns_ndarray(dataset):
    """get_image must return a numpy ndarray."""
    assert isinstance(dataset.get_image(0), np.ndarray)


def test_get_image_dtype_float32(dataset):
    """Images must be float32."""
    assert dataset.get_image(0).dtype == np.float32


def test_get_image_values_normalized(dataset):
    """Image pixel values must be in [0.0, 1.0]."""
    img = dataset.get_image(0)
    assert img.min() >= 0.0, f"min value {img.min()} is below 0"
    assert img.max() <= 1.0, f"max value {img.max()} is above 1"


def test_get_image_shape_channels_last(dataset):
    """Default (channels_first=False) images should have shape (H, W, C)."""
    img = dataset.get_image(0)
    assert img.shape == (IMG_H, IMG_W, IMG_C)


def test_get_image_shape_channels_first(config, tmp_path):
    """With channels_first=True images should have shape (C, H, W)."""
    config["external_hyrax_example"]["galaxy10_dataset"]["channels_first"] = True
    images = np.random.randint(0, 256, (N_SAMPLES, IMG_H, IMG_W, IMG_C), dtype=np.uint8)
    labels = np.random.randint(0, 10, (N_SAMPLES,))
    with patch("h5py.File", return_value=_make_mock_h5(images, labels)):
        ds = Galaxy10Dataset(config, tmp_path)
    img = ds.get_image(0)
    assert img.shape == (IMG_C, IMG_H, IMG_W)


# ---------------------------------------------------------------------------
# get_label
# ---------------------------------------------------------------------------


def test_get_label_not_none(dataset):
    """get_label must return a value (not None) for supervised datasets."""
    assert dataset.get_label(0) is not None
