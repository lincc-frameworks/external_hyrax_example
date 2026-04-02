"""Contract tests for Galaxy10Dataset.

This file shows how to use HyraxDatasetContractTests to verify a dataset
implementation.  Copy the pattern here when writing contracts for a new dataset:

1. Set ``getter_names`` to the list of getter methods your dataset implements.
2. Provide a ``dataset`` fixture that returns an instantiated dataset.
   Mock any external dependencies (files, databases, etc.) inside the fixture.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from external_hyrax_example.datasets.galaxy10_dataset import Galaxy10Dataset
from hyrax_contract_helpers.dataset_contracts import HyraxDatasetContractTests

# ---------------------------------------------------------------------------
# Synthetic data used by the fixture below
# ---------------------------------------------------------------------------
_N_SAMPLES = 5
_IMAGE_SHAPE = (_N_SAMPLES, 69, 69, 3)  # (N, H, W, C) as stored in the HDF5 file

_rng = np.random.default_rng(0)
_FAKE_IMAGES = _rng.integers(0, 256, _IMAGE_SHAPE, dtype=np.uint8)
_FAKE_LABELS = _rng.integers(0, 10, (_N_SAMPLES,))


# ---------------------------------------------------------------------------
# Contract test class
# ---------------------------------------------------------------------------


class TestGalaxy10DatasetContracts(HyraxDatasetContractTests):
    """Verify Galaxy10Dataset satisfies the Hyrax dataset interface."""

    getter_names = ["get_image", "get_label"]

    @pytest.fixture
    def dataset(self, tmp_path):
        config = {
            "external_hyrax_example": {
                "galaxy10_dataset": {"channels_first": False},
            }
        }

        mock_h5 = MagicMock()
        mock_h5.__enter__.return_value = mock_h5
        mock_h5.__getitem__.side_effect = lambda key: {
            "images": _FAKE_IMAGES,
            "ans": _FAKE_LABELS,
        }[key]

        with patch("h5py.File", return_value=mock_h5):
            yield Galaxy10Dataset(config, data_location=tmp_path)
