"""Contract tests for VGG11.

This file shows how to use HyraxModelContractTests to verify a model
implementation.  Copy the pattern here when writing contracts for a new model:

1. Set ``has_validate_batch`` / ``has_test_batch`` based on what your model
   implements.
2. Provide ``model_class``, ``data_dict``, and ``model`` fixtures.

The ``config`` fixture below is a minimal config that satisfies both VGG11 and
the @hyrax_model decorator (which reads ``optimizer``, ``criterion``, and
``scheduler`` keys to inject them onto the model).
"""

import pytest
import torch

from external_hyrax_example.models.vgg11 import VGG11
from hyrax_contract_helpers.model_contracts import HyraxModelContractTests

# ---------------------------------------------------------------------------
# Shared config fixture
# ---------------------------------------------------------------------------

# Batch size and spatial dimensions for synthetic test tensors.
# The images must be channels-first (C, H, W) because VGG11's __init__
# unpacks the batch as (batch, channels, width, height).
_BATCH = 2
_C, _H, _W = 3, 69, 69
_NUM_CLASSES = 10


@pytest.fixture
def _vgg11_config():
    return {
        "external_hyrax_example": {
            "VGG11": {
                "dropout": 0.0,  # disable dropout for deterministic tests
                "num_classes": _NUM_CLASSES,
                "batch_norm": False,
            }
        },
        # The @hyrax_model decorator reads these keys to inject optimizer /
        # criterion / scheduler onto the model after __init__ runs.
        "optimizer": {"name": "torch.optim.SGD"},
        "torch.optim.SGD": {"lr": 0.01},
        "criterion": {"name": "torch.nn.CrossEntropyLoss"},
        "scheduler": {"name": None},
    }


# ---------------------------------------------------------------------------
# Contract test class
# ---------------------------------------------------------------------------


class TestVGG11Contracts(HyraxModelContractTests):
    """Verify VGG11 satisfies the Hyrax model interface."""

    has_validate_batch = True
    has_test_batch = True

    @pytest.fixture
    def model_class(self):
        return VGG11

    @pytest.fixture
    def data_dict(self):
        torch.manual_seed(0)
        return {
            "data": {
                "image": torch.rand(_BATCH, _C, _H, _W),
                "label": torch.randint(0, _NUM_CLASSES, (_BATCH,)),
            },
            "object_id": [f"obj_{i}" for i in range(_BATCH)],
        }

    @pytest.fixture
    def model(self, model_class, data_dict, _vgg11_config):
        batch = model_class.prepare_inputs(data_dict)
        return model_class(_vgg11_config, data_sample=batch)
