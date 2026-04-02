"""Reusable contract tests for @hyrax_model classes.

Copy this file into your project's test directory and subclass
HyraxModelContractTests to verify your model implementation.

Example
-------
::

    class TestMyModel(HyraxModelContractTests):
        has_validate_batch = True
        has_test_batch = False

        @pytest.fixture
        def model_class(self):
            return MyModel

        @pytest.fixture
        def data_dict(self):
            return {
                "data": {"spectrum": torch.rand(2, 256), "label": torch.randint(0, 5, (2,))},
                "object_id": ["obj_0", "obj_1"],
            }

        @pytest.fixture
        def model(self, model_class, data_dict):
            config = {
                "my_package": {"MyModel": {"num_classes": 5}},
                "optimizer": {"name": "torch.optim.Adam"},
                "torch.optim.Adam": {"lr": 1e-3},
                "criterion": {"name": "torch.nn.CrossEntropyLoss"},
                "scheduler": {"name": None},
            }
            batch = model_class.prepare_inputs(data_dict)
            return model_class(config, data_sample=batch)
"""

import math

import pytest
import torch


class HyraxModelContractTests:
    """Mixin class that verifies a @hyrax_model class satisfies the Hyrax interface.

    Subclass this in your test file, configure the class attributes, and provide
    ``model_class``, ``data_dict``, and ``model`` fixtures.

    Class attributes
    ----------------
    has_validate_batch : bool
        Set to True if your model implements ``validate_batch``.
    has_test_batch : bool
        Set to True if your model implements ``test_batch``.

    Required fixtures
    -----------------
    model_class
        The model class itself (not an instance).
    data_dict
        A sample data dictionary in the format the DataProvider produces:
        ``{"data": {...}, "object_id": [...]}``.
    model
        An instantiated model, ready to call methods on.
    """

    has_validate_batch: bool = False
    has_test_batch: bool = False

    # ------------------------------------------------------------------
    # Required fixtures — concrete subclass must override these
    # ------------------------------------------------------------------

    @pytest.fixture
    def model_class(self):  # pragma: no cover
        raise NotImplementedError("Provide a 'model_class' fixture returning the model class.")

    @pytest.fixture
    def data_dict(self):  # pragma: no cover
        raise NotImplementedError(
            "Provide a 'data_dict' fixture returning a sample "
            '{"data": {...}, "object_id": [...]} dict.'
        )

    @pytest.fixture
    def model(self):  # pragma: no cover
        raise NotImplementedError(
            "Provide a 'model' fixture returning an instantiated model."
        )

    # ------------------------------------------------------------------
    # Contract tests
    # ------------------------------------------------------------------

    def test_prepare_inputs_is_staticmethod(self, model_class):
        """prepare_inputs must be a @staticmethod defined on the class."""
        assert "prepare_inputs" in vars(model_class), (
            f"{model_class.__name__} must define 'prepare_inputs' in the class body. "
            "Hyrax raises a RuntimeError at decoration time if it is missing or not static."
        )
        assert isinstance(vars(model_class)["prepare_inputs"], staticmethod), (
            f"{model_class.__name__}.prepare_inputs must be decorated with @staticmethod."
        )

    def test_prepare_inputs_returns_tuple(self, model_class, data_dict):
        """prepare_inputs must return a tuple that forward() and *_batch() can unpack."""
        result = model_class.prepare_inputs(data_dict)
        assert isinstance(result, tuple), (
            f"prepare_inputs must return a tuple, got {type(result).__name__!r}"
        )

    def test_forward_returns_tensor(self, model, model_class, data_dict):
        """forward must return a torch.Tensor."""
        batch = model_class.prepare_inputs(data_dict)
        model.eval()
        with torch.no_grad():
            result = model(batch)
        assert isinstance(result, torch.Tensor), (
            f"forward must return a torch.Tensor, got {type(result).__name__!r}"
        )

    def test_train_batch_returns_dict(self, model, model_class, data_dict):
        """train_batch must return a dict."""
        batch = model_class.prepare_inputs(data_dict)
        model.train()
        result = model.train_batch(batch)
        assert isinstance(result, dict), (
            f"train_batch must return a dict, got {type(result).__name__!r}"
        )

    def test_train_batch_values_are_finite_scalars(self, model, model_class, data_dict):
        """All values in the train_batch result dict must be finite numeric scalars."""
        batch = model_class.prepare_inputs(data_dict)
        model.train()
        result = model.train_batch(batch)
        for key, value in result.items():
            scalar = float(value)
            assert math.isfinite(scalar), (
                f"train_batch result[{key!r}] must be a finite scalar, got {value!r}"
            )

    def test_infer_batch_returns_something(self, model, model_class, data_dict):
        """infer_batch must return a non-None value."""
        batch = model_class.prepare_inputs(data_dict)
        model.eval()
        with torch.no_grad():
            result = model.infer_batch(batch)
        assert result is not None, "infer_batch must not return None"

    def test_validate_batch_returns_dict(self, model, model_class, data_dict):
        """validate_batch must return a dict (skipped if has_validate_batch is False)."""
        if not self.has_validate_batch:
            pytest.skip("has_validate_batch is False")
        batch = model_class.prepare_inputs(data_dict)
        model.eval()
        with torch.no_grad():
            result = model.validate_batch(batch)
        assert isinstance(result, dict), (
            f"validate_batch must return a dict, got {type(result).__name__!r}"
        )

    def test_validate_batch_values_are_finite_scalars(self, model, model_class, data_dict):
        """All values in the validate_batch result dict must be finite scalars."""
        if not self.has_validate_batch:
            pytest.skip("has_validate_batch is False")
        batch = model_class.prepare_inputs(data_dict)
        model.eval()
        with torch.no_grad():
            result = model.validate_batch(batch)
        for key, value in result.items():
            scalar = float(value)
            assert math.isfinite(scalar), (
                f"validate_batch result[{key!r}] must be a finite scalar, got {value!r}"
            )

    def test_test_batch_returns_dict(self, model, model_class, data_dict):
        """test_batch must return a dict (skipped if has_test_batch is False)."""
        if not self.has_test_batch:
            pytest.skip("has_test_batch is False")
        batch = model_class.prepare_inputs(data_dict)
        model.eval()
        with torch.no_grad():
            result = model.test_batch(batch)
        assert isinstance(result, dict), (
            f"test_batch must return a dict, got {type(result).__name__!r}"
        )

    def test_test_batch_values_are_finite_scalars(self, model, model_class, data_dict):
        """All values in the test_batch result dict must be finite scalars."""
        if not self.has_test_batch:
            pytest.skip("has_test_batch is False")
        batch = model_class.prepare_inputs(data_dict)
        model.eval()
        with torch.no_grad():
            result = model.test_batch(batch)
        for key, value in result.items():
            scalar = float(value)
            assert math.isfinite(scalar), (
                f"test_batch result[{key!r}] must be a finite scalar, got {value!r}"
            )
