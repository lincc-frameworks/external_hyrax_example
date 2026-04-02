"""Reusable contract tests for HyraxDataset subclasses.

Copy this file into your project's test directory and subclass
HyraxDatasetContractTests to verify your dataset implementation.

Example
-------
::

    class TestMyDataset(HyraxDatasetContractTests):
        getter_names = ["get_spectrum", "get_label"]
        n_samples = 5

        @pytest.fixture
        def dataset(self, tmp_path):
            config = {"my_package": {"MyDataset": {...}}}
            return MyDataset(config, data_location=tmp_path)
"""

import numpy as np
import pytest
from hyrax.datasets import HyraxDataset


class HyraxDatasetContractTests:
    """Mixin class that verifies a HyraxDataset subclass satisfies the Hyrax interface.

    Subclass this in your test file, set ``getter_names`` to the list of getter
    methods your dataset implements, and provide a ``dataset`` fixture that returns
    an instantiated dataset object.

    Class attributes
    ----------------
    getter_names : list[str]
        Names of getter methods to verify (e.g. ``["get_image", "get_label"]``).
        Each method must accept an integer index and return a ``numpy.ndarray``
        with a numeric dtype.  Leave empty to skip getter checks.
    n_samples : int
        Number of samples to spot-check.  Defaults to 3.
    """

    getter_names: list = []
    n_samples: int = 3

    # ------------------------------------------------------------------
    # Required fixture — concrete subclass must override this
    # ------------------------------------------------------------------

    @pytest.fixture
    def dataset(self):  # pragma: no cover
        raise NotImplementedError(
            "Provide a 'dataset' fixture in your concrete test class that returns "
            "an instantiated HyraxDataset subclass."
        )

    # ------------------------------------------------------------------
    # Contract tests
    # ------------------------------------------------------------------

    def test_inherits_hyrax_dataset(self, dataset):
        """Dataset must be an instance of HyraxDataset.

        Failing this usually means super().__init__(config) was not called,
        or was called before all instance attributes were set.
        """
        assert isinstance(dataset, HyraxDataset), (
            f"{type(dataset).__name__} is not an instance of HyraxDataset. "
            "Make sure super().__init__(config) is called at the END of __init__."
        )

    def test_len_is_positive_int(self, dataset):
        """__len__ must return a positive integer."""
        n = len(dataset)
        assert isinstance(n, int), f"__len__ must return int, got {type(n).__name__}"
        assert n > 0, f"__len__ must return a positive value, got {n}"

    def test_get_object_id_returns_str(self, dataset):
        """get_object_id must return a str for each index."""
        for i in range(min(self.n_samples, len(dataset))):
            result = dataset.get_object_id(i)
            assert isinstance(result, str), (
                f"get_object_id({i}) must return str, got {type(result).__name__!r}"
            )

    def test_get_object_id_unique(self, dataset):
        """get_object_id must return unique IDs across samples."""
        ids = [dataset.get_object_id(i) for i in range(min(self.n_samples, len(dataset)))]
        assert len(ids) == len(set(ids)), f"get_object_id returned duplicate IDs: {ids}"

    def test_getters_return_ndarray(self, dataset):
        """Each getter in getter_names must return a numpy array or numpy scalar.

        ``numpy.ndarray`` (for multi-dimensional data like images or spectra) and
        ``numpy.generic`` (numpy scalars like ``np.int64`` or ``np.float32``) are
        both accepted.  Plain Python types (``int``, ``list``, ``torch.Tensor``,
        etc.) are not.
        """
        if not self.getter_names:
            pytest.skip("getter_names is empty — add getter method names to check them")
        for name in self.getter_names:
            assert hasattr(dataset, name), (
                f"Dataset {type(dataset).__name__!r} is missing getter method {name!r}"
            )
            result = getattr(dataset, name)(0)
            assert isinstance(result, (np.ndarray, np.generic)), (
                f"{name}(0) must return a numpy array or numpy scalar, "
                f"got {type(result).__name__!r}"
            )

    def test_getters_dtype_is_numeric(self, dataset):
        """Each getter in getter_names must return data with a numeric dtype."""
        if not self.getter_names:
            pytest.skip("getter_names is empty — add getter method names to check them")
        for name in self.getter_names:
            result = getattr(dataset, name)(0)
            dtype = np.dtype(type(result)) if isinstance(result, np.generic) else result.dtype
            assert np.issubdtype(dtype, np.number), (
                f"{name}(0) must return numeric data, got dtype {dtype!r}"
            )
