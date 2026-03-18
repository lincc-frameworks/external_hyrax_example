import logging
from pathlib import Path

import h5py
import numpy as np
from hyrax.data_sets import HyraxDataset

logger = logging.getLogger(__name__)


class Galaxy10Dataset(HyraxDataset):
    """Minimal dataset class for the Galaxy10 dataset. This dataset is a collection
    of ~22k images of galaxies categorized into 10 classes. The dataset is stored
    in an HDF5 file available here:
    http://www.astro.utoronto.ca/~bovy/Galaxy10/Galaxy10.h5

    The dataset is made available by AstroNN: https://github.com/henrysky/astroNN
    Deep learning of multi-element abundances from high-resolution spectroscopic data [arXiv:1808.04428]
    """

    def __init__(self, config, data_location):
        """Basic initialization with architecture definition"""
        self.data_location = Path(data_location)

        # Load the file and get the images and labels
        with h5py.File(self.data_location / "Galaxy10.h5", "r") as f:
            self.images = np.array(f["images"])
            self.labels = np.array(f["ans"])

        # The images are stored in (N, H, W, C) format, but if channels_first is
        # True, we transpose to (N, C, H, W)
        self.channels_first = config["external_hyrax_example"]["galaxy10_dataset"]["channels_first"]
        if self.channels_first:
            self.images = self.images.transpose(0, 3, 1, 2)

        # used to pad object IDs with leading zeros for consistent sorting and display
        n_id = len(self.images)
        self.id_width = len(str(n_id))

        super().__init__(config)

    def __len__(self):
        """Returns the number of samples in the dataset"""
        return len(self.images)

    def get_object_id(self, index):
        """Returns the object ID for the sample at the specified index.
        Since this dataset does not have object IDs, we return the index as a string."""
        return f"{index:0{self.id_width}d}"

    def get_image(self, index):
        """Returns the image at the specified index"""
        # return (self.images[index].transpose(2, 0, 1) / 255.0).astype(np.float32)
        return (self.images[index] / 255.0).astype(np.float32)

    def get_label(self, index):
        """Returns the label at the specified index"""
        return self.labels[index]
