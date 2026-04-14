from typing import Union, cast

import torch
import torch.nn as nn
from hyrax.models.model_registry import hyrax_model

cfgs = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
}


# HYRAX REQUIREMENT: all model classes must use this decorator.
# It registers the class with Hyrax's model discovery system.
# Omitting it means Hyrax cannot find the model at runtime.
@hyrax_model
class VGG11(nn.Module):
    """VGG11 CNN — canonical example of a Hyrax-compatible external model.

    This class demonstrates every requirement for integrating a PyTorch model
    with the Hyrax framework:

    1. Decorated with ``@hyrax_model`` so Hyrax can discover this class by its
       fully-qualified dotted path (e.g. ``"external_hyrax_example.models.vgg11.VGG11"``).
    2. Constructor accepts ``config`` and ``data_sample`` (see ``__init__``).
    3. Implements the five Hyrax batch methods: ``forward``, ``train_batch``,
       ``validate_batch``, ``test_batch``, and ``infer_batch``.
    4. Implements the static ``prepare_inputs`` method to convert the raw
       data dictionary supplied by Hyrax into the tuple format the model expects.
    5. Reads all hyperparameters from ``config`` using the hierarchical key
       ``config["<package>"]["<ClassName>"]["<param>"]``.

    Reference: https://docs.pytorch.org/vision/main/models/generated/torchvision.models.vgg11.html
    """

    def __init__(self, config, data_sample=None):
        """Initialize the VGG11 architecture.

        Args:
            config: Nested configuration dictionary populated from the package's
                ``default_config.toml`` and any runtime overrides set via
                ``hyrax.Hyrax.set_config``.  Model hyperparameters are read from
                ``config["external_hyrax_example"]["VGG11"]``.
            data_sample: A single batch tuple ``(images, labels)`` prepared for
                the model by Hyrax via ``prepare_inputs``. The first element must
                be a float tensor of shape ``(N, C, H, W)``. The channel count
                ``C`` is used to build the first convolutional layer so that the
                model adapts to the dataset without hard-coding input dimensions.
                Raises ``ValueError`` if ``None``.

        Raises:
            ValueError: If ``data_sample`` is not provided.

        See Also:
            https://docs.pytorch.org/vision/main/models/generated/torchvision.models.vgg11.html#torchvision.models.vgg11
        """
        super().__init__()
        if data_sample is None:
            raise ValueError(
                "VGG11 expected 'data_sample' to be provided at construction time "
                "so that input channel dimensions can be inferred, but received None."
            )

        # Infer input channels from the sample rather than hard-coding them.
        # data_sample[0] is a batch of images with shape (N, C, H, W).
        image_sample = data_sample[0]
        batch_size, self.in_channels, width, height = image_sample.shape

        self.config = config
        dropout = self.config["external_hyrax_example"]["VGG11"]["dropout"]
        num_classes = self.config["external_hyrax_example"]["VGG11"]["num_classes"]
        batch_norm = self.config["external_hyrax_example"]["VGG11"]["batch_norm"]

        self.features = self._make_layers(cfgs["A"], batch_norm=batch_norm)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes),
        )

    def _make_layers(self, cfg: list[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
        """Helper function to create the convolutional layers of the VGG11 architecture"""
        layers: list[nn.Module] = []
        in_channels = self.in_channels
        for v in cfg:
            if v == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                v = cast(int, v)
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, batch: tuple) -> torch.Tensor:
        """Run the model's core forward pass.

        Args:
            batch: Tuple ``(images, labels)`` where ``images`` is a float
                tensor of shape ``(N, C, H, W)``.  ``labels`` is ignored here
                but included so that ``batch`` can be passed directly from
                ``train_batch`` / ``validate_batch`` without unpacking.

        Returns:
            Class logits of shape ``(N, num_classes)``.
        """
        x, _ = batch
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def infer_batch(self, batch):
        """Innermost logic of the Hyrax inference loop.

        Called once per batch by ``hyrax.Hyrax.infer()``.  No loss is computed.

        Args:
            batch: Tuple ``(images, labels)`` as produced by ``prepare_inputs``.

        Returns:
            Class logits tensor of shape ``(N, num_classes)``.
        """
        return self(batch)

    def train_batch(self, batch):
        """Innermost logic of the Hyrax training loop.

        Called once per batch by ``hyrax.Hyrax.train()``.  Hyrax handles the
        outer epoch loop, gradient context, and device placement.  This method
        is responsible only for the per-batch forward pass, loss computation,
        and parameter update.

        ``self.optimizer`` and ``self.criterion`` are injected by Hyrax before
        training starts.  Hyrax provides configurable built-in PyTorch optimizers
        and loss functions by default; the model can override them if needed.

        Args:
            batch: Tuple ``(images, labels)`` as produced by ``prepare_inputs``.

        Returns:
            Dict of scalar metrics, e.g. ``{"loss": 0.312}``.
        """
        _, labels = batch
        self.optimizer.zero_grad()
        outputs = self(batch)
        loss = self.criterion(outputs, labels)
        loss.backward()
        self.optimizer.step()
        return {"loss": loss.item()}

    def validate_batch(self, batch):
        """Innermost logic of the Hyrax validation loop.

        Called once per batch during the validation phase of ``hyrax.Hyrax.train()``.
        Hyrax runs this inside a ``torch.no_grad()`` context, so no backward pass
        is needed or expected.

        Args:
            batch: Tuple ``(images, labels)`` as produced by ``prepare_inputs``.

        Returns:
            Dict of scalar metrics, e.g. ``{"loss": 0.298}``.
        """
        _, labels = batch
        outputs = self(batch)
        loss = self.criterion(outputs, labels)
        return {"loss": loss.item()}

    def test_batch(self, batch):
        """Innermost logic of the Hyrax test loop.

        Called once per batch by ``hyrax.Hyrax.test()``.  Like ``validate_batch``,
        this runs inside a ``torch.no_grad()`` context.

        Args:
            batch: Tuple ``(images, labels)`` as produced by ``prepare_inputs``.

        Returns:
            Dict of scalar metrics, e.g. ``{"loss": 0.301}``.
        """
        _, labels = batch
        outputs = self(batch)
        loss = self.criterion(outputs, labels)
        return {"loss": loss.item()}

    # HYRAX REQUIREMENT: this staticmethod receives the raw data_dict from the
    # DataProvider and must return the tuple that forward() and train_batch() expect.
    # It is the ONLY place to reshape/convert data between the dataset and the model.
    @staticmethod
    def prepare_inputs(data_dict):
        """Convert a Hyrax data dictionary into the tuple format the model expects.

        Hyrax calls this method before passing data to ``train_batch``,
        ``validate_batch``, ``test_batch``, and ``infer_batch``.  The returned
        tuple must match the signature expected by ``forward``.

        The incoming ``data_dict`` has the structure::

            {
                "data": {
                    "image": <tensor>,   # always present
                    "label": <tensor>,   # present during train/val/test, absent during infer
                    ...                  # any other fields declared in data_request
                }
            }

        Args:
            data_dict: Dictionary produced by the dataset class and Hyrax's
                data loader.  Must contain a ``"data"`` key.

        Returns:
            Tuple ``(image, label)`` where ``label`` is ``None`` if not present.

        Raises:
            RuntimeError: If ``data_dict`` does not contain a ``"data"`` key.
        """
        if "data" not in data_dict:
            raise RuntimeError("Unable to find `data` key in data_dict")

        data = data_dict["data"]

        image = data["image"]

        label = None
        if "label" in data_dict["data"]:
            label = data_dict["data"]["label"]

        return (image, label)
