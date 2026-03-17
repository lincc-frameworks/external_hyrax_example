import torch
import torch.nn as nn
from typing import Union, cast
from hyrax.models.model_registry import hyrax_model


cfgs = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
}

@hyrax_model
class VGG11(nn.Module):
    """Simple example of an externally defined model for testing and demonstration
    purposes."""

    def __init__(self, config, data_sample=None):
        """Basic initialization with architecture definition"""
        super().__init__()
        image_sample = data_sample[0]
        self.in_channels, width, height = image_sample.shape
        self.config = config

        dropout = self.config["external_hyrax_example"]["VGG11"]["dropout"]
        num_classes = self.config["external_hyrax_example"]["VGG11"]["num_classes"]
        batch_norm = self.config["external_hyrax_example"]["VGG11"]["batch_norm"]

        self.features = self.make_layers(cfgs["A"], batch_norm=batch_norm)
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


    def make_layers(self, cfg: list[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
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
        x, _ = batch
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def infer_batch(self, batch):
        """The innermost logic in the inference loop"""
        return self(batch)

    def train_batch(self, batch):
        """The innermost logic in the training loop"""
        _, labels = batch
        self.optimizer.zero_grad()
        outputs = self(batch)
        loss = self.criterion(outputs, labels)
        loss.backward()
        self.optimizer.step()
        return {"loss": loss.item()}

    def validate_batch(self, batch):
        """The innermost logic in the validation loop"""
        _, labels = batch
        outputs = self(batch)
        loss = self.criterion(outputs, labels)
        return {"loss": loss.item()}

    def test_batch(self, batch):
        """The innermost logic in the testing loop"""
        _, labels = batch
        outputs = self(batch)
        loss = self.criterion(outputs, labels)
        return {"loss": loss.item()}

    @staticmethod
    def prepare_data(data_dict):
        """Method that converts the data in dictionary into the form the model expects"""
        image = data_dict["data"]["image"]
        label = data_dict["data"]["label"]
        return (image, label)
