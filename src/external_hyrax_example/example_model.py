import torch
import torch.nn as nn
from hyrax.models.model_registry import hyrax_model


@hyrax_model
class ExampleModel(nn.Module):
    """Simple example of an externally defined model for testing and demonstration
    purposes."""

    def __init__(self, config, data_sample=None):
        """Basic initialization with architecture definition"""
        super().__init__()
        channels, width, height = data_sample["data"]["image"].shape
        self.config = config
        layer = self.config["model"]["ExampleModel"]["layer"]
        self.linear = nn.Linear(channels * width * height, layer)

    def forward(self, x):
        """Standard PyTorch forward method"""
        return self.linear(x)

    def train_step(self, batch):
        """The innermost logic in the training loop"""
        x, y = batch
        y_pred = self(x)
        loss = nn.functional.mse_loss(y_pred, y)
        return loss

    @staticmethod
    def to_tensor(data_dict):
        """Method that converts the data in dictionary into the form the model expects"""
        image = data_dict["data"]["image"][0]
        label = data_dict["data"]["label"]
        return (torch.tensor(image, dtype=torch.float32), torch.tensor(label, dtype=torch.float32))
