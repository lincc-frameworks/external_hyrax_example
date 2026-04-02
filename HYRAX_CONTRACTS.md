# Hyrax Interface Contracts

This is the typed reference for the interfaces every Hyrax component must satisfy.
It is intended as a precise spec for agents and advanced users. For action-oriented
guidance ("what to do"), see `AGENTS.md`.

---

## HyraxDataset

### Import

```python
from hyrax.datasets import HyraxDataset
```

### `__init__` signature

```python
def __init__(self, config: dict, data_location: os.PathLike | str):
```

- `config`: The full Hyrax config dict. Access your values as
  `config["<package_name>"]["<ClassName>"]["<key>"]`.
- `data_location`: Path to the directory (or file) containing the dataset.
  Typically convert with `Path(data_location)` immediately.
- **`super().__init__(config)` must be the last statement** in `__init__`.
  The base class introspects the subclass after all attributes are set.

### Required methods

#### `__len__`

```python
def __len__(self) -> int:
```

Returns the total number of samples in the dataset.

#### `get_object_id`

```python
def get_object_id(self, index: int) -> str:
```

Returns a unique string identifier for sample `index`. This is used for
tracking, logging, and output labeling. If your data has no natural IDs,
return the integer index zero-padded to a consistent width:
`return f"{index:0{self.id_width}d}"`.

#### `get_image`

```python
def get_image(self, index: int) -> np.ndarray:
```

Returns the image for sample `index`.

- dtype: `np.float32`
- value range: `[0.0, 1.0]` (normalize by dividing by 255 for uint8 sources)
- shape: `(H, W, C)` by default; `(C, H, W)` if `channels_first=True`

#### `get_label` (optional)

```python
def get_label(self, index: int):
```

Returns the label for sample `index`. Omit this method for unsupervised tasks.
Return type is task-dependent (commonly an integer class index or a float).

### Do NOT define

- `__getitem__`: provided by `HyraxDataset`; defining your own overrides the
  DataProvider pipeline and will cause runtime failures.

---

## Hyrax Model

### Import

```python
from hyrax.models.model_registry import hyrax_model
```

### Required decorator

```python
@hyrax_model
class MyModel(nn.Module):
```

Every model class must be decorated with `@hyrax_model`. This registers the class
with Hyrax's model discovery system. The class will not be loadable via config
without this decorator, even if the libpath is correct.

### `__init__` signature

```python
def __init__(self, config: dict, data_sample=None):
```

- `config`: The full Hyrax config dict.
- `data_sample`: A single batch from `prepare_inputs`, used to infer input
  dimensions (e.g. number of channels). Raise `ValueError` if `None`.

Store `self.config = config` for use in other methods.

### `prepare_inputs`

```python
@staticmethod
def prepare_inputs(data_dict: dict) -> tuple:
```

Converts the raw `data_dict` from the DataProvider into the tuple consumed by
`forward`, `train_batch`, `validate_batch`, `test_batch`, and `infer_batch`.

**This is the only place to reshape, reformat, or repackage data** between the
dataset and the model. Do not duplicate this logic in `forward`.

#### Shape of `data_dict`

```python
{
    "data": {
        "image": torch.Tensor,   # shape: (B, C, H, W) or (B, H, W, C)
        "label": torch.Tensor,   # shape: (B,) — present only if dataset has get_label
    },
    "object_id": list[str],      # length B
}
```

#### Typical implementation

```python
@staticmethod
def prepare_inputs(data_dict):
    data = data_dict["data"]
    image = data["image"]
    label = data.get("label", None)
    return (image, label)
```

### `forward`

```python
def forward(self, batch: tuple) -> torch.Tensor:
```

Core forward pass. Receives the tuple from `prepare_inputs`. Returns the model
output (e.g. logits). Unpack `batch` to get image and label:

```python
def forward(self, batch):
    x, _ = batch
    ...
    return output
```

### `train_batch`

```python
def train_batch(self, batch: tuple) -> dict:
```

Single training step. Hyrax calls this inside the training loop.

```python
def train_batch(self, batch):
    _, labels = batch
    self.optimizer.zero_grad()
    outputs = self(batch)
    loss = self.criterion(outputs, labels)
    loss.backward()
    self.optimizer.step()
    return {"loss": loss.item()}
```

- `self.optimizer` and `self.criterion` are injected by Hyrax before training.
- Return a dict with at least `"loss"`.
- Do not manage epochs, data loading, or device placement here.

### `validate_batch`

```python
def validate_batch(self, batch: tuple) -> dict:
```

Single validation step. No backward pass. Same return shape as `train_batch`.

### `test_batch`

```python
def test_batch(self, batch: tuple) -> dict:
```

Single test step. Same signature as `validate_batch`.

### `infer_batch`

```python
def infer_batch(self, batch: tuple) -> torch.Tensor:
```

Single inference step. Returns predictions (not a loss dict).

---

## Config namespacing

All config keys are accessed via three-level namespacing:

```python
config["<package_name>"]["<ClassName>"]["<key>"]
```

Example for a package `my_survey_cnn` with a model class `ResNet18`:

```python
dropout   = config["my_survey_cnn"]["ResNet18"]["dropout"]
num_classes = config["my_survey_cnn"]["ResNet18"]["num_classes"]
```

The corresponding TOML section:

```toml
[my_survey_cnn.ResNet18]
dropout = 0.3
num_classes = 5
```

Never access config with flat keys (`config["dropout"]`). The flat access pattern
will raise a `KeyError` at runtime because Hyrax always nests config under the
package and class names.

---

## Summary table

| Contract | Rule |
|----------|------|
| `super().__init__(config)` | Last line of dataset `__init__` |
| `@hyrax_model` | Required on every model class |
| `__getitem__` | Do not define; provided by base class |
| `get_image` return type | `np.float32`, values in `[0, 1]` |
| `get_object_id` return type | `str` |
| `prepare_inputs` | `staticmethod`; sole place to reshape data |
| `train_batch` return | `dict` with at least `"loss"` key |
| Config access | `config["pkg"]["Class"]["key"]` — never flat |
