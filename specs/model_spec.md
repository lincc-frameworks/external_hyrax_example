# Hyrax Model Spec

This document is the authoritative reference for converting a PyTorch model to be
Hyrax-compatible. It is written for coding agents. Read it in full before modifying
or generating any model code.

---

## 1. Repository Layout

Place model files and configuration in the following locations:

```
src/<package>/
    models/
        <model_name>.py      # @hyrax_model decorated nn.Module subclass
    default_config.toml      # default hyperparameters for all models and datasets
    __init__.py
```

Reference implementation: `src/external_hyrax_example/models/vgg11.py`

---

## 2. `@hyrax_model` Decorator

```python
from hyrax.models.model_registry import hyrax_model

@hyrax_model
class MyModel(nn.Module):
    ...
```

- Required on every model class.
- Registers the class in Hyrax's model registry so it can be loaded by its
  fully-qualified dotted path, e.g. `"mypkg.models.mymodel.MyModel"`.
- Without it, `h.set_config("model.name", "...")` will fail at runtime.

---

## 3. Constructor: `__init__(self, config, data_sample=None)`

Required signature:

```python
def __init__(self, config, data_sample=None):
    super().__init__()
```

### `config`

- Nested `dict` populated from `default_config.toml` plus any runtime overrides.
- Access model hyperparameters via `config["<package>"]["<ClassName>"]["<param>"]`.
- Example: `config["mypkg"]["MyModel"]["num_classes"]`

### `data_sample`

- A batch tuple `(images, ...)` where `images` has shape `(N, C, H, W)`.
- Use `data_sample[0].shape` to infer input dimensions dynamically instead of
  hard-coding them.
- Raise `ValueError` if the model requires `data_sample` but receives `None`.

### `default_config.toml` entry

Add a TOML section for every configurable hyperparameter:

```toml
[mypkg.MyModel]
num_classes = 10
dropout = 0.5
```

Runtime override syntax:

```python
h.set_config("mypkg.MyModel.num_classes", 5)
```

---

## 4. Required Batch Methods

Each method implements only the **innermost logic** of its respective loop.
Hyrax owns the outer loop, gradient context (`torch.no_grad()` for val/test/infer),
and device placement.

| Method | Hyrax caller | Returns |
|---|---|---|
| `forward(batch: tuple) -> Tensor` | all batch methods | raw model output |
| `train_batch(batch) -> dict` | `h.train()` | `{"loss": float, ...}` |
| `validate_batch(batch) -> dict` | `h.train()` (val phase) | `{"loss": float, ...}` |
| `test_batch(batch) -> dict` | `h.test()` | `{"loss": float, ...}` |
| `infer_batch(batch) -> Tensor` | `h.infer()` | raw model output (same as `forward`) |

### `forward`

Standard PyTorch forward pass. Called internally by all batch methods.

```python
def forward(self, batch: tuple) -> torch.Tensor:
    x, _ = batch
    return self.net(x)
```

### `train_batch`

Must zero gradients, compute loss, back-propagate, and step the optimizer.
`self.optimizer` and `self.criterion` are injected by Hyrax before training starts.
Hyrax provides configurable built-in PyTorch optimizers and loss functions by default;
the model may supply its own if needed.

```python
def train_batch(self, batch):
    _, labels = batch
    self.optimizer.zero_grad()
    loss = self.criterion(self(batch), labels)
    loss.backward()
    self.optimizer.step()
    return {"loss": loss.item()}
```

### `validate_batch` and `test_batch`

Same as `train_batch` but without the gradient update. Do **not** call
`loss.backward()` or `self.optimizer.step()`.

```python
def validate_batch(self, batch):
    _, labels = batch
    return {"loss": self.criterion(self(batch), labels).item()}

def test_batch(self, batch):
    _, labels = batch
    return {"loss": self.criterion(self(batch), labels).item()}
```

### `infer_batch`

No loss computation. Return raw model output.

```python
def infer_batch(self, batch):
    return self(batch)
```

---

## 5. `prepare_inputs` Static Method

Hyrax calls this before every batch to convert the raw data dictionary into
the tuple format the batch methods expect.

```python
@staticmethod
def prepare_inputs(data_dict):
    if "data" not in data_dict:
        raise RuntimeError("Unable to find `data` key in data_dict")
    data = data_dict["data"]
    label = data.get("label", None)
    return (data["image"], label)
```

- Must be a `@staticmethod`.
- Input structure: `{"data": {"image": Tensor, "label": Tensor, ...}}`.
- `"label"` is absent during inference; always use `.get("label", None)`.
- Best practice: raise `RuntimeError` when the `"data"` key is missing.
- Return value must match the tuple signature consumed by `forward`.

---

## 6. Wiring a Model into Hyrax

```python
from hyrax import Hyrax

h = Hyrax()
h.set_config("model.name", "mypkg.models.mymodel.MyModel")
h.set_config("data_request", {
    "train": {
        "data": {
            "dataset_class": "mypkg.datasets.mydataset.MyDataset",
            "data_location": "./data/my_data",
            "fields": ["image", "label"],
            "primary_id_field": "object_id",
            "split_fraction": 0.8,
        },
    },
})

model = h.train()
results = h.infer()
```

---

## 7. Common Pitfalls

- **Hard-coding input shape**: always infer channels from `data_sample[0].shape`
  so the model works across datasets with different channel counts.
- **Missing `@hyrax_model`**: the model will not be discoverable by dotted path
  and `h.train()` will raise an error.
- **Wrong config namespace**: the top-level key must match the Python package name
  (e.g. `config["mypkg"]`, not `config["MyModel"]`).
- **Backward pass in `validate_batch` / `test_batch`**: these run inside
  `torch.no_grad()`; calling `loss.backward()` will raise a runtime error.
- **Defining `self.optimizer` / `self.criterion` in `__init__`**: Hyrax
  overwrites these attributes before training begins. Initialising them in the
  constructor bypasses Hyrax's built-in optimizer and loss configurability.

---

## 8. Reference Files

| File | Purpose |
|---|---|
| `src/external_hyrax_example/models/vgg11.py` | Complete, annotated working example (read this first) |
| `src/external_hyrax_example/default_config.toml` | TOML config pattern |
| `docs/pre_executed/model_usage_example.ipynb` | End-to-end usage walkthrough |
