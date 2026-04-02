# Agent Instructions: Converting a Notebook to Hyrax

This file tells AI agents (Claude, Codex, etc.) how to convert a Jupyter notebook
into a working Hyrax project. Read this file completely before writing any code.

---

## What you are being asked to do

Convert a user's Jupyter notebook into three Python source files that integrate with
the [Hyrax](https://github.com/lincc-frameworks/hyrax) framework. The notebook
contains data loading and model training logic. Your job is to restructure that logic
into Hyrax's three-file pattern.

If the user's prompt is terse (e.g. "convert my notebook"), treat this file as
authoritative. Do not invent a file layout or class structure — follow the spec below.

---

## Output: three files

Given a user project named `<package_name>`, produce exactly these files:

```
src/<package_name>/
    datasets/<snake_case_dataset_name>.py   # HyraxDataset subclass
    models/<snake_case_model_name>.py       # @hyrax_model class
    default_config.toml                     # TOML config with namespaced sections
```

Derive `<package_name>` from the user's stated project name, converting to
`snake_case`. If the user does not specify one, ask before proceeding.

---

## Dataset class requirements

### Mandatory base class and init ordering

```python
from hyrax.datasets import HyraxDataset

class MyDataset(HyraxDataset):
    def __init__(self, config, data_location):
        # Load ALL instance attributes here first
        self.data_location = Path(data_location)
        self.images = ...   # load data
        self.labels = ...   # load data

        # HYRAX REQUIREMENT: super().__init__ must be called LAST in __init__,
        # after all instance attributes are set. The base class inspects the
        # subclass attributes; calling super() first will cause silent failures.
        super().__init__(config)
```

**Do not call `super().__init__(config)` at the top of `__init__`. Always call it last.**

### Required methods

| Method | Signature | Returns |
|--------|-----------|---------|
| `__len__` | `(self) -> int` | Number of samples |
| `get_object_id` | `(self, index: int) -> str` | Unique string ID for this sample |
| `get_image` | `(self, index: int) -> np.ndarray` | Image array, `float32`, values in `[0, 1]` |
| `get_label` | `(self, index: int)` | Label (optional, omit if unsupervised) |

**Do not define `__getitem__`.** Hyrax's base class provides `__getitem__`; defining
your own will break the DataProvider pipeline.

**Do not put data loading logic in `get_image` or `get_label`** (aside from simple
indexing). Load the full dataset in `__init__` and index into it in the getters.

### Image format

Return images from `get_image` as `np.float32` normalized to `[0, 1]`.
If `channels_first` is configured, return shape `(C, H, W)`; otherwise `(H, W, C)`.
Check `config["<package_name>"]["<dataset_class_name>"]["channels_first"]`.

---

## Model class requirements

### Mandatory decorator

```python
from hyrax.models.model_registry import hyrax_model

# HYRAX REQUIREMENT: all model classes must use this decorator.
# It registers the class with Hyrax's model discovery system.
# Omitting it means Hyrax cannot find the model at runtime.
@hyrax_model
class MyModel(nn.Module):
    ...
```

**Do not omit `@hyrax_model`.** It is required. The class will not be discoverable
by Hyrax without it, even if everything else is correct.

### `__init__` signature

```python
def __init__(self, config, data_sample=None):
```

`data_sample` is a batch from `prepare_inputs` used to infer input dimensions.
Raise a clear `ValueError` if `data_sample is None`.

### Required methods

| Method | Role |
|--------|------|
| `prepare_inputs(data_dict)` | **staticmethod.** Converts the raw `data_dict` from the DataProvider into the tuple that `forward()`, `train_batch()`, etc. receive. |
| `forward(batch)` | Core forward pass. Receives the tuple from `prepare_inputs`. |
| `train_batch(batch)` | Training step. Calls `self(batch)`, computes loss, calls `loss.backward()`, returns `{"loss": ...}`. |
| `validate_batch(batch)` | Validation step. Same signature, no backward pass. |
| `test_batch(batch)` | Test step. Same signature as `validate_batch`. |
| `infer_batch(batch)` | Inference step. Returns predictions. |

### The `prepare_inputs` / `forward` / `train_batch` pattern

This is the most important design point. The data flows like this:

```
DataProvider.data_dict  →  prepare_inputs(data_dict)  →  batch tuple
batch tuple             →  forward(batch)              →  model output
batch tuple             →  train_batch(batch)          →  {"loss": value}
```

`prepare_inputs` is a **staticmethod** that receives a dict structured as:

```python
{
    "data": {
        "image": <tensor>,   # image data
        "label": <tensor>,   # label data (if supervised)
    },
    "object_id": <list of str>,
}
```

It must return a tuple. The rest of the model methods unpack that tuple.

**Do not collapse this into a single `forward` call.** Do not put data reshaping
logic inside `forward`. `prepare_inputs` is the single designated place for
data preparation.

**Do not import PyTorch's training loop utilities** (e.g. `torch.optim` step
management, loss aggregation). Hyrax owns the outer training loop. Your
`train_batch` only needs to: zero gradients, run the forward pass, compute loss,
call `loss.backward()`, step the optimizer, and return the loss dict.
`self.optimizer` and `self.criterion` are provided by Hyrax.

---

## Config TOML requirements

The config must use Hyrax's namespacing convention:

```toml
[<package_name>]

[<package_name>.<ModelClassName>]
dropout = 0.5
num_classes = 10
batch_norm = true

# libpath to specify in runtime config when using this model
# name = "<package_name>.models.<module>.<ClassName>"


[<package_name>.<DatasetClassName>]
channels_first = false

# libpath to specify in runtime config when using this dataset
# name = "<package_name>.datasets.<module>.<ClassName>"
```

Key rules:
- The outer section `[<package_name>]` must exist even if empty.
- Section names use the **exact Python class name** (e.g. `VGG11`, not `vgg11`).
- The commented-out `name = "..."` lines are **not active config**. They are
  documentation showing the libpath a user would paste into their runtime config
  to activate this class. Do not uncomment them in the default config.
- Access config values in code as:
  `config["<package_name>"]["<ClassName>"]["<key>"]`
- Do not use flat keys like `config["key"]` or `self.config["dropout"]`.

---

## What NOT to do

| Wrong | Right |
|-------|-------|
| `super().__init__(config)` at top of `__init__` | Call it **last** |
| Define `__getitem__` | Let `HyraxDataset` provide it |
| Put data loading in `get_image` | Load in `__init__`, index in `get_image` |
| Omit `@hyrax_model` | Always decorate model classes |
| Collapse `prepare_inputs` into `forward` | Keep them separate |
| Manage the training loop manually | Hyrax owns the outer loop |
| `config["dropout"]` flat access | `config["pkg"]["Class"]["dropout"]` |
| Import `torch.utils.data.Dataset` as base | Import `HyraxDataset` |

---

## Reference files in this repository

The canonical example implementation is in:

- `src/external_hyrax_example/datasets/galaxy10_dataset.py` — dataset example
- `src/external_hyrax_example/models/vgg11.py` — model example
- `src/external_hyrax_example/default_config.toml` — config example

For precise interface contracts (types, shapes, signatures), see `HYRAX_CONTRACTS.md`.

For a before/after worked example, see `docs/notebook_to_hyrax_guide.md`.
