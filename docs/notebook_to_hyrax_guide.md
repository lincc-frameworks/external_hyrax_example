# Converting a Notebook to Hyrax: Before & After

This guide shows four common notebook patterns and how each maps to Hyrax's
structure. Each section shows the "notebook-style" code on the left and the
correct Hyrax equivalent on the right.

---

## Pattern 1: Flat data loading → `__init__` + `get_image`

### Notebook style

```python
# Cell: load data
import h5py
import numpy as np

with h5py.File("Galaxy10.h5", "r") as f:
    images = np.array(f["images"]) / 255.0   # shape: (N, H, W, C), float32
    labels = np.array(f["ans"])

# Cell: access a sample
idx = 42
img = images[idx]
label = labels[idx]
```

### Hyrax style

```python
# src/my_package/datasets/galaxy10_dataset.py

from pathlib import Path
import h5py
import numpy as np
from hyrax.datasets import HyraxDataset

class Galaxy10Dataset(HyraxDataset):
    def __init__(self, config, data_location):
        self.data_location = Path(data_location)

        with h5py.File(self.data_location / "Galaxy10.h5", "r") as f:
            self.images = (np.array(f["images"]) / 255.0).astype(np.float32)
            self.labels = np.array(f["ans"])

        # HYRAX REQUIREMENT: super().__init__ must be called LAST,
        # after all instance attributes are set.
        super().__init__(config)

    def __len__(self):
        return len(self.images)

    def get_object_id(self, index):
        return str(index)

    def get_image(self, index):
        return self.images[index]   # already float32, already normalized

    def get_label(self, index):
        return self.labels[index]
```

**Key changes:**
- Data loading moves from free cells into `__init__`.
- `images[idx]` becomes `get_image(index)`.
- `labels[idx]` becomes `get_label(index)`.
- `super().__init__(config)` is the last line of `__init__`.
- No `__getitem__` — Hyrax provides it.

---

## Pattern 2: Raw training loop → `train_batch` + `validate_batch`

### Notebook style

```python
# Cell: training loop
model = VGG11()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for epoch in range(10):
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # validation
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            val_loss = criterion(outputs, labels)
```

### Hyrax style

```python
# src/my_package/models/vgg11.py
# (Hyrax handles epochs, loaders, device placement, and the outer loop)

from hyrax.models.model_registry import hyrax_model
import torch.nn as nn

@hyrax_model
class VGG11(nn.Module):

    def train_batch(self, batch):
        """Called once per batch by Hyrax's training loop."""
        _, labels = batch
        self.optimizer.zero_grad()          # self.optimizer injected by Hyrax
        outputs = self(batch)
        loss = self.criterion(outputs, labels)  # self.criterion injected by Hyrax
        loss.backward()
        self.optimizer.step()
        return {"loss": loss.item()}        # must return a dict

    def validate_batch(self, batch):
        """Called once per batch by Hyrax's validation loop. No backward pass."""
        _, labels = batch
        outputs = self(batch)
        loss = self.criterion(outputs, labels)
        return {"loss": loss.item()}
```

**Key changes:**
- Delete the epoch loop, the data loader loop, and the optimizer/criterion construction.
- Hyrax injects `self.optimizer` and `self.criterion` before calling `train_batch`.
- `train_batch` handles exactly one batch. Return `{"loss": value}`.
- `validate_batch` is the same but without `loss.backward()` or `optimizer.step()`.

---

## Pattern 3: Hardcoded hyperparameters → config TOML keys

### Notebook style

```python
# Cell: model config (hardcoded)
DROPOUT = 0.5
NUM_CLASSES = 10
BATCH_NORM = True

model = VGG11(dropout=DROPOUT, num_classes=NUM_CLASSES, batch_norm=BATCH_NORM)
```

### Hyrax style

`src/my_package/default_config.toml`:

```toml
[my_package]

[my_package.VGG11]
dropout = 0.5
num_classes = 10
batch_norm = true

# libpath to specify in runtime config when using this model
# name = "my_package.models.vgg11.VGG11"
```

`src/my_package/models/vgg11.py`:

```python
def __init__(self, config, data_sample=None):
    super().__init__()
    ...
    dropout     = config["my_package"]["VGG11"]["dropout"]
    num_classes = config["my_package"]["VGG11"]["num_classes"]
    batch_norm  = config["my_package"]["VGG11"]["batch_norm"]
```

**Key changes:**
- Hardcoded constants move to the TOML config under
  `[<package_name>.<ClassName>]`.
- Access them via `config["<package_name>"]["<ClassName>"]["<key>"]`.
- The `# name = "..."` lines are documentation only — they show the libpath a
  user would put in their runtime config. **Do not uncomment them** in the
  default config.

---

## Pattern 4: `torch.utils.data.Dataset` subclass → `HyraxDataset` subclass

### Notebook style

```python
from torch.utils.data import Dataset

class GalaxyDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]
```

### Hyrax style

```python
from hyrax.datasets import HyraxDataset

class GalaxyDataset(HyraxDataset):
    def __init__(self, config, data_location):
        self.data_location = Path(data_location)
        # ... load self.images and self.labels ...

        # HYRAX REQUIREMENT: super().__init__ must be called LAST.
        super().__init__(config)

    def __len__(self):
        return len(self.images)

    # Replace __getitem__ with these three methods:

    def get_object_id(self, index):
        return str(index)

    def get_image(self, index):
        return (self.images[index] / 255.0).astype(np.float32)

    def get_label(self, index):
        return self.labels[index]
```

**Key changes:**
- Change base class from `torch.utils.data.Dataset` to `HyraxDataset`.
- Change `__init__` signature from `(self, images, labels)` to
  `(self, config, data_location)` — loading logic moves inside.
- Replace `__getitem__` with `get_object_id`, `get_image`, and `get_label`.
- Add `super().__init__(config)` as the last line of `__init__`.

---

## Quick reference: what goes where

| Notebook concept | Hyrax location |
|-----------------|----------------|
| Data file opening and array loading | `Dataset.__init__` |
| `dataset[i]` image access | `Dataset.get_image(i)` |
| `dataset[i]` label access | `Dataset.get_label(i)` |
| Hyperparameter constants | `default_config.toml` |
| Optimizer and loss construction | Hyrax (do not define) |
| Training loop | Hyrax (do not define) |
| Single training step | `Model.train_batch(batch)` |
| Single validation step | `Model.validate_batch(batch)` |
| Data reshaping before forward | `Model.prepare_inputs(data_dict)` |
| Forward pass | `Model.forward(batch)` |
