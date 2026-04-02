# Notebook-to-Hyrax Conversion Prompt Template

Copy the block below and fill in every `[PLACEHOLDER]` before sending to an AI agent.
The more precisely you fill this in, the better the output will be.

---

```
Convert my Jupyter notebook to a Hyrax project. Follow the instructions in AGENTS.md.

## Project name
Output package name: [e.g. my_survey_cnn]

## Data source
File format: [e.g. HDF5 / FITS / NumPy .npy / JPEG files in a directory / etc.]
Storage layout: [e.g. single file at data_location/catalog.h5 / one file per object at data_location/<id>.fits / etc.]
Loading library: [e.g. h5py / astropy.io.fits / numpy / PIL / etc.]

## Data fields per sample
Each sample contains:
- Image field: [e.g. dataset key "images", shape (H, W, C), dtype uint8]
- Label field: [e.g. dataset key "labels", integer class index 0–9]   (omit if unsupervised)
- Object ID field: [e.g. dataset key "ids", string / or: no IDs, use integer index]
- Other fields: [list any other metadata fields, or "none"]

Total dataset size: [e.g. ~22,000 samples]

## Model architecture
Type: [e.g. CNN classifier / autoencoder / regression / etc.]
Input: [e.g. (3, 224, 224) float32 image tensor]
Output: [e.g. logits over 10 classes]
Key layers: [brief description or paste the architecture from the notebook]

## Hyperparameters to expose in config
List the values you want adjustable in the TOML config (not hardcoded):
- [e.g. dropout: 0.5]
- [e.g. num_classes: 10]
- [e.g. learning_rate: 1e-4]
- [e.g. channels_first: false]

Hardcode everything else.

## Notebook file
[Paste the notebook content here, or attach the .ipynb file]
```

---

## Tips for filling in the template

**Data source:** Be specific about file paths. "HDF5 file" is less useful than
"single file named `Galaxy10.h5` inside the `data_location` directory, opened
with `h5py`". The agent needs to know how to reconstruct the loading code.

**Data fields:** Include the exact key names used to access each field (e.g.
`f["images"]`, `row["flux"]`). Include dtype and shape if you know them — this
affects normalization and channel handling.

**Object ID:** If your data has natural IDs (catalog IDs, file names, sky
coordinates), say so. If not, say "use the integer index, zero-padded to N digits".
`get_object_id` must return a string.

**Model architecture:** You don't need to be exhaustive. The agent will read the
notebook. But calling out the input/output shapes saves a common inference step.

**Hyperparameters:** Only list values you actually want to tune between runs.
Architectural constants (layer counts, kernel sizes) are usually better hardcoded
unless you plan to sweep them.

**Package name:** Use `snake_case`. The agent will create
`src/<package_name>/datasets/` and `src/<package_name>/models/`. Pick something
that won't collide with existing Python packages.
