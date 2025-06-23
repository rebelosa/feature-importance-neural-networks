# neural-feature-importance

[![PyPI version](https://img.shields.io/pypi/v/neural-feature-importance.svg)](https://pypi.org/project/neural-feature-importance/)
[![Python versions](https://img.shields.io/pypi/pyversions/neural-feature-importance.svg)](https://pypi.org/project/neural-feature-importance/)

Variance-based feature importance for deep learning models.

`neural-feature-importance` implements the method described in
[CR de Sá, *Variance-based Feature Importance in Neural Networks*](https://doi.org/10.1007/978-3-030-33778-0_24).
It tracks the variance of the first trainable layer using Welford's algorithm
and produces normalized importance scores for each feature.

## Features

- `VarianceImportanceKeras` — drop-in callback for TensorFlow/Keras models
- `VarianceImportanceTorch` — helper class for PyTorch training loops
- `MetricThreshold` — early-stopping callback based on a monitored metric
- Example scripts to reproduce the experiments from the paper

## Installation

```bash
pip install "neural-feature-importance[tensorflow]"  # for Keras
pip install "neural-feature-importance[torch]"       # for PyTorch
```

Retrieve the package version via:

```python
from neural_feature_importance import __version__
print(__version__)
```

## Quick start

### Keras

```python
from neural_feature_importance import VarianceImportanceKeras
from neural_feature_importance.utils import MetricThreshold

viann = VarianceImportanceKeras()
monitor = MetricThreshold(monitor="val_accuracy", threshold=0.95)
model.fit(X, y, validation_split=0.05, epochs=30, callbacks=[viann, monitor])
print(viann.feature_importances_)
```

### PyTorch

```python
from neural_feature_importance import VarianceImportanceTorch

tracker = VarianceImportanceTorch(model)
tracker.on_train_begin()
for epoch in range(num_epochs):
    train_one_epoch(model, optimizer, dataloader)
    tracker.on_epoch_end()
tracker.on_train_end()
print(tracker.feature_importances_)
```

## Example scripts

Run `compare_feature_importance.py` to train a small network on the Iris dataset
and compare the scores with a random forest baseline:

```bash
python compare_feature_importance.py
```

Run `full_experiment.py` to reproduce the experiments from the paper:

```bash
python full_experiment.py
```

## Development

After making changes, run the following checks:

```bash
python -m py_compile neural_feature_importance/callbacks.py
python -m py_compile "variance-based feature importance in artificial neural networks.ipynb" 2>&1 | head
jupyter nbconvert --to script "variance-based feature importance in artificial neural networks.ipynb" --stdout | head
```

## Citation

If you use this package in your research, please cite:

```bibtex
@inproceedings{DBLP:conf/dis/Sa19,
  author       = {Cl{\'a}udio Rebelo de S{\'a}},
  editor       = {Petra Kralj Novak and
                  Tomislav Smuc and
                  Saso Dzeroski},
  title        = {Variance-Based Feature Importance in Neural Networks},
  booktitle    = {Discovery Science - 22nd International Conference, {DS} 2019, Split,
                  Croatia, October 28-30, 2019, Proceedings},
  series       = {Lecture Notes in Computer Science},
  volume       = {11828},
  pages        = {306--315},
  publisher    = {Springer},
  year         = {2019},
  url          = {https://doi.org/10.1007/978-3-030-33778-0\_24},
  doi          = {10.1007/978-3-030-33778-0\_24},
  timestamp    = {Thu, 07 Nov 2019 09:20:36 +0100},
  biburl       = {https://dblp.org/rec/conf/dis/Sa19.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```

We appreciate citations as they help the community discover this work.
