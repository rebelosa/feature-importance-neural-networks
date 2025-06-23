# neural-feature-importance

Variance-based feature importance for deep learning models.

`neural-feature-importance` implements the method described in [CR de Sá, *Variance-based Feature Importance in Neural Networks*](https://doi.org/10.1007/978-3-030-33778-0_24). The library tracks the variance of input weights during training using Welford's algorithm and computes normalized importance scores.

## Features
- `VarianceImportanceKeras` callback for TensorFlow/Keras
- `VarianceImportanceTorch` helper for PyTorch
- Early-stopping `MetricThreshold` callback
- Utility scripts to reproduce the experiments from the paper

## Installation

```bash
pip install "neural-feature-importance[tensorflow]"  # Keras support
pip install "neural-feature-importance[torch]"       # PyTorch support
```

## Getting the version

```python
from neural_feature_importance import __version__
print(__version__)
```

## Usage

### Keras

```python
from neural_feature_importance import VarianceImportanceKeras
from neural_feature_importance.utils import MetricThreshold

viann = VarianceImportanceKeras()
monitor = MetricThreshold(monitor="val_accuracy", threshold=0.95)
model.fit(X, Y, validation_split=0.05, epochs=30, callbacks=[viann, monitor])
print(viann.var_scores)
```

### PyTorch

```python
from neural_feature_importance import VarianceImportanceTorch

tracker = VarianceImportanceTorch(model)
tracker.on_train_begin()
for epoch in range(num_epochs):
    train_one_epoch(model, optimizer, data_loader)
    tracker.on_epoch_end()
tracker.on_train_end()
print(tracker.var_scores)
```

## Example scripts

Run `compare_feature_importance.py` to train a small network on the Iris dataset and compare the scores with a random forest baseline:

```bash
python compare_feature_importance.py
```

Run `full_experiment.py` to reproduce the paper's experiment across several datasets:

```bash
python full_experiment.py
```

## Development

After making changes, run:

```bash
python -m py_compile neural_feature_importance/callbacks.py
python -m py_compile "variance-based feature importance in artificial neural networks.ipynb" 2>&1 | head
jupyter nbconvert --to script "variance-based feature importance in artificial neural networks.ipynb" --stdout | head
```
