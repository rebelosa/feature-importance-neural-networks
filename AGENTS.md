# Repository Guidelines

This repository contains a minimal Python package named `neural-feature-importance` which provides utilities to compute variance-based feature importances. It exposes `VarianceImportanceCallback` for `tf.keras` models, `VarianceImportanceTorch` for PyTorch, and an `AccuracyMonitor` helper. The method is described in the article [CR de Sá, *Variance-based Feature Importance in Neural Networks*](https://doi.org/10.1007/978-3-030-33778-0_24).

## Style
- Follow PEP8 and PEP257 standards.
- Use `tf.keras` APIs instead of the standalone `keras` package.
- Use Python `logging` for all user-facing messages. Do not add print statements or verbose flags.
- Provide type hints where practical and keep docstrings concise and descriptive.

## Package layout
- The public API lives in `neural_feature_importance/callbacks.py` and is re-exported from `neural_feature_importance/__init__.py`.
- `VarianceImportanceCallback` exposes the attribute `var_scores` and property `feature_importances_`. The module also provides `AccuracyMonitor` and `VarianceImportanceTorch`.

## Testing
Run these commands after making changes:

```bash
python -m py_compile neural_feature_importance/callbacks.py
python -m py_compile "variance-based feature importance in artificial neural networks.ipynb" 2>&1 | head
# Optionally check notebook conversion if `jupyter` is available
jupyter nbconvert --to script "variance-based feature importance in artificial neural networks.ipynb" --stdout | head
```

