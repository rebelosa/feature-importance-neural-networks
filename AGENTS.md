# Repository Guidelines

This repository contains a minimal Python package named `variance_importance` which provides utilities to compute variance-based feature importances. The main class is `VarianceImportanceCallback` for `tf.keras` models and there is an equivalent `VarianceImportanceTorch` helper for PyTorch networks. The method is described in the article [CR de Sá, *Variance-based Feature Importance in Neural Networks*](https://doi.org/10.1007/978-3-030-33778-0_24).

## Style
- Follow PEP8 and PEP257 standards.
- Use `tf.keras` APIs instead of the standalone `keras` package.
- Use Python `logging` for all user-facing messages. Do not add print statements or verbose flags.
- Provide type hints where practical and keep docstrings concise and descriptive.

## Package layout
- The public API lives in `variance_importance/callbacks.py` and is re-exported from `variance_importance/__init__.py`.
- The callback class is `VarianceImportanceCallback` and exposes the attribute `var_scores` along with the property `feature_importances_`.

## Testing
Run these commands after making changes:

```bash
python -m py_compile variance_importance/callbacks.py
python -m py_compile "variance-based feature importance in artificial neural networks.ipynb" 2>&1 | head
# Optionally check notebook conversion if `jupyter` is available
jupyter nbconvert --to script "variance-based feature importance in artificial neural networks.ipynb" --stdout | head
```

