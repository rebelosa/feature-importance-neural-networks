# Variance-based Feature Importance in Neural Networks / Deep Learning

This file provides a working example of how to measure the importance of features (inputs) in neural networks. 

This method is a new method to measure the relative importance of features in Artificial Neural Networks (ANN) models. Its underlying principle assumes that the more important a feature is, the more the weights, connected to the respective input neuron, will change during the training of the model. To capture this behavior, a running variance of every weight connected to the input layer is measured during training. For that, an adaptation of Welford's online algorithm for computing the online variance is proposed.

When the training is finished, for each input, the variances of the weights are combined with the final weights to obtain the measure of relative importance for each feature.

The file **variance-based feature importance in artificial neural networks.ipynb** includes the code to fully replicate the results obtained in the paper:

CR de Sá [**Variance-based Feature Importance in Neural Networks**](https://doi.org/10.1007/978-3-030-33778-0_24)  
22st International Conference on Discovery Science (DS 2019) Split, Croatia, October 28-30, 2019


## VIANN
#### Variance-based Feature Importance of Artificial Neural Networks

This repository exposes the feature importance callback as a small Python package named `neural-feature-importance`.
It will automatically track the first layer that contains trainable weights so you can use it with models that start with an `InputLayer` or other preprocessing layers.
There is also a helper for PyTorch models that follows the same API.

Install with pip and select the extras that match your framework:

```bash
pip install "neural-feature-importance[tensorflow]"  # for Keras
pip install "neural-feature-importance[torch]"       # for PyTorch
```

The package uses `setuptools_scm` to derive its version from Git tags. Access it
via:

```python
from neural_feature_importance import __version__

print(__version__)
```

```python
from neural_feature_importance import VarianceImportanceCallback, AccuracyMonitor

import logging

logging.basicConfig(level=logging.INFO)

VIANN = VarianceImportanceCallback()
monitor = AccuracyMonitor(baseline=0.95)
```

For a PyTorch model, use ``VarianceImportanceTorch`` and call its
``on_train_begin``, ``on_epoch_end`` and ``on_train_end`` methods inside your
training loop:

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

Use this callback during model training:

```python
model = Sequential()
model.add(Dense(50, input_dim=input_dim, activation='relu', kernel_initializer='normal', kernel_regularizer=l2(0.01)))
model.add(Dense(100, activation='relu', kernel_initializer='normal', kernel_regularizer=l2(0.01)))
model.add(Dense(50, activation='relu', kernel_initializer='normal', kernel_regularizer=l2(0.01)))
model.add(Dense(5, activation='softmax', kernel_initializer='normal'))

model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
model.fit(X, Y, validation_split=0.05, epochs=30, batch_size=64, shuffle=True,
          verbose=1, callbacks=[VIANN, monitor])

print(VIANN.var_scores)
```

## Comparing with Random Forest

To verify the variance-based scores, run `compare_feature_importance.py`. The
script trains a small neural network on the Iris dataset and compares the scores
with those from a `RandomForestClassifier`.

```bash
python compare_feature_importance.py
```

For a larger experiment across several datasets, run `full_experiment.py`. The script builds a simple network for each dataset, applies the `AccuracyMonitor` for early stopping, and prints the correlation between neural network importances and a random forest baseline.

```bash
python full_experiment.py
```
