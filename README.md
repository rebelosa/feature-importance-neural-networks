# Variance-based Feature Importance in Neural Networks / Deep Learning

This file provides a working example of how to measure the importance of features (inputs) in neural networks. 

This method is a new method to measure the relative importance of features in Artificial Neural Networks (ANN) models. Its underlying principle assumes that the more important a feature is, the more the weights, connected to the respective input neuron, will change during the training of the model. To capture this behavior, a running variance of every weight connected to the input layer is measured during training. For that, an adaptation of Welford's online algorithm for computing the online variance is proposed.

When the training is finished, for each input, the variances of the weights are combined with the final weights to obtain the measure of relative importance for each feature.

The file **variance-based feature importance in artificial neural networks.ipynb** includes the code to fully replicate the results obtained in the paper:

CR de Sá [**Variance-based Feature Importance in Neural Networks**](https://doi.org/10.1007/978-3-030-33778-0_24)  
22st International Conference on Discovery Science (DS 2019) Split, Croatia, October 28-30, 2019


## VIANN
#### Variance-based Feature Importance of Artificial Neural Networks

This repository exposes the feature importance callback as a small Python package named `variance_importance`.

```python
from variance_importance import VarianceImportanceCallback

import logging

logging.basicConfig(level=logging.INFO)

VIANN = VarianceImportanceCallback()
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
          verbose=1, callbacks=[VIANN])

print(VIANN.var_scores)
```

## Comparing with Random Forest

To verify the variance-based scores, run `compare_feature_importance.py`. The
script trains a small neural network on the Iris dataset and compares the scores
with those from a `RandomForestClassifier`.

```bash
python compare_feature_importance.py
```
