# Variance-based Feature Importance in Neural Networks / Deep Learning

This file provides a working example of how to measure the importance of features (inputs) in neural networks. 

This method is a new method to measure the relative importance of features in Artificial Neural Networks (ANN) models. Its underlying principle assumes that the more important a feature is, the more the weights, connected to the respective input neuron, will change during the training of the model. To capture this behavior, a running variance of every weight connected to the input layer is measured during training. For that, an adaptation of Welford's online algorithm for computing the online variance is proposed.

When the training is finished, for each input, the variances of the weights are combined with the final weights to obtain the measure of relative importance for each feature.

The file **variance-based feature importance in artificial neural networks.ipynb** includes the code to fully replicate the results obtained in the paper:
CR de Sá **Variance-based Feature Importance in NeuralNetworks**
22st International Conference on Discovery Science (DS 2019) Split, Croacia, October 28-30, 2019 (to appear)

