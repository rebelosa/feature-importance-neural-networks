# Variance-based Feature Importance in Neural Networks / Deep Learning

This file provides a working example of how to measure the importance of features (inputs) in neural networks. 

This method is a new method to measure the relative importance of features in Artificial Neural Networks (ANN) models. Its underlying principle assumes that the more important a feature is, the more the weights, connected to the respective input neuron, will change during the training of the model. To capture this behavior, a running variance of every weight connected to the input layer is measured during training. For that, an adaptation of Welford's online algorithm for computing the online variance is proposed.

When the training is finished, for each input, the variances of the weights are combined with the final weights to obtain the measure of relative importance for each feature.

The file **variance-based feature importance in artificial neural networks.ipynb** includes the code to fully replicate the results obtained in the paper:

CR de Sá [**Variance-based Feature Importance in Neural Networks**](https://doi.org/10.1007/978-3-030-33778-0_24)  
22st International Conference on Discovery Science (DS 2019) Split, Croatia, October 28-30, 2019


## VIANN
#### Variance-based Feature Importance of Artificial Neural Networks
```python
class VarImpVIANN(keras.callbacks.Callback):
    def __init__(self, verbose=0):
        self.verbose = verbose
        self.n = 0
        self.M2 = 0.0

    def on_train_begin(self, logs={}, verbose = 1):
        if self.verbose:
            print("VIANN version 1.0 (Wellford + Mean) update per epoch")
        self.diff = self.model.layers[0].get_weights()[0]
        
    def on_epoch_end(self, batch, logs={}):
        currentWeights = self.model.layers[0].get_weights()[0]
        
        self.n += 1
        delta = np.subtract(currentWeights, self.diff)
        self.diff += delta/self.n
        delta2 = np.subtract(currentWeights, self.diff)
        self.M2 += delta*delta2
            
        self.lastweights = self.model.layers[0].get_weights()[0]

    def on_train_end(self, batch, logs={}):
        if self.n < 2:
            self.s2 = float('nan')
        else:
            self.s2 = self.M2 / (self.n - 1)
        
        scores = np.sum(np.multiply(self.s2, np.abs(self.lastweights)), axis = 1)
        
        self.varScores = (scores - min(scores)) / (max(scores) - min(scores))
        if self.verbose:
            print("Most important variables: ",
                  np.array(self.varScores).argsort()[-10:][::-1])
```
## Example of usage
```python
VIANN = VarImpVIANN(verbose=1)

model = Sequential()
model.add(Dense(50, input_dim=input_dim, activation='relu', kernel_initializer='normal', kernel_regularizer=l2(0.01)))
model.add(Dense(100, activation='relu', kernel_initializer='normal', kernel_regularizer=l2(0.01)))
model.add(Dense(50, activation='relu', kernel_initializer='normal', kernel_regularizer=l2(0.01)))
model.add(Dense(5, activation='softmax', kernel_initializer='normal'))

model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

model.fit(X, Y, validation_split=0.05, epochs=30, batch_size=64, shuffle=True, 
      verbose=1, callbacks = [VIANN])
      
print(VIANN.varScores)
[0.75878453 0.2828902  0.85303473 0.6568499  0.07119488 0.20491114
 0.3517472  0.844915   0.03618119 0.03033427 0.4099664  0.
 0.3236221  0.21142973 0.00467986 0.02231793 0.0134031  0.03483544
 0.02274348 0.02686052 1.         0.6406668  0.80592436 0.6484351
 0.3022079  0.35150513 0.54522735 0.8139651  0.24560207 0.04865947]
```

