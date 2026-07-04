import numpy as np
import json

class Neuralnetwork:
    def __init__(self, layer_size, learning_rate, activation):
        self.layer_size = layer_size
        self.learning_rate = learning_rate
        self.activation = activation
        self.num_layers = len(layer_size)
        self.weights = []
        self.biases = []

        for i in range(self.num_layers - 1):
            w = np.random.randn(layer_size[i], layer_size[i + 1]) * np.sqrt(2.0 / layer_size[i])
            b = np.zeros((1, layer_size[i + 1]))
            self.weights.append(w)
            self.biases.append(b)
        
        self.history = {
            'loss': [],
            'accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }