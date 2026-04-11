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

    def relu(self, z):
        return np.maximum(0, z)
    
    def relu_derivative(self, z):
        return (z > 0).astype(float)
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def sigmoid_derivative(self, z):
        s = self.sigmoid(z)
        return s * (1 - s)
    
    def tanh(self, z):
        return np.tanh(z)
    
    def tanh_derivative(self, z):
        return 1 - np.tanh(z) ** 2
    
    def soft_max(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def activate(self, z, derivative):
        if derivative:
            if self.activation == 'relu':
                return self.relu_derivative(z)
            elif self.activation == 'sigmoid':
                return self.sigmoid_derivative(z)
            elif self.activation == 'tanh':
                return self.tanh_derivative(z)
        else:
            if self.activation == 'relu':
                return self.relu(z)
            elif self.activation == 'sigmoid':
                return self.sigmoid(z)
            elif self.activation == 'tanh':
                return self.tanh(z)
    
    def forward(self, X):
        activations = [X]
        z_values = []
        
        for i in range(self.num_layers - 2):
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            z_values.append(z)
            a = self.activate(z, derivative=False) 
            activations.append(a)
        
        z = np.dot(activations[-1], self.weights[-1]) + self.biases[-1]
        z_values.append(z)
        a = self.soft_max(z)  
        activations.append(a)
        
        return activations, z_values
    
    def backward(self, X, Y, activations, z_values):
        m = X.shape[0]
        delta = activations[-1] - Y


        for i in range(len(self.weights) - 1, -1, -1):
            dW = np.dot(activations[i].T, delta) / m
            db = np.sum(delta, axis=0, keepdims=True) / m

            self.weights[i] -= self.learning_rate * dW
            self.biases[i] -= self.learning_rate * db

            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * self.activate(z_values[i - 1], derivative=True)


    def cross_entropy_loss(self, y_pred, y_true):
        m = y_true.shape[0]
        y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)
        loss = -np.sum(y_true * np.log(y_pred)) / m
        return loss
    
    def train(self, X_train, y_train, X_val, y_val, epochs, batch_size, verbose):
        n_samples = X_train.shape[0]
        
        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train[indices]
            
            epoch_loss = 0
            for i in range(0, n_samples, batch_size):
                X_batch = X_train_shuffled[i:i + batch_size]
                y_batch = y_train_shuffled[i:i + batch_size]
                
                activations, z_values = self.forward(X_batch)
                self.backward(X_batch, y_batch, activations, z_values)
                
                batch_loss = self.cross_entropy_loss(activations[-1], y_batch)
                epoch_loss += batch_loss
            
            epoch_loss /= (n_samples // batch_size)
            self.history['loss'].append(epoch_loss)
            
            train_predictions = self.predict(X_train)
            train_accuracy = np.mean(train_predictions == np.argmax(y_train, axis=1))
            self.history['accuracy'].append(train_accuracy)
            
            if X_val is not None and y_val is not None:
                val_activations, _ = self.forward(X_val)
                val_loss = self.cross_entropy_loss(val_activations[-1], y_val)
                val_predictions = self.predict(X_val)
                val_accuracy = np.mean(val_predictions == np.argmax(y_val, axis=1))
                
                self.history['val_loss'].append(val_loss)
                self.history['val_accuracy'].append(val_accuracy)
                
                if verbose:
                    print(f"Epoch {epoch + 1}/{epochs} - "
                          f"Loss: {epoch_loss:.4f} - Acc: {train_accuracy:.4f} - "
                          f"Val Loss: {val_loss:.4f} - Val Acc: {val_accuracy:.4f}")
            else:
                if verbose:
                    print(f"Epoch {epoch + 1}/{epochs} - "
                          f"Loss: {epoch_loss:.4f} - Acc: {train_accuracy:.4f}")
        
        return self.history

    def predict(self, X: np.ndarray) -> np.ndarray:
        activations, _ = self.forward(X)
        return np.argmax(activations[-1], axis=1)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        activations, _ = self.forward(X)
        return activations[-1]

    def save(self, filepath):
        def convert_arrays_to_lists(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, list):
                return [convert_arrays_to_lists(item) for item in obj]
            elif isinstance(obj, dict):
                return {k: convert_arrays_to_lists(v) for k, v in obj.items()}
            return obj

        model_data = {
            'layer_sizes': self.layer_size,
            'learning_rate': self.learning_rate,
            'activation': self.activation,
            'weights': convert_arrays_to_lists(self.weights),
            'biases': convert_arrays_to_lists(self.biases),
            'history': self.history
        }
    
        with open(filepath, 'w') as f:
            json.dump(model_data, f, indent=2)
        print(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath):
        def convert_lists_to_arrays(obj):
            if isinstance(obj, list):
                if obj and isinstance(obj[0], (int, float, list)):
                    return np.array([convert_lists_to_arrays(item) for item in obj])
                return [convert_lists_to_arrays(item) for item in obj]
            return obj

        with open(filepath, 'r') as f:
            model_data = json.load(f)
    
        model = cls(
            model_data['layer_sizes'], 
            model_data['learning_rate'],
            model_data['activation']
        )
    
        model.weights = [np.array(w) for w in model_data['weights']]
        model.biases = [np.array(b) for b in model_data['biases']]
        model.history = model_data['history']
    
        print(f"Model loaded from {filepath}")
        return model
    
    def summary(self):
        print("=" * 60)
        print("Neural Network Summary")
        print("=" * 60)
        
        total_params = 0
        for i in range(len(self.weights)):
            layer_params = self.weights[i].size + self.biases[i].size
            total_params += layer_params
            
            layer_name = "Input → Hidden 1" if i == 0 else \
                        f"Hidden {i} → Hidden {i+1}" if i < len(self.weights) - 1 else \
                        f"Hidden {i} → Output"
            
            print(f"{layer_name:30} | Params: {layer_params:,}")
        
        print("=" * 60)
        print(f"Total parameters: {total_params:,}")
        print(f"Activation function: {self.activation}")
        print(f"Learning rate: {self.learning_rate}")
        print("=" * 60)