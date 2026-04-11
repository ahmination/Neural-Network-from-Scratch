# Neural Network functions

```Python
def relu(self, z):
        return np.maximum(0, z)
```


```Python
    def relu_derivative(self, z):
        return (z > 0).astype(float)
```


![sigmoid](Sigmoid-Activation-Function.png "Sigmoid")
```Python
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
```


```Python
    def sigmoid_derivative(self, z):
        s = self.sigmoid(z)
        return s * (1 - s)
```


```Python

```

```Python

```

```Python

```

```Python

```