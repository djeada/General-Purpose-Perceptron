import numpy as np
import matplotlib.pyplot as plt

# Activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)  # This is correct as x is the output of sigmoid


# Error function
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


class Perceptron:
    def __init__(self, input_size, output_size, learning_rate=0.1):
        # Adjust weight and bias initialization
        self.weights = np.random.randn(input_size, output_size)
        self.bias = np.random.randn(1, output_size)
        self.learning_rate = learning_rate
        self.output = None
        self.input = None

    def forward(self, X):
        self.input = X
        self.output = sigmoid(np.dot(X, self.weights) + self.bias)
        return self.output

    def backward(self, error):
        delta = error * sigmoid_derivative(self.output)
        weight_gradient = np.dot(self.input.T, delta)
        bias_gradient = np.sum(delta, axis=0, keepdims=True)
        return delta, weight_gradient, bias_gradient

    def update_weights(self, weight_gradient, bias_gradient):
        self.weights += weight_gradient * self.learning_rate
        self.bias += bias_gradient * self.learning_rate


class NeuralNetwork:
    def __init__(self, layer_sizes, learning_rate=0.1):
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            self.layers.append(
                Perceptron(layer_sizes[i], layer_sizes[i + 1], learning_rate)
            )

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, y):
        error = y - self.layers[-1].output
        for layer in reversed(self.layers):
            delta, weight_gradient, bias_gradient = layer.backward(error)
            layer.update_weights(weight_gradient, bias_gradient)
            error = np.dot(delta, layer.weights.T)

    def train(self, X, y, epochs):
        errors = []
        for _ in range(epochs):
            output = self.forward(X)
            self.backward(y)
            errors.append(mean_squared_error(y, output))
        return errors


# XOR input and output
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Neural Network initialization and training
nn = NeuralNetwork([2, 2, 1], learning_rate=0.1)
errors = nn.train(X, y, 20000)
print(min(errors))
# Plotting the error over epochs
plt.plot(errors)
plt.xlabel("Epochs")
plt.ylabel("Mean Squared Error")
plt.title("Training Error over Epochs with Adjusted Network Architecture")
plt.show()
