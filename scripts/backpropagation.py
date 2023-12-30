import numpy as np
from matplotlib import pyplot as plt


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_derivative(z):
    return z * (1 - z)


class Neuron:
    def __init__(self, n_inputs):
        self.weights = np.random.rand(n_inputs)
        self.bias = np.random.rand()

    def calculate_output(self, inputs):
        self.inputs = np.array(inputs)
        self.output = sigmoid(np.dot(self.weights, self.inputs) + self.bias)
        return self.output

    def calculate_delta(self, target_output=None, weighted_sum_delta=0):
        if target_output is not None:
            self.delta = (self.output - target_output) * sigmoid_derivative(self.output)
        else:
            self.delta = weighted_sum_delta * sigmoid_derivative(self.output)

    def update_weights(self, learning_rate):
        self.weights -= learning_rate * self.delta * self.inputs
        self.bias -= learning_rate * self.delta


class NeuralNetwork:
    def __init__(self, n_inputs, n_hidden, n_outputs, learning_rate):
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_outputs = n_outputs
        self.learning_rate = learning_rate

        self.hidden_layer = [Neuron(n_inputs) for _ in range(n_hidden)]
        self.output_layer = [Neuron(n_hidden) for _ in range(n_outputs)]

    def forward_propagate(self, inputs):
        hidden_layer_outputs = [
            neuron.calculate_output(inputs) for neuron in self.hidden_layer
        ]
        return [
            neuron.calculate_output(hidden_layer_outputs)
            for neuron in self.output_layer
        ]

    def back_propagate(self, target_outputs):
        for i in range(self.n_outputs):
            self.output_layer[i].calculate_delta(target_outputs[i])
        for i in range(self.n_hidden):
            weighted_sum_delta = sum(
                neuron.weights[i] * neuron.delta for neuron in self.output_layer
            )
            self.hidden_layer[i].calculate_delta(weighted_sum_delta=weighted_sum_delta)

    def update_weights(self):
        for neuron in self.hidden_layer + self.output_layer:
            neuron.update_weights(self.learning_rate)

    def fit(self, training_inputs, training_outputs, epochs):
        for epoch in range(epochs):
            for inputs, outputs in zip(training_inputs, training_outputs):
                self.forward_propagate(inputs)
                self.back_propagate(outputs)
                self.update_weights()

    def predict(self, inputs):
        outputs = self.forward_propagate(inputs)
        return [
            np.round(output) for output in outputs
        ]  # round to 0 or 1 for binary classification


inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
targets = np.array([[0], [1], [1], [1]])

nn = NeuralNetwork(n_inputs=2, n_hidden=2, n_outputs=1, learning_rate=0.5)
nn.fit(inputs, targets, epochs=1500)


def plot(nn, inputs, targets):
    if inputs.shape[1] != 2:
        raise ValueError("Can only plot for 2 inputs.")
    # Generate a grid of points in the input space
    x1_values = np.linspace(-0.5, 1.5, 100)
    x2_values = np.linspace(-0.5, 1.5, 100)
    grid = np.array([[x1, x2] for x1 in x1_values for x2 in x2_values])

    # Compute the output for each point in the grid
    outputs = np.array([nn.forward_propagate(point)[0] for point in grid])
    outputs = np.round(outputs).astype(int)  # round to 0 or 1 for binary classification

    # Scatter plot of the grid points, colored by the output
    plt.scatter(grid[:, 0], grid[:, 1], c=outputs, cmap="viridis", alpha=0.5)

    # Scatter plot of the inputs, colored by the target
    plt.scatter(inputs[:, 0], inputs[:, 1], c=targets, cmap="viridis", edgecolors="k")
    plt.title("Decision Boundary")
    plt.grid(True)
    plt.show()


plot(nn, inputs, targets.ravel())
