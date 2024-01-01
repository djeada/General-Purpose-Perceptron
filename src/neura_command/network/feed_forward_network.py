import numpy as np

from neura_command.network_utils.loss_functions import LossFunction


class FeedForwardNetwork:
    def __init__(self, layers, loss_func):
        self.layers = layers
        # Ensure the passed loss_func is an instance of LossFunction
        if not isinstance(loss_func, LossFunction):
            raise ValueError("loss_func must be a subclass of LossFunction")
        self.loss_func = loss_func

    def forward(self, X):
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def backward(self, y_true, y_pred):
        error = self.loss_func.derivative(y_pred, y_true)
        for layer in reversed(self.layers):
            error = layer.backward(error)[0]
            error = error @ layer.weights[:-1].T  # Exclude the bias weights

    def update_weights(self):
        for layer in self.layers:
            layer.update_weights()

    def train(
        self,
        X,
        y,
        epochs,
        batch_size,
        learning_rate_schedule=None,
        X_val=None,
        y_val=None,
    ):
        for epoch in range(epochs):
            # Apply learning rate schedule if provided
            if learning_rate_schedule:
                for layer in self.layers:
                    layer.learning_rate = learning_rate_schedule(epoch)

            # Training loop
            epoch_loss = 0
            for i in range(0, len(X), batch_size):
                X_batch = X[i : i + batch_size]
                y_batch = y[i : i + batch_size]
                output = self.forward(X_batch)
                self.backward(y_batch, output)
                self.update_weights()
                epoch_loss += np.mean(self.loss_func.compute(y_batch, output))

            # Average loss for the epoch
            epoch_loss /= len(X) / batch_size
            print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {epoch_loss}")

            # Validation
            if X_val is not None and y_val is not None:
                val_loss = self.calculate_loss(X_val, y_val)
                print(f"Validation Loss: {val_loss}")

    def calculate_loss(self, X, y):
        predictions = self.forward(X)
        return np.mean(self.loss_func.compute(y, predictions))
