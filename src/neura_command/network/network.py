from neura_command.layer.layer import Layer


class Network:
    def __init__(self):
        self.layers = []

    def add_layer(self, layer: Layer):
        self.layers.append(layer)

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, error):
        for layer in reversed(self.layers):
            error = layer.backward(error)

    def update_weights(self, learning_rate):
        for layer in self.layers:
            layer.update_weights(learning_rate)

    def train(self, X, Y, epochs, batch_size, learning_rate, verbose=False):
        for epoch in range(epochs):
            for i in range(0, len(X), batch_size):
                X_batch = X[i : i + batch_size]
                Y_batch = Y[i : i + batch_size]

                # Forward pass
                output = self.forward(X_batch)

                # Compute error (Assuming a function to calculate error exists)
                error = self.compute_error(Y_batch, output)

                # Backward pass
                self.backward(error)

                # Update weights
                self.update_weights(learning_rate)

            if verbose and epoch % 100 == 0:
                # Compute loss for monitoring (Assuming a function to calculate loss exists)
                loss = self.compute_loss(Y, self.forward(X))
                print(f"Epoch {epoch}, Loss: {loss}")

    def compute_error(self, Y_true, Y_pred):
        # Implement or assume an error computation function
        pass

    def compute_loss(self, Y_true, Y_pred):
        # Implement or assume a loss computation function
        pass
