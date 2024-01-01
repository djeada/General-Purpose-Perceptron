import random
from typing import Callable

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from neura_command.network.feed_forward_network import FeedForwardNetwork
from neura_command.network_utils.activation_functions import ReLUActivation
from neura_command.network_utils.loss_functions import MSELoss
from neura_command.neuron.perceptron import Perceptron


def main():
    # Initialize constants and functions
    input_size = 784
    hidden_layer_size = 64
    output_size = 10

    # Create Perceptron layers
    layers = [
        Perceptron(
            input_size,
            hidden_layer_size,
            activation_func=ReLUActivation(),
            learning_rate=0.01,
        ),
        Perceptron(
            hidden_layer_size,
            hidden_layer_size,
            activation_func=ReLUActivation(),
            learning_rate=0.01,
        ),
        Perceptron(
            hidden_layer_size,
            output_size,
            activation_func=ReLUActivation(),
            learning_rate=0.01,
        ),
    ]

    # Create and train the network
    network = FeedForwardNetwork(layers, loss_func=MSELoss())

    mnist = fetch_openml("mnist_784")
    X = mnist.data.astype("float32") / 255
    y = mnist.target.astype("int")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    train_labels = np.eye(10)[y_train]
    test_labels = np.eye(10)[y_test]

    network.train(X_train, train_labels, epochs=10, batch_size=10)
    test_loss = network.calculate_loss(X_test, test_labels)
    print(f"Test Loss: {test_loss}")

    # Step 1: Make predictions on the test set
    predicted_labels = np.argmax(network.forward(X_test), axis=1)

    # Assuming your test_labels are one-hot encoded, convert them to class labels
    true_labels = np.argmax(test_labels, axis=1)

    # Step 2: Compare predictions to true labels
    correct_predictions = np.sum(predicted_labels == true_labels)

    # Step 3: Calculate accuracy
    accuracy = (correct_predictions / len(true_labels)) * 100

    print(f"Accuracy on Test Set: {accuracy}%")

    # Convert X_test to a NumPy array if it's a DataFrame
    X_test_array = X_test.to_numpy() if isinstance(X_test, pd.DataFrame) else X_test

    # Define the number of samples to display
    num_samples = 10

    # Increase the overall figure size
    plt.figure(figsize=(15, 6))  # You can adjust the size as needed

    # Generate random indices to select samples
    sample_indices = random.sample(range(len(X_test_array)), num_samples)

    # Create a subplot for each sample in a 2x5 grid
    for i, index in enumerate(sample_indices):
        # Extract and reshape the image
        image = X_test_array[index].reshape(28, 28)

        # Get the true label
        true_label = np.argmax(test_labels[index])

        # Make a prediction using the network
        predicted_label = np.argmax(network.forward(X_test_array[index].reshape(1, -1)))

        # Create a subplot within a 2x5 grid
        plt.subplot(2, num_samples // 2, i + 1)

        # Display the image in grayscale
        plt.imshow(image, cmap="gray")

        # Set the title with predicted and true labels, adjust font size if needed
        plt.title(f"Pred: {predicted_label}\nTrue: {true_label}", fontsize=10)

        # Turn off axis labels
        plt.axis("off")

    # Adjust spacing between subplots
    plt.subplots_adjust(hspace=0.5, wspace=0.3)

    plt.show()


if __name__ == "__main__":
    main()
