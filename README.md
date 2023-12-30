# NeuraCommand
NeuraCommand is an innovative CLI tool designed for creating, training, and deploying neural networks. It provides a seamless interface for handling complex neural network architectures, including perceptrons with multiple layers. With NeuraCommand, users can efficiently load data, train models, and use them for predictions, saving the trained models for future use or further training.

TODO:

Classifying a non-linearly separable dataset

Consider the following data points that cannot be linearly separated:

`from sklearn.datasets import make_moons`


## Features
- **Easy-to-use CLI**: Intuitive command-line interface for all operations.
- **Flexible Network Configuration**: Define custom neural network architectures with various layers.
- **Matrix Input Compatibility**: Load and train networks with any matrix data.
- **Save and Load Models**: Save trained models and load them for later use or additional training.
- **Real-time Predictions**: Utilize trained networks for immediate predictions.

* should allow to display outputs of all layers and generally the whole process
* should allow to select trained model
* should allow to make only a prediction


## Installation

```shell
pip install gp_perceptron_framework
```
## Usage

Basic usage of NeuraCommand:

```bash

# Create a new neural network
neuracommand create-network --layers 3 --neurons 64

# Load data for training
neuracommand load-data --file path/to/data.csv

# Train the network
neuracommand train --epochs 100

# Save the trained model
neuracommand save-model --output model.ncmd

# Make predictions
neuracommand predict --input sample_input.csv
```

## Quickstart

```python
from gp_perceptron_framework import GeneralPurposePerceptron

# Define a new network
network = GeneralPurposePerceptron()

# Add layers
network.add_layer(neuron_count=5)  # Hidden layer with 5 neurons
network.add_layer(neuron_count=3)  # Output layer with 3 neurons

# Train the network
data = [([0.1, 0.2, 0.3], [0, 1, 0])]  # List of (input, output) pairs
network.train(data, epochs=10)

# Make predictions
inputs = [0.1, 0.2, 0.3]
outputs = network.predict(inputs)
print(outputs)  # Prints: [0.2, 0.8, 0.1]

# Evaluate the network
performance = network.evaluate(data)
print(performance)  # Prints: 0.05 (assuming the loss is 0.05)
```


## Refrences:

* https://playground.tensorflow.org/




