## Problem Introduction

We have a simple linear regression problem where we want to model a linear relationship between an input variable `X` and an output variable `y`. The underlying relationship is given by:

$$y = 2X + 1$$

However, we also introduce some random noise to simulate real-world data. Our goal is to train a perceptron to approximate this linear relationship and make predictions based on the input `X`. We will visualize the data and the perceptron's predictions using Matplotlib.

## How Perceptron Solves the Problem

A perceptron is a basic building block of a neural network, and in this context, we use it as a simple linear model. Here's how the perceptron solves the linear regression problem:

1. **Data Generation**: We generate random data points `(X, y)` that follow a linear relationship `y = 2X + 1` but with added random noise to simulate real-world data.

2. **Perceptron Initialization**: We create an instance of the `Perceptron` class, which represents a single neuron in a neural network. This neuron has one input and one output, making it suitable for linear regression.

3. **Training the Perceptron**: We train the perceptron using a training loop. During each epoch, we iterate through the data points, compute the prediction of the perceptron, calculate the error (the difference between the prediction and the actual target value), and adjust the perceptron's weights and bias using gradient descent. This process continues for multiple epochs to minimize the error.

4. **Making Predictions**: After training, the perceptron has learned to approximate the linear relationship between `X` and `y`. We can use it to make predictions for new input values. We create a set of input values (`X_test`), and for each input, we obtain a prediction from the perceptron.

5. **Visualization**: We visualize the data points as blue dots and the perceptron's predictions as a red line using Matplotlib. This visualization helps us see how well the perceptron has learned to approximate the linear regression model.


