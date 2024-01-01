## Introduction

In this example, we tackle a fundamental problem in machine learning: the classification of logical operations using perceptrons. Specifically, we aim to implement perceptrons to solve the logical AND and OR operations.

## Problem Overview

Logical gates, such as AND and OR, are essential components of digital circuits and represent binary operations. The AND gate outputs 1 if both of its inputs are 1; otherwise, it outputs 0. The OR gate outputs 1 if at least one of its inputs is 1; otherwise, it outputs 0. These gates are simple but fundamental in computing.

### Logical AND Table

| Input 1 | Input 2 | Output |
| ------- | ------- | ------ |
|    0    |    0    |   0    |
|    0    |    1    |   0    |
|    1    |    0    |   0    |
|    1    |    1    |   1    |

### Logical OR Table

| Input 1 | Input 2 | Output |
| ------- | ------- | ------ |
|    0    |    0    |   0    |
|    0    |    1    |   1    |
|    1    |    0    |   1    |
|    1    |    1    |   1    |


Our task is to train perceptrons to mimic the behavior of these logical gates. We'll use the AND and OR operations as examples to illustrate how perceptrons can learn and perform binary classification tasks.

## How Perceptrons Solve the Problem

Perceptrons are basic building blocks of neural networks and can be used for binary classification tasks. Here's how perceptrons solve the logical gate classification problem:

1. **Data Generation**: We define the inputs and corresponding outputs for the logical gates. For example, for the AND gate, we generate input-output pairs where the output is 1 only if both inputs are 1. Similarly, for the OR gate, the output is 1 if at least one input is 1.

2. **Data Splitting**: We split the generated dataset into training and testing sets. This allows us to train the perceptron on one subset of the data and evaluate its performance on unseen data to assess generalization.

3. **Perceptron Initialization**: We create separate perceptron instances for each logical gate. These perceptrons have two input features (corresponding to the binary inputs of the gates) and a single output (representing the predicted output).

4. **Training**: We train the perceptrons using a training loop. During each epoch, we iterate through the training data, compute the prediction of the perceptron, calculate the error (the difference between the predicted and actual outputs), and update the perceptron's weights and bias using gradient descent. This process continues for multiple epochs to minimize the error.

Training Steps Table (AND Gate):

| Input | Weights | Bias | Output | Error |
|-------|---------|------|--------|-------|
| [0,0] | [0,0]   | 0    |   0    |   0   |
| [0,1] | [0,0]   | 0    |   0    |   0   |
| [1,0] | [0,0]   | 0    |   0    |   0   |
| [1,1] | [0,0]   | 0    |   0    |   1   |
| [0,0] | [-0.01,-0.01] | 0.01 |   0    |   0   |
| [0,1] | [-0.01,-0.01] | 0.01 |   0.01 |  -0.01 |
| [1,0] | [-0.01,-0.01] | 0.01 |   0.01 |  -0.01 |
| [1,1] | [-0.01,-0.01] | 0.01 |   0.02 |   0.98 |

6. **Testing**: After training, we evaluate the performance of the perceptrons on the testing dataset. We calculate accuracy, which measures how well the perceptrons classify the inputs based on the learned logical gate rules.


By iterating through these steps, perceptrons learn to approximate the behavior of logical gates, and the testing phase assesses their accuracy in binary classification tasks.


### Predictions Using Trained Perceptron (Logical AND Gate)

- **For True (1) Input [1, 1]:**

  - Weighted Sum: `(1 * (-0.01)) + (1 * (-0.01)) + 0.01 = -0.01 - 0.01 + 0.01 = -0.01`
  - Output (After Activation): `1 / (1 + exp(-(-0.01))) ≈ 0.4975`
  
- **For False (0) Input [0, 1]:**

  - Weighted Sum: `(0 * (-0.01)) + (1 * (-0.01)) + 0.01 = -0.01`
  - Output (After Activation): `1 / (1 + exp(-(-0.01))) ≈ 0.4975`
