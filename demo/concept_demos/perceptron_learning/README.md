## Introduction to Artificial Neurons

Artificial neurons, also known as nodes, units, or perceptrons, are the fundamental building blocks of artificial neural networks. They are designed to mimic the behavior of biological neurons in the human brain.

### What is an Artificial Neuron?

An artificial neuron takes a set of input data, performs calculations on that data, and produces an output. Each input is associated with a weight, which represents its importance or influence on the output. Inputs are combined using a weighted sum, which includes an additional parameter called the bias or threshold.

Different types of artificial neurons employ different activation functions. A common activation function is the sigmoid function, which provides a smooth, continuous output that ranges between 0 and 1, and is especially useful for binary classification problems.

### Why are Artificial Neurons Important?

Artificial neurons are the essence of artificial neural networks. They allow us to model complex, non-linear relationships between inputs and outputs, making them capable of solving a wide variety of problems ranging from image recognition to natural language processing.

The ability to adjust weights and biases through learning algorithms allows the network to 'learn' from data, improving its performance over time. This learning process can be thought of as a balancing act in which the network adjusts the weights and biases in response to the difference between its current output and the expected output.

### How Do Artificial Neurons Work?

1. **Initialization**: Weights and biases are initialized, typically with small random values.

2. **Combination**: Each input is multiplied by its corresponding weight and the results are summed together, then the bias is added. This is also known as the weighted sum.

3. **Activation**: The result is passed through an activation function, which decides whether the artificial neuron should be activated or not based on the input it receives. For a perceptron, this function is the Heaviside Step function.

4. **Output**: The output from the activation function is the output of the neuron.

5. **Learning**: During the learning phase, an error value is calculated by comparing the output of the neuron to the expected output. This error value is then used to adjust the weights and biases in a process called backpropagation. The weights and biases are adjusted in such a way that the error is minimized, meaning that the output of the neuron gets closer to the expected output.

## Perceptron learning

We'll illustrate the learning process using a simple model that performs the AND operation. This is a binary operation where the output is 1 if and only if both inputs are 1.

Our model will use a sigmoid activation function, which provides a smooth, continuous output that ranges between 0 and 1.

We'll initialize our weights and bias randomly and use a simple learning rate of 0.1.

Our perceptron looks like this:

```
Input1 (x1) ----> * ----\
                          \
                           (Weighted Sum + Bias) ----> (Activation Function) ----> Output
                          /
Input2 (x2) ----> * ----/
```

With weights, bias and learning rate as:

- Weight for Input1 (w1): 0.3
- Weight for Input2 (w2): -0.1
- Bias (b): 0.0
- Learning rate (η): 0.1
- Activation Function: Heaviside Step function

The updates are calculated using the rule 

$$Δw = η * (target - output) * input$$

$$Δb = η * (target - output)$$

Let's run one training step on each possible input:

| Step | Input1 (x1) | Input2 (x2) | Target Output | Weighted Sum + Bias | Output  | w1 update | w2 update | b update |
|------|-------------|-------------|---------------|---------------------|---------|-----------|-----------|----------|
|  1   |     0       |     0       |      0        |         0.0         |  0.5    |    0.0    |    0.0    |  -0.05   |
|  2   |     0       |     1       |      0        |       -0.05         | ~0.487  |    0.0    |   -0.05   |  -0.05   |
|  3   |     1       |     0       |      0        |         0.25        | ~0.562  |   -0.05   |    0.0    |  -0.05   |
|  4   |     1       |     1       |      1        |         0.2         | ~0.550  |    0.05   |    0.05   |   0.05   |

After one training step for each input, our updated parameters are:

- Weight for Input1 (w1): 0.25
- Weight for Input2 (w2): 0.0
- Bias (b): -0.1

Note that the perceptron has not yet learned the AND operation perfectly. The output for the inputs (1,1) should be 1, but our perceptron predicts a value of ~0.550. Therefore, we would typically need to run multiple iterations (epochs) over the training data, updating the weights and bias each time, until the outputs are close enough to the target values, or until further training no longer improves the outputs.

## Decision Boundary and Its Relation to the Perceptron Prediction

In the case of a perceptron, the decision boundary is the hyperplane where the output of the activation function changes. For a binary classification problem with two classes, one class lies on one side of the hyperplane and the other class lies on the other side. 

For the activation function of a perceptron, the decision boundary occurs when the weighted sum of the inputs plus the bias is zero. Mathematically, this is where:

$$w_1x_1 + w_2x_2 + \cdots + w_nx_n + b = 0$$

The perceptron makes a prediction of 1 if the weighted sum is greater than zero and a prediction of 0 if it's less than zero. Therefore, the decision boundary is intimately related to the prediction of the perceptron. It is the point at which the perceptron switches its prediction from one class to the other.

Scaling up to a perceptron with more inputs, the decision boundary becomes a hyperplane in a higher-dimensional space. For a perceptron with n inputs, the decision boundary is defined by:

$$w_1x_1 + w_2x_2 + \cdots + w_nx_n + b = 0$$

So as the number of inputs (features) to the perceptron increases, the decision boundary becomes a hyperplane in an n-dimensional space. This means that the complexity of the decision boundary increases with the number of inputs, but the relationship between the perceptron's prediction and the decision boundary remains the same.

### Decision Boundary for the AND Gate Perceptron

For the AND gate example, we can calculate and plot the decision boundary. We have our perceptron with weights $w_1 = 0.25$, $w_2 = 0.0$, and $b = -0.1$.

Firstly, we solve the equation $w_1x + w_2y + b = 0$ for $y$ to get the equation of the line in terms of $y$.

$$y = -(w_1x + b) / w_2$$

Note that because our $w_2$ is 0, the equation simplifies to:

$$y = -b / w_2$$

which is undefined because we are dividing by zero. In this case, the decision boundary line is vertical and given by:

g$$x = -b / w_1$$

Substituting our perceptron's weights and bias into this equation gives:

$$x = -(-0.1) / 0.25 = 0.4$$

So, the decision boundary is a vertical line at $x = 0.4$. 

This implies that for inputs where $x_1 (\text{input1}) > 0.4$, the perceptron will output 1, and for inputs where $x_1 (\text{input1}) <= 0.4$, the perceptron will output 0.
