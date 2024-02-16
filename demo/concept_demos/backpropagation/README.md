# Multilayer Perceptrons and Neural Networks

Multilayer perceptrons (MLPs) are a type of artificial neural network consisting of multiple layers of neurons, or perceptrons. In an MLP, each layer is fully connected to the next, meaning every neuron in a layer receives input from all neurons of the previous layer and sends its output to all neurons of the subsequent layer.

## Structure of a Multilayer Perceptron

An MLP generally consists of an input layer, one or more hidden layers, and an output layer:

1. **Input Layer:** The input layer contains neurons that accept input and transmit it to the next layer. Typically, each neuron corresponds to one feature in the dataset. For instance, in an image, each pixel could represent an input neuron.

2. **Hidden Layer(s):** The hidden layers perform computations on the inputs received and relay the results to the next layer. The term "hidden" is used because these layers are neither directly observable from the input nor from the output. The number of hidden layers and the number of neurons in each hidden layer are modifiable parameters and can significantly influence the learning capacity of the network.

3. **Output Layer:** The output layer generates the final result of the computations. The number of neurons in this layer corresponds to the number of possible output values. For example, for binary classification, there would be two neurons in the output layer.

Each neuron in an MLP takes weighted inputs, applies an activation function, and passes the output to the next layer:

```
Input1 (x1) ----> * ----\
                          \
                           (Weighted Sum + Bias) ----> (Activation Function) ----> Output
                          /
Input2 (x2) ----> * ----/
```

This structure is similar to the one we examined in the context of the simple perceptron model.

## How Does an MLP Work?

- Inputs are fed into the layers to produce output.
- Every node from the previous layer is connected to each node in the next layer.
- Each node aggregates the input from the previous layer using the formula:

$$y_i = \sum_{j=1}^N w_{ij}x_j + b_i$$

- The result is then passed through an activation function, such as the sigmoid function: $$\phi = \frac{1}{1+e^{-x}}$$
  
- The MLP starts with a certain set of weights and biases (collectively represented by $\theta$), and the goal is to adjust these such that the output in the last layer closely matches the expected output. This is done using a learning algorithm, such as gradient descent:

$$\theta = \theta - \alpha \nabla J(\theta)$$

where $\alpha$ is the learning rate and $\nabla J(\theta)$ is the gradient of the loss function $J(\theta)$ with respect to the parameters $\theta$.

- An initial measure of loss can be calculated as:

$$J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_{\theta}(x^{(i)}) - y^{(i)})^2$$

where $h_{\theta}(x)$ is our model's prediction and $y$ is the true value.

- However, cross-entropy often provides a better measure of loss, especially for classification tasks:

$$J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} [y^{(i)} \ln(h_{\theta}(x^{(i)})) + (1-y^{(i)}) \ln(1-h_{\theta}(x^{(i)}))]$$

where $h_{\theta}(x)$ is the predicted probability of the positive class, $y$ is the true label (1 for the positive class, 0 for the negative class), and $m$ is the number of training examples.

## From Perceptrons to Neural Networks

An artificial neural network (ANN) is a system of interconnected artificial neurons or perceptrons, inspired by the biological neural networks that constitute animal brains. 

A single perceptron can learn to make predictions for simple binary classification tasks. However, many real-world problems are not linearly separable and require more complex decision boundaries. By combining perceptrons into a multilayered network, we can model and learn these more complex relationships.

Neural networks are effective due to their ability to learn hierarchical representations. For instance, in image recognition, lower layers might learn to recognize edges and textures, middle layers might learn to recognize more complex shapes, and higher layers might learn to recognize high-level features such as objects or faces.

Each neuron in a network takes the outputs of the previous layer, applies a set of weights to these inputs, adds a bias term, and then passes the result through an activation function. The weights and biases in the network are the parameters that get adjusted during training. The training process involves feeding data through the network (forward propagation), comparing the output of the network with the true output, calculating a loss, and then adjusting the weights and biases to minimize this loss (backpropagation).

In summary, neural networks are powerful computational models that consist of interconnected perceptrons. They can model complex, non-linear relationships and learn hierarchical representations, which makes them suitable for a wide range of tasks in machine learning and artificial intelligence.

## Backpropagation

Backpropagation is the method used to calculate the gradient of the loss function with respect to the weights and biases, $\theta = \{w, b\}$, of the MLP. This gradient is then used in the gradient descent update rule to adjust the parameters of the network and minimize the loss.

Here's a basic overview of how backpropagation works:

1. **Forward Propagation:** First, we perform a forward pass through the network to compute the output for each neuron from the input layer, through the hidden layers, to the output layer. This output is our model's prediction, $h_{\theta}(x^{(i)})$.

2. **Compute Loss:** We then compute the loss, $J(\theta)$, using the true output value, $y^{(i)}$, and our model's prediction, $h_{\theta}(x^{(i)})$.

3. **Backward Propagation:** Starting from the output layer and moving backward through the hidden layers, we compute the error derivative for each neuron, which represents how much that neuron contributed to the total error. 

For the output layer, the error derivative (also known as the delta) for a given neuron $i$ is:

$$\delta^{(L)}_i = \frac{\partial J}{\partial a^{(L)}_i} \cdot \phi'(z^{(L)}_i)$$

where $L$ denotes the output layer, $a^{(L)}_i$ is the activation of the $i^{th}$ neuron in the output layer, $\phi'(z^{(L)}_i)$ is the derivative of the activation function applied to the weighted sum of inputs of the neuron, and $\frac{\partial J}{\partial a^{(L)}_i}$ is the derivative of the loss function with respect to the activation of the neuron.

For the hidden layers, the error derivative for a given neuron $i$ in layer $l$ is:

$$\delta^{(l)}_i = \left(\sum_{j=1}^{N} w^{(l+1)}_{ij} \delta^{(l+1)}_j\right) \cdot \phi'(z^{(l)}_i)$$

where $w_{ij}^{(l+1)}$ is the weight connecting neuron $i$ in layer $l$ to neuron $j$ in layer $l+1$.

4. **Update Weights and Biases:** Finally, we use the calculated error derivatives to update the weights and biases via the gradient descent rule:

$$w_{ij}^{(l)} = w_{ij}^{(l)} - \alpha \delta_j^{(l+1)} a_i^{(l)}$$

$$b_{i}^{(l)} = b_{i}^{(l)} - \alpha \delta_i^{(l+1)}$$

where $\alpha$ is the learning rate, $a_i^{(l)}$ is the activation of the $i^{th}$ neuron in layer $l$, and $\delta_j^{(l+1)}$ is the error derivative of the $j^{th}$ neuron in layer $l+1$.

This process of forward propagation, loss computation, backpropagation, and parameter update is repeated for each batch of data in our training set until the entire set has been processed. This constitutes one epoch of training. We typically repeat this process for many epochs until the loss on our validation set stops improving.

## Example

Let's create a simple example of a multilayer perceptron (MLP) on paper. Our MLP will have an input layer with 2 neurons, a hidden layer with 2 neurons, and an output layer with 1 neuron. We will use the sigmoid activation function for all neurons. Our goal is to train the MLP to perform the XOR operation.

The network architecture can be represented as:

```
Input1 (x1) ----> o ----\           o ----\
                          \        /        \
                           [Hidden Layer]   [Output Layer] ----> Output
                          /        \        /
Input2 (x2) ----> o ----/           o ----/
```

Let's initialize the weights and biases with random values:

Weights for input-to-hidden connections:
- w1: 0.5
- w2: 0.6
- w3: 0.3
- w4: 0.4

Bias for hidden layer:
- b1: 0.1
- b2: 0.2

Weights for hidden-to-output connections:
- w5: 0.7
- w6: 0.8

Bias for output layer:
- b3: 0.3

We will use a learning rate of 0.1. Now let's calculate the forward and backward pass for the input (0, 1), which has a target output of 1 (since XOR(0, 1) = 1).

## orward pass:

Calculate the weighted sum for hidden neurons:

- $z1 = w1 * x1 + w3 * x2 + b1 = 0.5 * 0 + 0.3 * 1 + 0.1 = 0.4$

- $z2 = w2 * x1 + w4 * x2 + b2 = 0.6 * 0 + 0.4 * 1 + 0.2 = 0.6$

Apply the sigmoid activation function:

- $h1 = \text{sigmoid}(z1) = \text{sigmoid}(0.4) ≈ 0.598$

- $h2 = \text{sigmoid}(z2) = \text{sigmoid}(0.6) ≈ 0.646$

Calculate the weighted sum for the output neuron:

$$
z3 = w5 * h1 + w6 * h2 + b3 = 0.7 * 0.598 + 0.8 * 0.646 + 0.3 ≈ 1.264
$$

Apply the sigmoid activation function:

$$
\text{output} = \text{sigmoid}(z3) = \text{sigmoid}(1.264) ≈ 0.779
$$

## Backward pass:

Calculate the output layer error:

$$
δ_{\text{output}} = (\text{target} - \text{output}) * \text{sigmoid derivative}(z3) = (1 - 0.779) * \text{sigmoid derivative}(1.264) ≈ 0.052
$$

Calculate the hidden layer errors:

$$
δ_{h1} = δ_{\text{output}} * w5 * \text{sigmoid derivative}(z1) ≈ 0.052 * 0.7 * \text{sigmoid derivative}(0.4) ≈ 0.006
$$

$$
δ_{h2} = δ_{\text{output}} * w6 * \text{sigmoid derivative}(z2)         ≈ 0.052 * 0.8 * \text{sigmoid derivative}(0.6) ≈ 0.009
$$

Update the weights and biases using the calculated errors and learning rate (η = 0.1):

- $Δw5 = η * δ_{\text{output}} * h1 ≈ 0.1 * 0.052 * 0.598 ≈ 0.003$
- $Δw6 = η * δ_{\text{output}} * h2 ≈ 0.1 * 0.052 * 0.646 ≈ 0.003$
- $Δb3 = η * δ_{\text{output}} ≈ 0.1 * 0.052 = 0.005$
- $Δw1 = η * δ_{h1} * x1 ≈ 0.1 * 0.006 * 0 = 0.0$
- $Δw2 = η * δ_{h2} * x1 ≈ 0.1 * 0.009 * 0 = 0.0$
- $Δw3 = η * δ_{h1} * x2 ≈ 0.1 * 0.006 * 1 = 0.001$
- $Δw4 = η * δ_{h2} * x2 ≈ 0.1 * 0.009 * 1 = 0.001$
- $Δb1 = η * δ_{h1} ≈ 0.1 * 0.006 = 0.001$
- $Δb2 = η * δ_{h2} ≈ 0.1 * 0.009 = 0.001$

So, the new weights and biases after the first training step are:

- $w1: 0.5 + 0.0 = 0.5$
- $w2: 0.6 + 0.0 = 0.6$
- $w3: 0.3 + 0.001 = 0.301$
- $w4: 0.4 + 0.001 = 0.401$
- $b1: 0.1 + 0.001 = 0.101$
- $b2: 0.2 + 0.001 = 0.201$
- $w5: 0.7 + 0.003 = 0.703$
- $w6: 0.8 + 0.003 = 0.803$
- $b3: 0.3 + 0.005 = 0.305$

## XOR - Exclusive OR Gate

The XOR gate is a fundamental building block in digital circuits. It operates with two inputs and one output, and the output is 'true' if and only if the number of true inputs is odd.

![gradient_evolution](https://github.com/djeada/NeuraCommand/assets/37275728/bfb8a529-ee46-4f9e-8f44-f90a6a5b7485)

![trainning_over_epochs](https://github.com/djeada/NeuraCommand/assets/37275728/24171511-fafe-4d9b-a36b-c02d05eed17a)

![decision_boundary](https://github.com/djeada/NeuraCommand/assets/37275728/b2a5334a-b5cb-4238-9cd4-0e1e49929d35)

Truth table for XOR:

| Input1 (x) | Input2 (y) | Output (z) |
| ---------- | ---------- | ---------- |
| 1          | 0          | 1          |
| 0          | 1          | 1          |
| 0          | 0          | 0          |
| 1          | 1          | 0          |

We represent the XOR gate as a function $f$ of two intermediary values $u$ and $v$, which are functions of $x$ and $y$.

```
x - u \
        f
y - v /
```

The intermediary values $u$ and $v$ are defined as follows, using the sigmoid function for activation:

$$u = \frac{1}{1 + e^{a_{11}x + a_{21}y + c_1}}$$

$$v = \frac{1}{1 + e^{a_{21}x + a_{22}y + c_2}}$$

The final function $f$ is defined as:

$$f = \frac{1}{1 + e^{b_1u + b_2v + c_3}}$$

Where $e$ is the base of the natural logarithm. The goal is to find parameters that minimize the loss function. The loss function can be defined as the mean squared error:

$$D = \frac{1}{2} (f^{x_1y_i} - z_i)^2$$

We optimize the loss function using gradient descent. We need gradients to perform the gradient descent optimization. The gradient of a function is a vector of its partial derivatives. We compute these using the back-propagation algorithm, which is an application of the chain rule from calculus.

Partial derivatives are calculated as follows:

$$\frac {\delta f}{\delta a_{11}} = f*(1-f)*b_1*u*(1-u)*x$$

$$\frac {\delta f}{\delta a_{12}} = f*(1-f)*b_1*u*(1-u)*y $$

$$\frac {\delta f}{\delta c_{1}} = f*(1-f)*b_1*u*(1-u) $$

$$\frac {\delta f}{\delta a_{21}} = f*(1-f)*b_2*v*(1-v)*x $$

$$\frac {\delta f}{\delta a_{22}} = f*(1-f)*b_2*v*(1-v)*y $$

$$\frac {\delta f}{\delta c_{2}} = f*(1-f)*b_2*v*(1-v) $$

$$\frac {\delta f}{\delta b_{1}} = f*(1-f)*u $$

$$\frac {\delta f}{\delta b_{2}} = f*(1-f)*v $$

$$\frac {\delta f}{\delta c_{3}} = f*(1-f) $$

By using these partial derivatives, we can perform gradient descent to iteratively adjust the parameters and minimize the loss function.

Consider a case where x=1, y=0 and the target output (z) is 1. The expected output corresponds to the XOR operation on x and y.

For our model, let's initialize parameters as follows:

$a_{11} = 0.6$, $a_{21} = 0.7$, $c_1 = 0.2$ 

$a_{21} = 0.8$, $a_{22} = 0.9$, $c_2 = 0.3$ 

$b_1 = 0.5$, $b_2 = 0.4$, $c_3 = 0.1$

Let's calculate $u$, $v$ and $f$:

$$u = \frac{1}{1 + e^{(0.6*1 + 0.7*0 + 0.2)}} = 0.668$$

$$v = \frac{1}{1 + e^{(0.8*1 + 0.9*0 + 0.3)}} = 0.710$$

$$f = \frac{1}{1 + e^{(0.5*0.668 + 0.4*0.710 + 0.1)}} = 0.671$$

The loss function, D:

$$D = \frac{1}{2} (0.671 - 1)^2 = 0.054$$

Now, let's calculate the partial derivatives for each parameter:

$$\frac {\delta f}{\delta a_{11}} = 0.671*(1-0.671)*0.5*0.668*(1-0.668)*1 = 0.030$$

And so on, you can calculate the rest of the partial derivatives in a similar manner.

Once you have all the partial derivatives, you can adjust the parameters using gradient descent with a learning rate (say, $\alpha = 0.1$):

$a_{11(new)} = a_{11(old)} - \alpha*\frac {\delta f}{\delta a_{11}} = 0.6 - 0.1*0.030 = 0.597$

And so on, for the rest of the parameters.

The parameters should then be updated and the process is repeated with the new parameters until the loss function is minimized to a satisfactory level.
