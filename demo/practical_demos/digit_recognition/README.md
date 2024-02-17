## 3-Layer MLP for Digit Recognition

A 3-layer Multilayer Perceptron (MLP) is a type of neural network used for tasks like digit recognition. It consists of an input layer, two hidden layers, and an output layer. Each layer is made up of perceptrons (or neurons) that perform calculations and pass their results to the next layer.

![digits](https://github.com/djeada/NeuraCommand/assets/37275728/02f3854d-7041-4018-a3d8-e98fdd98438b)

### Structure and Functions

1. **Input Layer:**
   - The input layer receives the digit image as a flattened vector. If the image is 28x28 pixels, this results in a 784-dimensional input vector (`input_size = 784`).

2. **Hidden Layers:**
   - Each hidden layer neuron computes a weighted sum of its inputs and applies an activation function.
   - The mathematical representation for each neuron `j` in the first hidden layer is:

$$a_j^{(1)} = \text{ReLU}\left(\sum_{i} w_{ij}^{(1)} x_i + b_j^{(1)}\right)$$

   - Where `x_i` are the input features, `w_{ij}^{(1)}` are the weights, `b_j^{(1)}` is the bias, and `a_j^{(1)}` is the activated output for neuron `j`.

3. **Output Layer:**
   - The output layer has 10 neurons (for digits 0-9), each giving the probability of the input being a specific digit.
   - The computation for output neuron `k` is similar to the hidden layer, but often a softmax function is used for classification:

$$o_k = \text{softmax}\left(\sum_{j} w_{jk}^{(2)} a_j^{(1)} + b_k^{(2)}\right)$$

   - The softmax function converts the outputs to probabilities that sum to 1.


### Image Representation as Input

Digit recognition involves processing images, typically hand-written digits, which are represented in a format suitable for the neural network. 

1. **Image as a 2D Array:**
   - Initially, an image is a 2D array where each element corresponds to a pixel.
   - In grayscale images, like those in the MNIST dataset, each pixel value ranges from 0 (black) to 255 (white). 

2. **Flattening the Image:**
   - Neural networks typically require input to be a 1D vector. Hence, the 2D image array is "flattened".
   - For a 28x28 image, this results in a 784-dimensional vector (28 * 28 = 784).
   - Each element of this vector represents the intensity of a pixel in the image.

3. **Normalization:**
   - Pixel values are usually normalized to aid in faster and more stable convergence during training.
   - Normalization often involves scaling pixel values to a range of 0 to 1, done by dividing each pixel value by 255.

### The Output Layer and Interpretation

1. **Output Layer Structure:**
   - The output layer of the MLP for digit recognition typically has 10 neurons, corresponding to the 10 possible digits (0 through 9).

2. **Softmax Activation:**
   - A softmax activation function is often used in the output layer for classification tasks.
   - It converts the output of the network into a probability distribution over the 10 digits.
   - The softmax function is defined as:

$$\text{softmax}(z)_i = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}$$

   - Where $ z $ is the input vector to the softmax function, and $ K $ is the number of classes (digits in this case).

3. **Interpreting the Output:**
   - Each output neuron's value represents the probability that the input image corresponds to a particular digit.
   - The predicted digit is typically taken to be the one corresponding to the neuron with the highest probability.

### Activation Function

- **ReLU (Rectified Linear Unit):**
  - Defined as $\text{ReLU}(x) = \max(0, x)$.
  - Introduces non-linearity, allowing the network to capture complex patterns.

### Learning: Feedforward and Backpropagation

1. **Feedforward:**
   - The process of passing input data through the layers to get an output.
   - Computation follows the sequence of weighted sums and ReLU activations described above.

2. **Backpropagation:**
   - Aimed at updating the weights to reduce the error between the predicted and actual outputs.
   - Involves calculating the gradient of the loss function with respect to each weight using the chain rule.
   - The weight update is given by:

$$w_{ij}^{(l)} = w_{ij}^{(l)} - \eta \frac{\partial \mathcal{L}}{\partial w_{ij}^{(l)}}$$

   - Where $\eta$ is the learning rate, and $\mathcal{L}$ is the loss function (e.g., cross-entropy for classification).
