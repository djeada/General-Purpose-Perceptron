## Gradient Descent

Gradient Descent is a first-order iterative optimization algorithm used to find the minimum of a function. It's commonly used in machine learning and data science to optimize loss functions, which measure the error or discrepancy between predicted and actual data.

![gradient_descent](https://github.com/djeada/NeuraCommand/assets/37275728/fba14b65-7e64-4a11-ba14-87e65a7a0530)

### What is Gradient Descent?

Imagine you're in a mountainous region and your goal is to reach the lowest point in the valley, but it's so foggy you can't see where to go. What you can do is feel the ground around you and take a step in the steepest direction downhill. That's essentially what gradient descent does, but in a mathematical function.

The 'ground' is the loss function, each 'step' represents an iteration of the algorithm, and the 'steepest direction' is determined by the negative of the gradient (or the derivative) of the function.

Here's an ASCII illustration of the process:

```
High Cost (Error)
|                 A
|                 /\
|                /  \
|               /    \ 
|              /      \ 
|         /\  /        \
|        /  \/          \
|       /    B           \ 
|      /                  \ 
|   C /                    \ C
|----------------------------- ---> Parameters (Weights)
```

* Point A represents our starting position, the initial parameters we randomly chose.
* Point B represents a local minimum in the loss function. This is a point where, if we were to take a step in any direction, the cost would increase. However, it's not the absolute lowest point in the entire function.
* Point C represents the global minimum, the absolute lowest point in the loss function and our ultimate target for the gradient descent algorithm.

The challenge with gradient descent is that it is a greedy algorithm, meaning it always takes the path of steepest descent without considering the bigger picture. From point A, the path of steepest descent leads to the local minimum at point B. Once the algorithm reaches point B, it considers it to be the minimum since any small movement increases the cost, and thus it stops iterating further.

Different versions of gradient descent and additional techniques have been developed to help alleviate this problem, such as stochastic gradient descent, mini-batch gradient descent, and methods that involve momentum, which can help "carry" the algorithm out of local minima and towards the global minimum. Also, often in practice, especially in high dimensional spaces, such local minima are not usually a significant issue, or the difference between the local and global minima is not significant in terms of model performance.

### How Does Gradient Descent Work?

1. **Initialization**: We start with random values for our parameters (weights and biases).

2. **Compute Gradient**: We calculate the gradient of the loss function at the current point. The gradient is a vector that points in the direction of the steepest increase of the function.

3. **Update Parameters**: We then update our parameters by taking a step in the direction of the steepest descent, which is the negative of the gradient. The size of the step is determined by the learning rate, a hyperparameter that we set beforehand.

4. **Iteration**: We repeat steps 2 and 3 until we reach a point where the gradient is close to zero, indicating we've reached a minimum (this could be a local or a global minimum).

The formula for the parameter update is: `new_parameter = old_parameter - learning_rate * gradient`.

### Why is Gradient Descent Important?

Gradient descent is crucial for training machine learning models because it's computationally efficient and relatively easy to implement. With the right tweaks and adjustments, gradient descent can handle complex, multi-dimensional loss functions, making it suitable for training complex models like neural networks.

## Gradient descent and perceptron

The original perceptron learning algorithm is actually a simpler form of gradient descent. In the perceptron learning rule, the weights are adjusted based on the difference between the desired output and the actual output. This can be viewed as a kind of gradient descent where the error function is a simple difference, and the gradient is either -1, 0, or 1, depending on the sign of the difference.

However, the perceptron learning rule has limitations. It only works for problems that are linearly separable, and it doesn't provide a way to calculate the amount of error for a particular set of weights. These limitations can be addressed by using gradient descent with a differentiable activation function (like the sigmoid function) and a more complex error function (like mean squared error or cross-entropy loss).

So, while the perceptron learning rule does use a form of gradient descent, it's often useful to switch to a more general form of gradient descent, especially for problems that are not linearly separable or when you need to measure the amount of error. In your notes, you actually already applied the gradient descent principles when you updated the weights and bias of the perceptron based on the error between the output and the target.

Here is the update rule we used: 

$$ Δw = η * (target - output) * input $$

$$ Δb = η * (target - output)$$

which resembles the gradient descent update rule: 

$$new\textunderscore parameter = old\textunderscore parameter - learning\textunderscore rate * gradient$$

In this case, the "gradient" is (target - output) for the bias and (target - output) * input for the weights. You can see how this follows the same principle of "descending" along the gradient to reach a minimum error.

## Mathematical form

$$
\theta = \theta - \alpha \nabla J(\theta)
$$

In this equation:

- $\theta$ represents the parameters of the function that we are trying to optimize. In the context of a machine learning model, these would be the model's weights and biases.
- $\alpha$ is the learning rate, a hyperparameter that determines the step size during each iteration while moving towards the minimum of our function. A smaller learning rate could get us closer to the minimum but may require more iterations to converge, while a larger learning rate may converge faster but risks overshooting the minimum.
- $\nabla J(\theta)$ is the gradient of the loss function $J(\theta)$ with respect to the parameters $\theta$. The gradient points in the direction of the steepest ascent in the function.

Now, let's look at a simple example where we apply the gradient descent algorithm. Suppose we have a simple linear regression problem with a cost function defined as the mean squared error (MSE):

$$
J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_{\theta}(x^{(i)}) - y^{(i)})^2
$$

Here, $h_{\theta}(x)$ is our hypothesis function (the model's prediction), defined for linear regression as $\theta_0 + \theta_1x$. Our goal is to find the parameters $\theta_0$ and $\theta_1$ that minimize the cost function $J(\theta)$.

For this example, let's consider a hypothetical dataset as follows:

| $x^{(i)}$ (Input) | $y^{(i)}$ (Output) |
|-------------------|--------------------|
|        1          |         2          |
|        2          |         3          |
|        3          |         4          |

We start with initial parameters $\theta_0 = 0$ and $\theta_1 = 0$, and a fixed learning rate of $\alpha = 0.01$. Now, let's apply the gradient descent algorithm:

| Step | $x^{(i)}$ | $y^{(i)}$ | $h_{\theta}(x^{(i)})$ | $J(\theta)$ | $\frac{\partial J}{\partial \theta_0}$ | $\frac{\partial J}{\partial \theta_1}$ | $\theta_0$ update | $\theta_1$ update |
|------|-----------|-----------|-----------------------|-------------|-----------------------------------|-----------------------------------|-----------------|-----------------|
|  1   |     1     |     2     |         0             |     2       |               -2                   |               -2                  |      0.02       |      0.02       |
|  2   |     2     |     3     |         0.04          |   1.8816    |             -2.92                 |             -5.84                |      0.0492     |      0.0984     |
|  3   |     3     |     4     |         0.1464        |   1.50145536|             -3.70752              |            -11.12256             |      0.086276   |      0.209112   |


Note: This table is a simplification of the process, and assumes the cost function is computed after each step. In practice, we usually compute the cost function after one complete pass (one epoch) over the data, and update the parameters accordingly. The derivatives of the cost function with respect to the parameters are computed using the chain rule of differentiation.
