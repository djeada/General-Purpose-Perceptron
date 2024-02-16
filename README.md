# NeuraCommand

NeuraCommand is a very simple CLI tool designed for creating, training, and deploying neural networks. It provides a seamless interface for handling complex neural network architectures, including perceptrons with multiple layers. With NeuraCommand, users can efficiently load data, train models, and use them for predictions.

## Features

- **Flexible Network Creation Library**: A library built on top of numpy, providing the flexibility to create various network configurations easily.

- **Intuitive Command-Line Interface (CLI)**: A user-friendly CLI that simplifies neural network operations, enhancing accessibility for usage in different programming languages or embedded systems.

- **Customizable Network Architectures**: Tailor your neural networks with a range of layers and structures. Create unique networks either through an interactive CLI experience or by populating a JSON template.

- **Matrix Input Support**: Offers compatibility with multiple matrix data formats, facilitating versatile network training. Once your network is set up, train it repeatedly using a variety of inputs.

- **Persistent Model Management**: Efficiently save and reload your neural network models for continuous training or future utilization. Once trained, models can be stored as blobs on disk and reloaded as needed.

- **Immediate Predictions**: Implement your trained networks for real-time prediction tasks. Input data can be manually entered or read from a disk, with the option to display predictions on-screen or save them to a file.

- **Layer Output Visualization**: Gain insights into your network's processing by displaying outputs from all layers. Choose between numerical printouts, CSV files, or other formats to keep track of your network's functionality—nothing remains hidden.

## Installation

Install NeuraCommand using pip:

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

Example code for using NeuraCommand in Python:

```python
from gp_perceptron_framework import GeneralPurposePerceptron

# Initialize a network
network = GeneralPurposePerceptron()

# Add layers
network.add_layer(neuron_count=5)  # Hidden layer
network.add_layer(neuron_count=3)  # Output layer

# Train the network
data = [([0.1, 0.2, 0.3], [0, 1, 0])]  # (input, output) pairs
network.train(data, epochs=10)

# Predictions
inputs = [0.1, 0.2, 0.3]
outputs = network.predict(inputs)
print(outputs)  # Example output: [0.2, 0.8, 0.1]

# Evaluate performance
performance = network.evaluate(data)
print(performance)  # Example loss: 0.05
```

## References

1. Nielsen, M. (2015). *Neural Networks and Deep Learning*. Determination Press.
   - [Online Book](http://neuralnetworksanddeeplearning.com/)

2. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.
   - [MIT Press](https://www.deeplearningbook.org/)

3. Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer.
   - [Springer](http://www.springer.com/gp/book/9780387310732)

4. Shalev-Shwartz, S., & Ben-David, S. (2014). *Understanding Machine Learning: From Theory to Algorithms*. Cambridge University Press.
   - [Cambridge Core](https://www.cambridge.org/core/books/understanding-machine-learning/3059709FEBB4176FC37311D0AD8B2B8E)

5. Hertz, J., Krogh, A., & Palmer, R. G. (1991). *Introduction to the Theory of Neural Computation*. Addison-Wesley.
   - [Amazon](https://www.amazon.com/Introduction-Theory-Neural-Computation-Computational/dp/0201515601)

6. Fausett, L. (1994). *Fundamentals of Neural Networks: Architectures, Algorithms, and Applications*. Prentice-Hall.
   - [Pearson](https://www.pearson.com/us/higher-education/program/Fausett-Fundamentals-of-Neural-Networks-Architectures-Algorithms-and-Applications/PGM80736.html)

7. Bishop, C. M. (1995). *Neural Networks for Pattern Recognition*. Oxford University Press.
   - [Amazon](https://www.amazon.com/Neural-Networks-Pattern-Recognition-Christopher/dp/0198538642)

8. Géron, A. (2019). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*. O'Reilly Media.
   - [O'Reilly](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/)

9. Vasilev, I., & Slater, D. (2019). *Python Deep Learning*. Packt Publishing.
   - [Packt](https://www.packtpub.com/product/python-deep-learning-second-edition/9781789348460)

10. Iyengar, R. S. (2017). *Convolutional Neural Networks in Visual Computing: A Concise Guide*. CRC Press.
    - [Amazon](https://www.amazon.com/Convolutional-Networks-Visual-Computing-Concise/dp/1498749936)

## TODO

- **Implement a Tutorial for Non-Linear Datasets**: Develop a comprehensive tutorial focused on classifying non-linearly separable datasets using NeuraCommand. This tutorial should cover the end-to-end process, from data loading and preprocessing to model training and evaluation.

`from sklearn.datasets import make_moons`

- **Enhance CLI for Network Analytics**: Expand the command-line interface to include more detailed network analytics and diagnostics. This expansion should encompass advanced metrics, visualization tools for training progress, and detailed error analysis. The goal is to provide users with a more nuanced understanding of their model's performance and behavior.

- **Develop Advanced Architecture Documentation**: Create additional documentation and practical examples for advanced neural network architectures, such as convolutional neural networks (CNNs), recurrent neural networks (RNNs), and Transformer models. This documentation should guide users through the intricacies of these architectures, their use cases, and how to implement them effectively in NeuraCommand.

- **Integrate Advanced Optimization Techniques**: Plan the integration of advanced optimization techniques and learning rate schedules into NeuraCommand. This includes implementing adaptive learning rate algorithms, regularization methods, and dropout techniques to improve model robustness and performance.

## Contributing

We warmly welcome contributions to NeuraCommand! Whether you're fixing bugs, adding new features, improving documentation, or spreading the word, your help is invaluable.

### How to Contribute

1. **Fork the Repository**: Start by forking the NeuraCommand repository to your GitHub account.
2. **Create a Branch**: For each new feature or fix, create a new branch in your forked repository.
3. **Develop and Test**: Implement your changes and ensure they are thoroughly tested.
4. **Submit a Pull Request**: Once you're satisfied with your changes, submit a pull request to the main repository. Include a clear description of your changes and any relevant issue numbers.
5. **Code Review**: Your pull request will be reviewed by the maintainers. They may suggest changes or improvements.
6. **Merge**: Once your pull request is approved, it will be merged into the main codebase.

### Contribution Guidelines

- Ensure your code adheres to the project's coding standards and style.
- Update the documentation to reflect your changes, if necessary.
- Keep your commits clean and understandable.
- Stay respectful and open to feedback during the review process.

We're excited to see what you bring to NeuraCommand!

## License

NeuraCommand is licensed under the MIT License. This allows for a wide range of uses, providing flexibility and freedom for both personal and commercial applications while requiring only the retention of copyright and license notices. For more detailed information, please refer to the [LICENSE](LICENSE) file included in the repository.
