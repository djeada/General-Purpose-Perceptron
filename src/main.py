import argparse
import csv

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Import your classes here
# from your_package import GeneralPurposePerceptron


def main(args):
    # Read the CSV file
    with open(args.csv_file, "r") as f:
        reader = csv.reader(f)
        data = np.array(list(reader)).astype(float)

    # Split data into inputs and outputs based on provided indices
    X = data[:, args.input_columns]
    y = data[:, args.output_columns]

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Standardize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Create your network
    gpp = GeneralPurposePerceptron(input_dim=X_train.shape[1])
    gpp.add_layer(neuron_count=args.hidden_neurons)
    gpp.add_layer(neuron_count=y_train.shape[1])

    # Train the network
    training_data = list(zip(X_train.tolist(), y_train.tolist()))
    gpp.train(training_data, epochs=args.epochs)

    # Evaluate the network
    evaluation_data = list(zip(X_test.tolist(), y_test.tolist()))
    loss = gpp.evaluate(evaluation_data)
    print(f"Evaluation loss: {loss}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A CLI for GeneralPurposePerceptron")
    parser.add_argument(
        "--csv_file",
        type=str,
        required=True,
        help="The CSV file to use for training and evaluation.",
    )
    parser.add_argument(
        "--input_columns",
        type=int,
        nargs="+",
        required=True,
        help="The indices of the input columns in the CSV file.",
    )
    parser.add_argument(
        "--output_columns",
        type=int,
        nargs="+",
        required=True,
        help="The indices of the output columns in the CSV file.",
    )
    parser.add_argument(
        "--hidden_neurons",
        type=int,
        default=10,
        help="The number of neurons in the hidden layer.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="The number of epochs to train the network.",
    )
    args = parser.parse_args()

    main(args)
