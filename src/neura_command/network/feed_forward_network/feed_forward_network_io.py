import pickle


class FeedForwardNetworkIO:
    @staticmethod
    def save(network, file_path):
        """Saves a FeedForwardNetwork object to a file."""
        with open(file_path, "wb") as file:
            pickle.dump(network, file)

    @staticmethod
    def load(file_path):
        """Loads a FeedForwardNetwork object from a file."""
        with open(file_path, "rb") as file:
            return pickle.load(file)
