
from utilities import NeuralNetwork
if __name__ == "__main__":

    simple_network = NeuralNetwork(3)
    simple_network.add_layer(5, "ReLU", 4)
    simple_network.add_layer(4, "ReLU", 3)
    simple_network.add_layer(3, "ReLU", 2)
    simple_network.add_layer(2, "ReLU", 1)

    simple_network.print_layout()
    