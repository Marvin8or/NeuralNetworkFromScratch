from utilities import *
from operator import xor
import numpy as np

#TODO create separate module for generating datasets
def generate_dummy_dataset(number_of_samples, operation="AND"):
    """
    Generates a dummy dataset to test various stages of the network developement
    number_of_samples represents the number of rows in the dataset
    operation represents the output of each row

    Example

    >>>AND_dataset = generate_dummy_dataset(5, operation="AND")
    >>>AND_dataset
    [[1, 0, False]
     [0, 0, True]
     [1, 1, True]
     [0, 1, False]
     [1, 1, True]]
    
    """

    #TODO create decorator
    def and_function(x, y):
        return x and y
    
    def or_function(x, y):
        return x or y

    operations = {"AND": and_function, 
                  "OR": or_function, 
                  "XOR": xor}

    dummy_dataset = np.zeros(shape=(number_of_samples, 3))
    for row in range(number_of_samples):
        np.random.seed(row)
        x, y = np.random.randint(0, 1 + 1), np.random.randint(0, 1 + 1)
        result = operations[operation](x, y)

        dummy_dataset[row][0], dummy_dataset[row][1], dummy_dataset[row][2] = x, y, result 
    
    return dummy_dataset
        
if __name__ == "__main__":

    # Create dummy dataset
    AND_operation_recognition_dataset = generate_dummy_dataset(10, "XOR")
    print(AND_operation_recognition_dataset)
    
    # Build neural network
    simple_network = NeuralNetwork(3)
    simple_network.add_layer(5, "ReLU", np.array([[1, 2, 3, 4, 5]]).T)
    simple_network.add_layer(4, "ReLU", 3)
    simple_network.add_layer(3, "ReLU", 2)
    simple_network.add_layer(2, "ReLU", 1)

    # Print network layout
    # simple_network.print_layout()

    # Feed forward
    # XXX For some reason gives ERROR!!
    simple_network.feed_forward(AND_operation_recognition_dataset[0])

    # Print the output after first feed forward
    # print(simple_network.output())
    
    