from typing import OrderedDict

import numpy as np
from collections import OrderedDict

class NeuralNetwork:
    """
    Base class for Neural Network generation
    """
    #TODO Add weight initialization method
    def __init__(self, neurons_in_input_layer, *, weight_initialization="Normal") -> None:

        self.weight_initialization = weight_initialization
        self.neurons_in_input_layer = neurons_in_input_layer
        self.__number_of_layers = 0
        self.__network_layout = OrderedDict()
        self.__network_layout["input_layer"] = {"number_of_neurons": self.neurons_in_input_layer}

    def add_layer(self, number_of_neurons, activation_function, bias: int) -> None:
        """
        Adds a new layer to the network
        """
        self.__network_layout[f"Layer {self.__number_of_layers}"] = {"number_of_neurons": number_of_neurons,
                                                                "bias": bias,
                                                                "activation_function": activation_function}

        self.__number_of_layers += 1

    def print_layout(self):

        for layer in self.__network_layout:
            print(f"{layer} | Number of neurons :{self.__network_layout[layer]['number_of_neurons']}")

        
    
    def __initialize_weights():
        """
        Initializes all weights based on option in constructor
        """
        pass

    def __forward_propagation():
        pass


    def __back_propagation():
        pass

        