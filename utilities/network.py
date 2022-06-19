import numpy as np
from .activations import functions

#TODO factory pattern for different kind of layers
class NeuralNetwork:
    """
    Base class for Neural Network generation
    """
    #TODO Add weight initialization method
    def __init__(self, neurons_in_input_layer, *, weight_initialization="Normal") -> None:

        self.weight_initialization = weight_initialization
        self.neurons_in_input_layer = neurons_in_input_layer

        #TODO too much lists, figure out a way to sort this out
        self.__network_layout = list()
        self.__weights: list = list()
        self.__biases: list = list()
        self.__layer_node_values: list = list()

        self.__network_layout.append({"name": "input_layer", 
                                    "number_of_neurons": self.neurons_in_input_layer})
        self.__number_of_layers = 1

    def add_layer(self, number_of_neurons, activation_function, bias: np.array) -> None:
        """
        Adds a new layer to the network
        """
        self.__network_layout.append({"name": f"layer_{self.__number_of_layers}", 
                                     "number_of_neurons": number_of_neurons,
                                     "bias": bias,
                                     "activation_function": activation_function})
        self.__number_of_layers += 1
        weights_rows = self.__network_layout[self.__number_of_layers -1]["number_of_neurons"]
        wights_cols = self.__network_layout[self.__number_of_layers - 2]["number_of_neurons"]
        weights = np.random.normal(size=(weights_rows, wights_cols))
        self.__weights.append(weights)
        self.__biases.append(bias)


    def print_layout(self):

        for layer in self.__network_layout:
            print(f"{layer['name']} | Number of neurons :{layer['number_of_neurons']}")

        
    
    def feed_forward(self, input: np.array) -> None:
        
        for indx in range(len(self.__weights)):
            
            activation_function = functions[self.__network_layout[indx]["activation_function"]]
            node_values_for_layer = activation_function(np.dot(self.__weights[indx], input) + self.__biases[indx])
            self.__layer_node_values.append(node_values_for_layer)
            


    def __initialize_weights():
        """
        Initializes all weights based on option in constructor
        """
        pass

    def __forward_propagation():
        pass


    def __back_propagation():
        pass

        