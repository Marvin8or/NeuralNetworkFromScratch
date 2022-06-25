import numpy as np
from .functions import activations, cost_functions


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
        self._network_layout = list()
        self._weights: list = list()
        self._biases: list = list()
        self._weighted_inputs: list = list()
        self._layer_node_values: list = list()

        self._network_layout.append({"name": "input_layer", 
                                    "number_of_neurons": self.neurons_in_input_layer})
        self._number_of_layers = 1


    def _initialize_weights_and_biases(self, rows, columns) -> np.array:
        """
        Initializes all weights based on option in constructor
        """
        weights = np.random.uniform(size=(rows, columns))
        biases = np.random.uniform(size=(rows, 1))
        return weights, biases

    def add_layer(self, number_of_neurons, activation_function) -> None:
        """
        Adds a new layer to the network by calling it on the class instance
        """
        
        rows = number_of_neurons
        cols = self._network_layout[self._number_of_layers - 1]["number_of_neurons"]
   
        weights, bias = self._initialize_weights_and_biases(rows, cols)
        self._network_layout.append({"name": f"layer_{self._number_of_layers}", 
                                     "number_of_neurons": number_of_neurons,
                                     "bias": bias,
                                     "activation_function": activation_function})
        self._number_of_layers += 1
        self._weights.append(weights)
        self._biases.append(bias)
        


    def print_layout(self):

        for layer in self._network_layout:
            print(f"{layer['name']} | Number of neurons :{layer['number_of_neurons']}")
            

        
    #This class method goes to the NeuralNetworkTrainer class
    def feed_forward(self, input: np.array) -> np.array:

        self._layer_node_values.append(input)

        for indx in range(len(self._weights)):
            

            activation_function = activations[self._network_layout[indx+1]["activation_function"]]

            self._weighted_inputs.append(np.dot(self._weights[indx], input) + self._biases[indx])

            node_values_for_layer = activation_function(self._weighted_inputs[indx])
            
            self._layer_node_values.append(node_values_for_layer)
            input = node_values_for_layer

        
        return input

    
    def back_propagation(self, ground_truth) -> tuple:
        """
        Function to propagate the error backwards in form of error gradients
        """
        # List of derivatives of the cost function with respect to the weights at each layer
        dC_dw = [np.zeros(shape=weight.shape) for weight in self._weights]

        # List of derivatives of the cost function with respect to the bias at each layer
        dC_db = [np.zeros(shape=bias.shape) for bias in self._biases]
        
        delta = cost_functions["dMSE"](self._layer_node_values[-1], ground_truth) * \
            activations['d' + self._network_layout[-1]["activation_function"]](self._weighted_inputs[-1])
        
        dC_dw[-1] = np.dot(delta, self._layer_node_values[-2].T)
        dC_db[-1] = delta

        for layer in range(2, self._number_of_layers):

            z = self._weighted_inputs[-layer]
            activation_prime = activations['d' + self._network_layout[-layer]["activation_function"]](z)
            delta = np.dot(self._weights[-layer + 1].T, delta) * activation_prime

            dC_dw[-layer] = np.dot(delta, self._layer_node_values[-layer-1].T)
            dC_db[-layer] = delta
        
        return dC_dw, dC_db



            


class NeuralNetworkTrainer:

    def _forward_propagation():
        pass


    def _back_propagation():
        pass

        