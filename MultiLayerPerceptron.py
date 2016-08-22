"""Created and edited by James Alexander Adriano Aymer"""

#importing the numpy library
from numpy import dot

#developing the ANN calss with its variables
class ANN_Network:
 
    size = \
    {
        "input":0,
        "hidden":0,
        "output":0
    }
    
    hidden_layer = None
    output_layer = None
    
    network = None
    
    #initiates the objects to be used with the 
    def __init__(this, INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE):
        import random
        
        #saving the size input, hidden and output layer
        this.size["input"], this.size["hidden"], this.size["output"] = (INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
        
        #settin the weights - hidden
        this.hidden_layer = [[random.random() for __ in range(this.size["input"]  + 1)] 
                                                 for _  in range(this.size["hidden"] )]
        #setting the weights - output
        this.output_layer = [[random.random() for __ in range(this.size["hidden"] + 1)] 
                                                 for _  in range(this.size["output"] )]

        this.network = [this.hidden_layer, this.output_layer]
        
        #printing the size of the input, hidden and output layers
        print("\tLayer\tNeurons\n\tinput\t%i\n\thidden\t%i\n\toutput\t%i"\
              %(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE))
        
    #the sigmoid activation function 
    def sigmoid_network_activation(this, s):
        from math import exp as e
        s *= 0.1
        return 1/(1 + e(-s))
    
    #initiating the function for eash neuron's output
    #this function returns the sigmoid activation of a neuron's weights multiplied by its weights
    def neuron_output(this, n_weights, n_inputs):
        return this.sigmoid_network_activation(dot(n_weights, n_inputs))
    
    
    def feed_forward_network(this, artificial_neural_network, network_input_vector):
        """takes in an artificial neural network (represented as a list of lists of lists of weights) 
           endeavours to return the output to the network, by means of forward propagating the network_input_vector"""
        
        #creating the initial
        neuron_outputs = []

        #processing one layer at a time in the ANN
        for layer in artificial_neural_network:
            input_with_a_bias = network_input_vector + [1]

            #iteration to calculate the output for each neuron in a layer
            output = [this.neuron_output(neuron, input_with_a_bias) for neuron in layer]

            #saving the outputs to a list called 'outputs'
            neuron_outputs.append(output)

            #now the input and output have switched round
            network_input_vector = output

        return neuron_outputs
    
    """Backpropagation algorithm for the ANN"""
    #the backpropagation function with its arguements
    def network_backprop(this, network_input_vector, network_targets, network_learning_rate):
    
        #setting the outputs to the layers in the network
        hidden_layer_outputs, outputs = this.feed_forward_network(this.network, network_input_vector)

        #finds the change that has occured within the output
        network_output_deltas = [network_learning_rate * output * (1 - output) * (output - network_target)
                         for output, network_target in zip(outputs, network_targets)]

        #making weight adjustments for the output layer, per neuron
        for i, network_output_neuron in enumerate(this.network[-1]):
          
            for j, hidden_layer_output in enumerate(hidden_layer_outputs + [1]):
                #changing the jth weight in relation to this neuron delta & jth input
                network_output_neuron[j] -= network_output_deltas[i] * hidden_layer_output

        #propagating the errors bakwards to the hidden layer
        hidden_layer_deltas = [network_learning_rate * hidden_layer_output * \
                               (1 - hidden_layer_output) * dot(network_output_deltas, [n[i] for n in this.output_layer])
                         for i, hidden_layer_output in enumerate( hidden_layer_outputs )]

        #making weight adjustments for the hidden layer, per neuron
        for i, hidden_layer_neuron in enumerate(this.network[0]):
            for j, input in enumerate(network_input_vector + [1]):
                hidden_layer_neuron[j] -= hidden_layer_deltas[i] * input
      
    #making the ANN prediction
    def ann_prediction(this, input):
        return this.feed_forward_network(this.network, input)[-1]