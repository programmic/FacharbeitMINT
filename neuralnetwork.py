"""
network.py
~~~~~~~~~~

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
"""

#### Libraries
# Standard library
import random

import numpy as np
import matplotlib as plt

# My help libarys
import helpful_functions as hlp # usefull functions as dot products, or different activation functions
import colors # colors for terminal output

class Network(object):

    def __init__(self, sizes, activation='sigmoid'):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [[random.gauss(0, 1) for _ in range(y)] for y in sizes[1:]]
        self.weights = [[[random.gauss(0, 1) for _ in range(x)] for _ in range(y)]
                        for x, y in zip(sizes[:-1], sizes[1:])]
        
    # Select activation functions
        self.activation_str = activation
        if activation == 'sigmoid':
            self.activation = hlp.sigmoid
            self.activation_prime = hlp.sigmoid_prime
        elif activation == 'relu':
            self.activation = hlp.relu
            self.activation_prime = hlp.relu_prime
        elif activation == 'leaky_relu':
            self.activation = hlp.leaky_relu
            self.activation_prime = hlp.leaky_relu_prime
        elif activation == 'tanh':
            self.activation = hlp.tanh
            self.activation_prime = hlp.tanh_prime
        else:
            raise ValueError("Unsupported activation function!")




    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for i, (b, w) in enumerate(zip(self.biases, self.weights)):
            a = [self.activation(hlp.dot(w_row, a) + bias) for w_row, bias in zip(w, b)]


            progress = (i + 1) / len(self.biases) * 100

        return a

    
    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None, return_metrics=False):
        """Train the neural network using mini-batch stochastic gradient descent."""
        if test_data: n_test = len(test_data)
        n = len(training_data)

        accuracies = []
        losses = []
        
        print(f"epochs:{epochs}\nmini_batch_size:{mini_batch_size}\neta:{eta}")
        print(f"{hlp.colors.green}Beginning training at {hlp.timeFormat(hlp.time.time())}{hlp.colors.clear}")

        cnt = 1
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k + mini_batch_size] for k in range(0, n, mini_batch_size)]
            
            start_time = hlp.time.time()  # Record the start time
            print(f"Epoch 1/{epochs}: 000%   [{'-'*50}]", end="")
            for number, mini_batch in enumerate(mini_batches):
                self.update_mini_batch(mini_batch, eta)
                print(f"\rEpoch {cnt}/{epochs}: {hlp.lenformat(round(number/len(mini_batches)*100), 3, ' ', 'front')}%    {hlp.progress(number/len(mini_batches), 50)}", end="")
            end_time = hlp.time.time()  # Record the end time
            print()
            duration = end_time - start_time  # Calculate the duration
            print(f"Proceesing mini batch epoch {cnt} complete after {duration:.4f}")
            

            if test_data:
                accuracy = self.evaluate(test_data)
                accuracies.append(accuracy / n_test * 100)  # Store accuracy
                print(f"\nEpoch {j}: {accuracy} / {n_test} - Accuracy: {accuracies[-1]:.2f}%")
            else:
                print(f"Epoch {j} complete")

            # Calculate and store the loss for the current epoch
            total_loss = 0  # Initialize total_loss
            print(f"Calculating loss: 000%   {hlp.progress(0, 50)}",end="")
            for i, (x, y) in enumerate(training_data):
                output = self.feedforward(x)  # Get the network output for input x
                loss = np.linalg.norm(np.array(output) - np.array(y)) ** 2  # Calculate loss for this instance
                total_loss += loss  # Accumulate loss
                progress = i / len(training_data)
                print(f"\rCalculating loss: {hlp.lenformat(round(progress)*100, 3, ' ', 'front')}%   {hlp.progress(progress, 50)}", end="")
         

            # Average the total loss by the number of samples
            total_loss /= n
            losses.append(total_loss)  # Store loss
            print(f"\r>  Epoch {j} Loss: {round(total_loss,6)}{' '*55}\n")
            cnt+=1

        if return_metrics:
            return accuracies, losses  # Return the accuracy and loss metrics

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [[0] * len(b) for b in self.biases]
        nabla_w = [[[0] * len(w_row) for w_row in w_layer] for w_layer in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [hlp.vector_add(nb, dnb) for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [[hlp.vector_add(nw_row, dnw_row) for nw_row, dnw_row in zip(nw_layer, dnw_layer)]
                       for nw_layer, dnw_layer in zip(nabla_w, delta_nabla_w)]
            
            
        self.weights = [[hlp.vector_subtract(w_row, hlp.scalar_vector_mult(eta / len(mini_batch), nw_row)) for w_row, nw_row in zip(w_layer, nw_layer)]
                for w_layer, nw_layer in zip(self.weights, nabla_w)]

        self.biases = [hlp.vector_subtract(b, hlp.scalar_vector_mult(eta / len(mini_batch), nb)) for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [[0] * len(b) for b in self.biases]
        nabla_w = [[[0] * len(w_row) for w_row in w_layer] for w_layer in self.weights]
        # feedforward
        activation = x
        activations = [x]  # list to store all the activations, layer by layer
        zs = []  # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = [hlp.dot(w_row, activation) + bias for w_row, bias in zip(w, b)]
            zs.append(z)
            activation = [self.activation(zi) for zi in z]
            activations.append(activation)
        # backward pass
        delta = [self.cost_derivative(a, y) * self.activation_prime(z) for a, z in zip(activations[-1], zs[-1])]
        nabla_b[-1] = delta
        nabla_w[-1] = [hlp.scalar_vector_mult(d, activations[-2]) for d in delta]
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = [self.activation_prime(zi) for zi in z]
            delta = [hlp.dot(w_col, delta) * sp_i for w_col, sp_i in zip(hlp.transpose(self.weights[-l + 1]), sp)]
            nabla_b[-l] = delta
            nabla_w[-l] = [hlp.scalar_vector_mult(d, activations[-l - 1]) for d in delta]
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = []
        print("Evaluating:   0%", end="")
        for i, (x, y) in enumerate(test_data):
            output = self.feedforward(x)
            predicted_label = np.argmax(output)  # Get the index of the highest score
            true_label = np.argmax(y) if isinstance(y, list) else y  # Handle one-hot encoding
            test_results.append((predicted_label, true_label))
            progress = i/len(test_data)
            print(f"\rEvaluating: {hlp.lenformat(round(progress*100), 6, ' ', 'front')}%   {hlp.progress(progress, 50)}", end="")
        return sum(int(predicted == true) for (predicted, true) in test_results)


    def cost_derivative(self, output_activations, y):
    # Assuming y is a list with a single element, convert it to a scalar value
        if type(output_activations) is list:
            return (output_activations - y[0])
        else:
            try:
                return (output_activations - y)
            except ValueError:
                raise ValueError("Invalid")


def test_activations(training_data, test_data, sizes, epochs, mini_batch_size, eta):
    activation_functions = ['sigmoid', 'relu', 'tanh', 'leaky_relu']
    results = {activation: {'accuracy': [], 'loss': []} for activation in activation_functions}  # Store results
    
    for activation in activation_functions:
        print(f"\nTesting with activation function: {activation}")
        net = Network(sizes, activation=activation)
        accuracies, losses = net.SGD(training_data, epochs, mini_batch_size, eta, test_data=test_data, return_metrics=True)
        results[activation]['accuracy'] = accuracies
        results[activation]['loss'] = losses

    try:
        # Plotting the results
        for activation in activation_functions:
            plt.plot(range(epochs), results[activation]['accuracy'], label=f'Accuracy ({activation})')
            plt.plot(range(epochs), results[activation]['loss'], label=f'Loss ({activation})')

        plt.xlabel('Epochs')
        plt.ylabel('Metrics')
        plt.title('Activation Function Comparison')
        plt.legend()
        plt.show()
    except:
        pass


@hlp.time_it
def loadData():
    trd = hlp.load_csv_data('input/digit-recognizer/train.csv')
    ted = hlp.load_csv_data('input/digit-recognizer/test.csv')
    return trd, ted


if __name__ == "__main__":
    hlp.clearTerminal()
    # Load CSV Data
    print("loading data...")
    training_data, test_data = loadData()

    print(f"{colors.green}Started training at {hlp.timeFormat(hlp.time.time())}{colors.clear}")
    startTime = hlp.time.time()

    test_activations(training_data, test_data, [784,20,10], 16, 16, 0.03)
    endTime = hlp.time.time()
    print(f"{colors.green}Completed training in {hlp.timeFormat(endTime - startTime)}{colors.clear}")
else: print("Successfully loaded 'neuralnetwork.py'")