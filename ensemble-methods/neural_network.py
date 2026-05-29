import numpy as np
from typing import List
from sklearn.utils import shuffle


class NeuralNetwork:
    
    def __init__(self, layer_sizes, random_seed=None):
        # store architecture and counts
        self.layer_sizes = list(layer_sizes)
        self.num_layers = len(self.layer_sizes)

        # initialize random generator for reproducibility
        if random_seed is None:
            rng = np.random.RandomState()
        else:
            rng = np.random.RandomState(random_seed)

        # Initialize weight matrices, theta, for each layer transition.
        # theta[i] has shape (next_layer_size, current_layer_size + 1)
        # we will add a bias term hence the extra column for the bias weight
        self.thetas = []
        for i in range(self.num_layers - 1):
            # current layer size (without bias) and next layer size
            rows = self.layer_sizes[i + 1]
            columns = self.layer_sizes[i] + 1
            
            # initialize weights randomly in range [-1, 1]
            W = rng.uniform(-1, 1, size=(rows, columns))
            self.thetas.append(W)
    
    def activation_func(self, z):
        # use numerically stable sigmoid function to avoid overflow
        z = np.asarray(z)
        output = np.empty_like(z, dtype=float)

        positive_mask = z >= 0
        negative_mask = z < 0

        # for positive z, use standard sigmoid formula
        # for negative z, use the alternative formulation to avoid overflow
        output[positive_mask] = 1 / (1 + np.exp(-z[positive_mask]))
        exp_z = np.exp(z[negative_mask])
        output[negative_mask] = exp_z / (1 + exp_z)

        return output
    
    def forward_propagation(self, X):
        num_instances = X.shape[0]
        # will store activations and z values for each layer
        # each element is a list of arrays (one per layer)
        a_values = [X]
        z_values = []

        for layer in range(self.num_layers - 1):
            a_current = a_values[layer]
            a_with_bias = np.column_stack([np.ones(num_instances), a_current])
            # a_with_bias has shape (num_instances, current_layer_size + bias column)
            # self.thetas[layer].T has shape (current_layer_size + bias column, next_layer_size)
            z = a_with_bias @ self.thetas[layer].T
            a = self.activation_func(z)

            z_values.append(z)
            a_values.append(a)

        return z_values, a_values
    
    def compute_cost(self, X, y, lambda_reg=0):
        num_instances = X.shape[0]

        # make sure one-output labels are shaped like [[y1], [y2], ...]
        y = np.asarray(y)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        
        # forward propagate to get output activations
        _, a_values = self.forward_propagation(X)
        a_output = a_values[-1]
        
        # use a small offset, epsilon, to avoid log(0)
        eps = 1e-15
        a_output = np.clip(a_output, eps, 1 - eps)
        J = -y * np.log(a_output) - (1 - y) * np.log(1 - a_output)
        J = np.sum(J) / num_instances
        
        # compute regularization term (exclude bias weights)
        regularization = 0
        if lambda_reg > 0:
            S = 0
            for theta in self.thetas:
                S += np.sum(theta[:, 1:] ** 2)
            regularization = (lambda_reg / (2 * num_instances)) * S
        
        return J + regularization 
    
    def backpropagation(self, X, y, lambda_reg=0):
        num_instances = X.shape[0]

        # make sure one-output labels are shaped like [[y1], [y2], ...]
        y = np.asarray(y)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        # forward propagate to get z and a values for all layers
        _, a_values = self.forward_propagation(X)

        gradients = [None] * len(self.thetas)
        deltas = [None] * self.num_layers
        # compute delta for output layer
        deltas[self.num_layers - 1] = a_values[-1] - y

        # compute deltas for hidden layers down to the first hidden layer
        for layer in range(self.num_layers - 2, 0, -1):
            # don't compute delta for bias terms
            theta = self.thetas[layer][:, 1:]
            a = a_values[layer]
            deltas[layer] = deltas[layer + 1] @ theta * (a * (1 - a))

        # compute gradients for each layer
        for layer in range(self.num_layers - 2, -1, -1):
            a = a_values[layer]
            a_with_bias = np.column_stack([np.ones(num_instances), a])
            gradients[layer] = (deltas[layer + 1].T @ a_with_bias) / num_instances

        # add regularization terms to gradients (exclude bias weights)
        if lambda_reg > 0:
            for layer in range(self.num_layers - 2, -1, -1):
                gradients[layer][:, 1:] += (lambda_reg * self.thetas[layer][:, 1:]) / num_instances

        return gradients, deltas

    def update_weights(self, gradients: List[np.ndarray], alpha: float):
        for layer in range(len(self.thetas)):
            self.thetas[layer] -= alpha * gradients[layer]
    
    def predict(self, X):
        _, a_values = self.forward_propagation(X)
        probabilities = a_values[-1]

        if probabilities.shape[1] == 1:
            return (probabilities >= 0.5).astype(int).ravel()
        return np.argmax(probabilities, axis=1)
    
    def train(self, X, y, learning_rate=0.1, lambda_reg=0, max_iterations=1000, batch_size=None, verbose=False):
        
        num_instances = X.shape[0]

        # make sure one-output labels are shaped like [[y1], [y2], ...]
        y = np.asarray(y)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        
        if batch_size is None:
            batch_size = num_instances
        
        for iteration in range(max_iterations):
            X_shuffled, y_shuffled = shuffle(X, y)
            
            # update weights in batches
            for batch_start in range(0, num_instances, batch_size):
                batch_end = min(batch_start + batch_size, num_instances)
                X_batch = X_shuffled[batch_start:batch_end]
                y_batch = y_shuffled[batch_start:batch_end]
                
                # do back propagation to compute gradients for this batch
                gradients, _ = self.backpropagation(X_batch, y_batch, lambda_reg)
                 
                # update weights using the computed gradients
                self.update_weights(gradients, learning_rate)
            
            if verbose and iteration % 10 == 0:
                # compute cost after processing all batches
                current_cost = self.compute_cost(X, y, lambda_reg)
                print(f"Iteration {iteration}: Cost = {current_cost}")

    def debug(self, X, y, lambda_reg=0):
        num_instances = X.shape[0]

        y = np.asarray(y)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        # used for formatting to match the example files
        def format_vector(values):
            return "   ".join(f"{float(value):0.5f}" for value in values)

        print("--------------------------------------------")
        print("Computing the error/cost, J, of the network")

        total_cost = 0.0
        eps = 1e-15

        for i in range(num_instances):
            x_i = X[i:i + 1]
            y_i = y[i:i + 1]
            z_values, a_values = self.forward_propagation(x_i)

            print(f"\tProcessing training instance {i + 1}")
            print(f"\tForward propagating the input [{format_vector(x_i.ravel())}]")

            # add bias to the input layer before printing
            a_1 = np.concatenate(([1.0], a_values[0].ravel()))
            print(f"\t\ta1: [{format_vector(a_1)}]")
            print()

            # print z and a for every layer after the input layer
            for layer in range(1, self.num_layers):
                z = z_values[layer - 1].ravel()
                print(f"\t\tz{layer + 1}: [{format_vector(z)}]")

                a = a_values[layer].ravel()

                # hidden layers are printed with their bias term, output layer is not
                if layer < self.num_layers - 1:
                    a = np.concatenate(([1.0], a))
                print(f"\t\ta{layer + 1}: [{format_vector(a)}]")
                print()

            # sum across output neurons first, then average across instances later
            a_output = np.clip(a_values[-1], eps, 1 - eps)
            instance_cost = -np.sum(
                y_i * np.log(a_output) + (1 - y_i) * np.log(1 - a_output)
            )
            total_cost += instance_cost

            print(f"\t\tf(x): [{format_vector(a_values[-1].ravel())}]")
            print(f"\tPredicted output for instance {i + 1}: [{format_vector(a_values[-1].ravel())}]")
            print(f"\tExpected output for instance {i + 1}: [{format_vector(y_i.ravel())}]")
            print(f"\tCost, J, associated with instance {i + 1}: {instance_cost:0.3f}")
            print()

        # regularization is added after averaging the unregularized costs
        regularization = 0.0
        if lambda_reg > 0:
            S = 0.0
            for theta in self.thetas:
                S += np.sum(theta[:, 1:] ** 2)
            regularization = (lambda_reg / (2 * num_instances)) * S

        final_cost = total_cost / num_instances + regularization
        print(f"Final (regularized) cost, J, based on the complete training set: {final_cost:0.5f}")
        print("\n\n")

        print("--------------------------------------------")
        print("Running backpropagation")

        # get the final gradients over the whole training set
        gradients, _ = self.backpropagation(X, y, lambda_reg)

        for i in range(num_instances):
            x_i = X[i:i + 1]
            y_i = y[i:i + 1]
            _, a_values = self.forward_propagation(x_i)

            # print deltas and gradients without regularization
            _, instance_deltas = self.backpropagation(x_i, y_i, lambda_reg=0)

            print(f"\tComputing gradients based on training instance {i + 1}")

            # print deltas from output layer back toward the first hidden layer
            for layer in range(self.num_layers - 1, 0, -1):
                print(f"\t\tdelta{layer + 1}: [{format_vector(instance_deltas[layer].ravel())}]")

            print("\t\t")

            # single instance gradient uses the same formula with m = 1
            for layer in range(self.num_layers - 2, -1, -1):
                a_with_bias = np.column_stack([np.ones(1), a_values[layer]])
                instance_gradient = instance_deltas[layer + 1].T @ a_with_bias
                theta_number = layer + 1

                print(f"\t\tGradients of Theta{theta_number} based on training instance {i + 1}:")
                for row in instance_gradient:
                    print(f"\t\t\t{format_vector(row)}  ")
                print()

        print("\tThe entire training set has been processed. Computing the average (regularized) gradients:")
        for theta_number, gradient in enumerate(gradients, start=1):
            print(f"\t\tFinal regularized gradients of Theta{theta_number}:")
            for row in gradient:
                print(f"\t\t\t{format_vector(row)}  ")
            print()