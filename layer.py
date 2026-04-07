import numpy as np

class Layer:
    def __init__(self):
        self.input = None
        self.output = None
        self.weights = None
        self.bias = None

    def forward(self, input):
        # Implementation based on referential linear forward pass: Z = W.dot(X) + b
        self.input = input
        self.output = np.dot(self.weights, self.input) + self.bias 
        return self.output

    def backward(self, output_gradient, learning_rate):
        # m = number of training examples in the batch
        m = self.input.shape[1] if len(self.input.shape) > 1 else 1

        # Gradients calculation based on the user's provided math:
        # dW = 1 / m * dZ.dot(X.T)
        weights_gradient = (1 / m) * np.dot(output_gradient, self.input.T)
        # db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
        bias_gradient = (1 / m) * np.sum(output_gradient, axis=1, keepdims=True)
        
        # Gradient passed back to previous layer: W.T.dot(dZ)
        input_gradient = np.dot(self.weights.T, output_gradient)

        # Update parameters 
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * bias_gradient

        return input_gradient
