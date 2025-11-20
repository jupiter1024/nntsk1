import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_layers, output_size, activation_func='sigmoid', use_bias=True):
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.activation_func = activation_func
        self.use_bias = use_bias
        
        # Initialize weights and biases
        self.weights = []
        self.biases = []
        
        # Layer sizes
        layer_sizes = [input_size] + hidden_layers + [output_size]
        
        # Initialize weights for each layer
        for i in range(len(layer_sizes) - 1):
            # Random weights
            weight = np.random.uniform(-0.5, 0.5, (layer_sizes[i], layer_sizes[i+1]))
            self.weights.append(weight)
            
            if use_bias:
                bias = np.random.uniform(-0.1, 0.1, (1, layer_sizes[i+1]))
                self.biases.append(bias)
            else:
                self.biases.append(np.zeros((1, layer_sizes[i+1])))
    
    def sigmoid(self, x):
        x = np.clip(x, -250, 250)  # Prevent overflow
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def tanh(self, x):
        return np.tanh(x)
    
    def tanh_derivative(self, x):
        return 1 - np.tanh(x) ** 2
    
    def softmax(self, x):
        # Stable softmax
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward(self, X):
        # Store layer outputs for backpropagation
        self.layer_outputs = [X]
        self.layer_inputs = []
        
        current_output = X
        
        # Forward pass through all layers
        for i in range(len(self.weights)):
            # Calculate net input
            net_input = np.dot(current_output, self.weights[i]) + self.biases[i]
            self.layer_inputs.append(net_input)
            
            # Apply activation function
            if i == len(self.weights) - 1:  # Output layer
                current_output = self.softmax(net_input)
            else:  # Hidden layers
                if self.activation_func == 'sigmoid':
                    current_output = self.sigmoid(net_input)
                else:  # tanh
                    current_output = self.tanh(net_input)
            
            self.layer_outputs.append(current_output)
        
        return current_output
    
    def backward(self, X, y, output, learning_rate):
        m = X.shape[0]  # Number of samples
        deltas = [None] * len(self.weights)
        
        # Output layer error
        error = output - y
        deltas[-1] = error
        
        # Backpropagate through hidden layers
        for i in range(len(self.weights) - 2, -1, -1):
            error = deltas[i+1].dot(self.weights[i+1].T)
            
            if self.activation_func == 'sigmoid':
                delta = error * self.sigmoid_derivative(self.layer_outputs[i+1])
            else:  # tanh
                delta = error * self.tanh_derivative(self.layer_outputs[i+1])
            
            deltas[i] = delta
        
        # Update weights and biases
        for i in range(len(self.weights)):
            dW = self.layer_outputs[i].T.dot(deltas[i]) / m
            dB = np.sum(deltas[i], axis=0, keepdims=True) / m
            
            self.weights[i] -= learning_rate * dW
            if self.use_bias:
                self.biases[i] -= learning_rate * dB
    
    def train(self, X, y, learning_rate, epochs):
        loss_history = []
        
        for epoch in range(epochs):
            # Forward pass
            output = self.forward(X)
            
            # Calculate loss
            loss = -np.mean(np.sum(y * np.log(output + 1e-8), axis=1))
            loss_history.append(loss)
            
            # Backward pass
            self.backward(X, y, output, learning_rate)
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
        
        return loss_history
    
    def predict(self, X):
        output = self.forward(X)
        return np.argmax(output, axis=1)
    
    def predict_proba(self, X):
        return self.forward(X)