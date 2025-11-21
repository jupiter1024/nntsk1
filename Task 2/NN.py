import numpy as np

class NN:
    def __init__(self, input_sz, hidden_layers, output_sz, activation_func='sigmoid', use_bias=True):
        self.input_size, self.output_size= input_sz, output_sz
        self.hidden_layers = np.array(hidden_layers)
        self.activation_func = activation_func
        self.use_bias = use_bias
        
        # Init weights and biases
        self.weights, self.biases = [], []
        layer_sizes = [input_sz] + list(hidden_layers) + [output_sz]
        
        for i in range(len(layer_sizes) - 1):
            self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.5)
            if self.use_bias:
                self.biases.append(np.random.randn(1, layer_sizes[i+1]) * 0.1)
            else:
                self.biases.append(np.zeros((1, layer_sizes[i+1])))
    
    def sigmoid(self, x):
        x = np.clip(x, -250, 250) # to prevent Overflow
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def tanh(self, x):
        return np.tanh(x)
    
    def tanh_derivative(self, x):
        return 1 - x ** 2
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward(self, x):
        x = np.asarray(x, dtype=np.float64)
        self.layer_outputs = [x.copy()]
        self.layer_inputs = []
        res = x.copy()
        num_layers = len(self.weights)
        for i in range(num_layers):
            net_inp = np.dot(res, self.weights[i]) + self.biases[i]
            self.layer_inputs.append(net_inp.copy())
            fn = self.sigmoid if self.activation_func == 'sigmoid' else self.tanh
            if i == num_layers - 1:  # Output layer
                res = self.softmax(net_inp)
            else:  # Hidden layers
                res = fn(net_inp)
            self.layer_outputs.append(res.copy())
        return res
    
    def backward(self, x, y, output, lr):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        output = np.asarray(output, dtype=np.float64)
        
        if x.ndim == 1:
            x = x.reshape(1, -1)
            y = y.reshape(1, -1) if y.ndim == 1 else y.reshape(1, -1)
            output = output.reshape(1, -1) if output.ndim == 1 else output.reshape(1, -1)
        
        m = x.shape[0]
        num_layers = len(self.weights)
        deltas = [None] * num_layers
        
        deltas[-1] = y - output
        
        # Backpropagate
        if num_layers > 1:
            for i in reversed(range(num_layers - 1)):
                error = np.dot(deltas[i+1], self.weights[i+1].T)
                fn = self.sigmoid_derivative if self.activation_func == 'sigmoid' else self.tanh_derivative
                deltas[i] = error * fn(self.layer_outputs[i+1])
        
        # Update weights and biases
        for i in range(num_layers):
            dW = np.dot(self.layer_outputs[i].T, deltas[i])
            dB = np.sum(deltas[i], axis=0, keepdims=True)
            self.weights[i] += lr * dW
            if self.use_bias:
                self.biases[i] += lr * dB
    
    def train(self, x, y, lr, epochs):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        loss_history = []
        
        num_samples = x.shape[0]
        
        for epoch in range(epochs):
            epoch_loss = 0
            
            # Shuffle samples
            indices = np.random.permutation(num_samples)
            x_shuffled = x[indices]
            y_shuffled = y[indices]
            
            # Train sample by sample
            for sample_idx in range(num_samples):
                x_sample = x_shuffled[sample_idx:sample_idx+1]  # Keep 2D shape
                y_sample = y_shuffled[sample_idx:sample_idx+1]  # Keep 2D shape
                
                # Forward
                output = self.forward(x_sample)
                
                sample_loss = -np.sum(y_sample * np.log(output + 1e-8))
                epoch_loss += sample_loss
                
                # Backward
                self.backward(x_sample, y_sample, output, lr)
            
            avg_loss = epoch_loss / num_samples
            loss_history.append(avg_loss)
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")
    
        return np.array(loss_history)
    
    def predict_prob(self, x):
        x = np.asarray(x, dtype=np.float64)
        return self.forward(x)
    def predict(self, x):
        return np.argmax(self.predict_prob(x), axis=1)
    