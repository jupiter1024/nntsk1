import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from neural_network import NeuralNetwork
from data_loader import prepare_data_for_nn, split_data_by_class, one_hot_encode

class NeuralNetworkApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Neural Network Penguin Classifier")
        self.root.geometry("800x600")
        
        # Load data
        self.load_data()
        self.create_widgets()
    
    def load_data(self):
        """Load and prepare data"""
        try:
            X, y, self.label_encoder, self.features = prepare_data_for_nn()
            self.X_train, self.y_train, self.X_test, self.y_test = split_data_by_class(X, y)
            self.y_train_onehot = one_hot_encode(self.y_train, 3)
            
            print("Data loaded successfully!")
            print(f"Training samples: {len(self.X_train)}")
            print(f"Test samples: {len(self.X_test)}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data: {str(e)}")
    
    def create_widgets(self):
        """Create GUI interface"""
        # Main title
        title = tk.Label(self.root, text="Neural Network Classifier", 
                        font=("Arial", 16, "bold"))
        title.pack(pady=10)
        
        # Configuration frame
        config_frame = ttk.LabelFrame(self.root, text="Network Settings")
        config_frame.pack(fill="x", padx=10, pady=5)
        
        # Hidden layers
        tk.Label(config_frame, text="Hidden Layers:").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        self.layers_entry = tk.Entry(config_frame)
        self.layers_entry.insert(0, "1")
        self.layers_entry.grid(row=0, column=1, padx=5, pady=2)
        
        # Neurons
        tk.Label(config_frame, text="Neurons per Layer:").grid(row=1, column=0, sticky="w", padx=5, pady=2)
        self.neurons_entry = tk.Entry(config_frame)
        self.neurons_entry.insert(0, "10")
        self.neurons_entry.grid(row=1, column=1, padx=5, pady=2)
        
        # Learning rate
        tk.Label(config_frame, text="Learning Rate:").grid(row=2, column=0, sticky="w", padx=5, pady=2)
        self.lr_entry = tk.Entry(config_frame)
        self.lr_entry.insert(0, "0.01")
        self.lr_entry.grid(row=2, column=1, padx=5, pady=2)
        
        # Epochs
        tk.Label(config_frame, text="Epochs:").grid(row=3, column=0, sticky="w", padx=5, pady=2)
        self.epochs_entry = tk.Entry(config_frame)
        self.epochs_entry.insert(0, "1000")
        self.epochs_entry.grid(row=3, column=1, padx=5, pady=2)
        
        # Activation function
        tk.Label(config_frame, text="Activation:").grid(row=4, column=0, sticky="w", padx=5, pady=2)
        self.activation_var = tk.StringVar(value="sigmoid")
        tk.Radiobutton(config_frame, text="Sigmoid", variable=self.activation_var, value="sigmoid").grid(row=4, column=1, sticky="w")
        tk.Radiobutton(config_frame, text="Tanh", variable=self.activation_var, value="tanh").grid(row=4, column=2, sticky="w")
        
        # Bias
        self.bias_var = tk.BooleanVar(value=True)
        tk.Checkbutton(config_frame, text="Use Bias", variable=self.bias_var).grid(row=5, column=0, sticky="w", padx=5, pady=2)
        
        # Buttons
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=10)
        
        tk.Button(button_frame, text="Train Network", command=self.train_network, 
                 bg="lightblue").pack(side="left", padx=5)
        tk.Button(button_frame, text="Test Network", command=self.test_network,
                 bg="lightgreen").pack(side="left", padx=5)
        tk.Button(button_frame, text="Classify Sample", command=self.classify_sample,
                 bg="lightyellow").pack(side="left", padx=5)
        
        # Results area
        self.results_text = tk.Text(self.root, height=20, width=80)
        scrollbar = tk.Scrollbar(self.root, command=self.results_text.yview)
        self.results_text.config(yscrollcommand=scrollbar.set)
        self.results_text.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        scrollbar.pack(side="right", fill="y", pady=10)
    
    def train_network(self):
        """Train the neural network"""
        try:
            # Get parameters
            num_layers = int(self.layers_entry.get())
            neurons = [int(x.strip()) for x in self.neurons_entry.get().split(",")]
            
            if len(neurons) == 1:
                hidden_layers = neurons * num_layers
            else:
                hidden_layers = neurons
            
            learning_rate = float(self.lr_entry.get())
            epochs = int(self.epochs_entry.get())
            activation = self.activation_var.get()
            use_bias = self.bias_var.get()
            
            # Create network
            self.network = NeuralNetwork(
                input_size=5,
                hidden_layers=hidden_layers,
                output_size=3,
                activation_func=activation,
                use_bias=use_bias
            )
            
            # Display info
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, "=== Training Started ===\n")
            self.results_text.insert(tk.END, f"Architecture: 5 input - {hidden_layers} hidden - 3 output\n")
            self.results_text.insert(tk.END, f"Learning rate: {learning_rate}, Epochs: {epochs}\n")
            self.results_text.insert(tk.END, f"Activation: {activation}, Bias: {use_bias}\n")
            self.results_text.insert(tk.END, "-" * 50 + "\n")
            
            # Train
            loss_history = self.network.train(self.X_train, self.y_train_onehot, learning_rate, epochs)
            
            self.results_text.insert(tk.END, f"\nTraining completed!\n")
            self.results_text.insert(tk.END, f"Final loss: {loss_history[-1]:.4f}\n")
            
        except Exception as e:
            messagebox.showerror("Error", f"Training failed: {str(e)}")
    
    def test_network(self):
        """Test the trained network"""
        try:
            if not hasattr(self, 'network'):
                messagebox.showerror("Error", "Please train the network first!")
                return
            
            # Make predictions
            predictions = self.network.predict(self.X_test)
            
            # Create confusion matrix
            cm = np.zeros((3, 3), dtype=int)
            for true, pred in zip(self.y_test, predictions):
                cm[true, pred] += 1
            
            # Calculate accuracy
            accuracy = np.trace(cm) / np.sum(cm)
            
            # Display results
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, "=== Testing Results ===\n")
            self.results_text.insert(tk.END, f"Overall Accuracy: {accuracy * 100:.2f}%\n\n")
            
            self.results_text.insert(tk.END, "Confusion Matrix:\n")
            class_names = self.label_encoder.classes_
            
            self.results_text.insert(tk.END, "     Predicted →\n")
            self.results_text.insert(tk.END, "True ")
            for name in class_names:
                self.results_text.insert(tk.END, f"{name:>10}")
            self.results_text.insert(tk.END, "\n")
            
            for i, true_name in enumerate(class_names):
                self.results_text.insert(tk.END, f"{true_name:5}")
                for j in range(3):
                    self.results_text.insert(tk.END, f"{cm[i, j]:>10}")
                self.results_text.insert(tk.END, "\n")
            
            # Plot confusion matrix
            self.plot_confusion_matrix(cm, class_names)
            
        except Exception as e:
            messagebox.showerror("Error", f"Testing failed: {str(e)}")
    
    def classify_sample(self):
        """Classify a single sample"""
        try:
            if not hasattr(self, 'network'):
                messagebox.showerror("Error", "Please train the network first!")
                return
            
            # Create input window
            input_window = tk.Toplevel(self.root)
            input_window.title("Classify Sample")
            input_window.geometry("300x300")
            
            tk.Label(input_window, text="Enter feature values:").pack(pady=10)
            
            # Input fields
            input_frame = tk.Frame(input_window)
            input_frame.pack(pady=10)
            
            entries = {}
            for i, feature in enumerate(self.features):
                tk.Label(input_frame, text=feature).grid(row=i, column=0, padx=5, pady=2, sticky="w")
                entry = tk.Entry(input_frame)
                entry.grid(row=i, column=1, padx=5, pady=2)
                entries[feature] = entry
            
            result_label = tk.Label(input_window, text="", font=("Arial", 10))
            result_label.pack(pady=10)
            
            def classify():
                try:
                    # Get input values
                    sample = []
                    for feature in self.features:
                        value = float(entries[feature].get())
                        sample.append(value)
                    
                    # Predict
                    sample_array = np.array([sample])
                    probabilities = self.network.predict_proba(sample_array)[0]
                    predicted_class = np.argmax(probabilities)
                    class_name = self.label_encoder.inverse_transform([predicted_class])[0]
                    
                    # Display results
                    result_text = f"Predicted: {class_name}\n"
                    result_text += f"Confidence: {probabilities[predicted_class] * 100:.1f}%"
                    result_label.config(text=result_text)
                    
                except Exception as e:
                    messagebox.showerror("Error", f"Please check input values: {str(e)}")
            
            tk.Button(input_window, text="Classify", command=classify).pack(pady=10)
            
        except Exception as e:
            messagebox.showerror("Error", f"Classification failed: {str(e)}")
    
    def plot_confusion_matrix(self, cm, class_names):
        """Plot confusion matrix"""
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.show()

def main():
    root = tk.Tk()
    app = NeuralNetworkApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()