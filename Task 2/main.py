import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np

from NN import NN
from preprocessing import prepare_data_for_nn, split_data_by_class, one_hot_encode, preprocess_sample
from helpers import create_confusion_matrix, calculate_accuracy, plot_confusion_matrix

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Neural Network Penguin Classifier")
        self.root.geometry("800x600")
        
        # Load data
        self.load_data()
        self.create_widgets()
    
    def load_data(self):
        try:
            X, y, self.label_encoder, self.features, self.origin_location_encoder, self.mean_imputer, self.median_imputer, self.scaler = prepare_data_for_nn()
            self.X_train, self.y_train, self.X_test, self.y_test = split_data_by_class(X, y)
            self.y_train_onehot = one_hot_encode(self.y_train, 3)
            
            print("Data loaded successfully!")
            print(f"Training samples: {len(self.X_train)}")
            print(f"Test samples: {len(self.X_test)}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data: {str(e)}")
    
    def create_widgets(self):
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
        try:
            # Get parameters
            num_layers = int(self.layers_entry.get())
            neurons = [int(x.strip()) for x in self.neurons_entry.get().split(",")]
            
            if len(neurons) == 1:
                num_of_neurons_of_each_hidden_layer = neurons * num_layers
            elif len(neurons) == num_layers:
                num_of_neurons_of_each_hidden_layer = neurons
            else:
                messagebox.showerror("Error", 
                    f"Mismatch: You specified {num_layers} hidden layers")
                return
            
            lr = float(self.lr_entry.get())
            num_epochs = int(self.epochs_entry.get())
            activation = self.activation_var.get()
            use_bias = self.bias_var.get()
            
            # Create network
            self.network = NN(
                input_sz=5,
                hidden_layers=num_of_neurons_of_each_hidden_layer,
                output_sz=3,
                activation_func=activation,
                use_bias=use_bias
            )
            
            # Display info
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, "=== Training Started ===\n")
            self.results_text.insert(tk.END, f"Architecture: 5 input - {num_of_neurons_of_each_hidden_layer} hidden - 3 output\n")
            self.results_text.insert(tk.END, f"Learning rate: {lr}, Epochs: {num_epochs}\n")
            self.results_text.insert(tk.END, f"Activation: {activation}, Bias: {use_bias}\n")
            self.results_text.insert(tk.END, "-" * 50 + "\n")
            
            # Train
            loss_history, accuracy_history = self.network.train(self.X_train, self.y_train_onehot, lr, num_epochs)
            
            self.results_text.insert(tk.END, f"\nTraining completed!\n")
            self.results_text.insert(tk.END, f"Final loss: {float(loss_history[-1]):.4f}\n")
            self.results_text.insert(tk.END, f"Final accuracy: {float(accuracy_history[-1])*100:.2f}%\n")

        except Exception as e:
            messagebox.showerror("Error", f"Training failed: {str(e)}")
    
    def test_network(self):
        try:
            if not hasattr(self, 'network'):
                messagebox.showerror("Error", "Please train the network first!")
                return
            
            # Make predictions
            predictions = self.network.predict(self.X_test)
            
            # Create confusion matrix and calculate accuracy using helpers
            cm = create_confusion_matrix(self.y_test, predictions, 3)
            accuracy = calculate_accuracy(cm)
            
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
            plot_confusion_matrix(cm, class_names)
            
        except Exception as e:
            messagebox.showerror("Error", f"Testing failed: {str(e)}")
    
    def classify_sample(self):
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
            origin_location_values = list(self.origin_location_encoder.classes_)
            for i, feature in enumerate(self.features):
                tk.Label(input_frame, text=feature).grid(row=i, column=0, padx=5, pady=2, sticky="w")
                if feature == 'OriginLocation':
                    entry = ttk.Combobox(input_frame, values=origin_location_values, state='readonly')
                    entry.current(0)
                else:
                    entry = tk.Entry(input_frame)
                entry.grid(row=i, column=1, padx=5, pady=2)
                entries[feature] = entry
            
            result_label = tk.Label(input_window, text="", font=("Arial", 10))
            result_label.pack(pady=10)
            
            def classify():
                try:
                    # Collect inputs
                    sample_dict = {}
                    for feature in self.features:
                        value = entries[feature].get()
                        if feature == 'OriginLocation':
                            sample_dict[feature] = str(value)
                        else:
                            sample_dict[feature] = float(value)

                    # Apply same preprocessing pipeline as training
                    preprocessed_sample = preprocess_sample(
                        sample_dict,
                        self.features,
                        self.origin_location_encoder,
                        self.mean_imputer,
                        self.median_imputer,
                        self.scaler
                    )

                    # Predict probabilities and class
                    probabilities = self.network.predict_prob(preprocessed_sample)[0]
                    predicted_class = np.argmax(probabilities)
                    class_name = self.label_encoder.inverse_transform([predicted_class])[0]

                    # Display results
                    result_text = f"Predicted: {class_name}\n"
                    result_text += f"Confidence: {probabilities[predicted_class] * 100:.1f}%"
                    result_label.config(text=result_text)
                except ValueError as ve:
                    messagebox.showerror("Error", f"Invalid input value: {ve}")
                except Exception as e:
                    messagebox.showerror("Error", f"Please check input values: {str(e)}")
            
            tk.Button(input_window, text="Classify", command=classify).pack(pady=10)
            
        except Exception as e:
            messagebox.showerror("Error", f"Classification failed: {str(e)}")

def main():
    root = tk.Tk()
    app = App(root)
    root.mainloop()

if __name__ == "__main__":
    main()