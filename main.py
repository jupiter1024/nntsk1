import tkinter as tk
from tkinter import ttk, messagebox
from adaline import Preprocessing_After_input2
from preprocessing import load_and_preprocess_data
from perceptron import Preprocessing_After_input_perceptron

def run_model():
    try:
        species1 = species1_entry.get().strip()
        species2 = species2_entry.get().strip()
        feature1 = feature1_entry.get().strip()
        feature2 = feature2_entry.get().strip()
        model_type = model_choice.get()
        lr = float(lr_entry.get())
        epoch = int(epoch_entry.get())
        mse_threshold = float(mse_entry.get())
        is_bias = bias_choice.get() == "True"

        if not all([species1, species2, feature1, feature2]):
            messagebox.showerror("Input Error", "Please fill all text fields!")
            return

        data = load_and_preprocess_data()

        if model_type == "Adaline":
            accuracy, final_data, w = Preprocessing_After_input2(
                data,
                species1,
                species2,
                feature1,
                feature2,
                lr,
                epoch,
                mse_threshold,
                is_bias
            )
        else:
            accuracy, final_data, w = Preprocessing_After_input_perceptron(
                data,
                species1,
                species2,
                feature1,
                feature2,
                lr,
                epoch,
                mse_threshold,
                is_bias
            )

        messagebox.showinfo("Result", f"Model: {model_type}\nAccuracy: {accuracy:.2f}%")

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred:\n{e}")



root = tk.Tk()
root.title("Adaline & Perceptron Classifier")
root.geometry("420x500")
root.resizable(False, False)

# Title Label
tk.Label(root, text="Penguins Classifier", font=("Arial", 16, "bold")).pack(pady=10)

frame = tk.Frame(root)
frame.pack(pady=10)

# Species
tk.Label(frame, text="Species 1:").grid(row=0, column=0, padx=5, pady=5, sticky="e")
species1_entry = tk.Entry(frame)
species1_entry.grid(row=0, column=1, padx=5, pady=5)

tk.Label(frame, text="Species 2:").grid(row=1, column=0, padx=5, pady=5, sticky="e")
species2_entry = tk.Entry(frame)
species2_entry.grid(row=1, column=1, padx=5, pady=5)

# Features
tk.Label(frame, text="Feature 1:").grid(row=2, column=0, padx=5, pady=5, sticky="e")
feature1_entry = tk.Entry(frame)
feature1_entry.grid(row=2, column=1, padx=5, pady=5)

tk.Label(frame, text="Feature 2:").grid(row=3, column=0, padx=5, pady=5, sticky="e")
feature2_entry = tk.Entry(frame)
feature2_entry.grid(row=3, column=1, padx=5, pady=5)

# Model Choice
tk.Label(frame, text="Model Type:").grid(row=4, column=0, padx=5, pady=5, sticky="e")
model_choice = ttk.Combobox(frame, values=["Adaline", "Perceptron"], state="readonly")
model_choice.current(0)
model_choice.grid(row=4, column=1, padx=5, pady=5)

# Learning Rate
tk.Label(frame, text="Learning Rate (lr):").grid(row=5, column=0, padx=5, pady=5, sticky="e")
lr_entry = tk.Entry(frame)
lr_entry.insert(0, "0.01")
lr_entry.grid(row=5, column=1, padx=5, pady=5)

# Epoch
tk.Label(frame, text="Epochs:").grid(row=6, column=0, padx=5, pady=5, sticky="e")
epoch_entry = tk.Entry(frame)
epoch_entry.insert(0, "100")
epoch_entry.grid(row=6, column=1, padx=5, pady=5)

# MSE Threshold
tk.Label(frame, text="MSE Threshold:").grid(row=7, column=0, padx=5, pady=5, sticky="e")
mse_entry = tk.Entry(frame)
mse_entry.insert(0, "0.01")
mse_entry.grid(row=7, column=1, padx=5, pady=5)

# Bias
tk.Label(frame, text="Use Bias:").grid(row=8, column=0, padx=5, pady=5, sticky="e")
bias_choice = ttk.Combobox(frame, values=["True", "False"], state="readonly")
bias_choice.current(0)
bias_choice.grid(row=8, column=1, padx=5, pady=5)

# Run Button
run_btn = tk.Button(root, text="Run Model", bg="#4CAF50", fg="white", font=("Arial", 12, "bold"), command=run_model)
run_btn.pack(pady=20)

# Exit Button
exit_btn = tk.Button(root, text="Exit", bg="gray", fg="white", font=("Arial", 10, "bold"), command=root.destroy)
exit_btn.pack(pady=5)

root.mainloop()
