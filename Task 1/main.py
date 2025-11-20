import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from adaline import Adaline
from preprocessing import load_and_preprocess_data
from perceptron import Perceptron


# =========================================
# 🔹 Helper Function: Plot Results in Tkinter
# =========================================
def plot_results_in_tkinter(parent_frame, X_test_np, y_test_np, w, cm, feature1, feature2, model_type):
    # Clear any previous plots
    for widget in parent_frame.winfo_children():
        widget.destroy()

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    fig.suptitle(f"{model_type} Results", fontsize=14, fontweight="bold")

    # --- (1) Confusion Matrix ---
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[0])
    axes[0].set_title("Confusion Matrix")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")
    axes[0].set_xticklabels(["+1", "-1"])
    axes[0].set_yticklabels(["+1", "-1"])

    # --- (2) Decision Boundary ---
    X = X_test_np
    y = y_test_np.flatten()
    X_no_bias = X[:, :2]  # always first two columns = features

    # Scatter data points
    axes[1].scatter(
        X_no_bias[y == 1, 0], X_no_bias[y == 1, 1],
        color="blue", label="+1 (Class 1)", alpha=0.7
    )
    axes[1].scatter(
        X_no_bias[y == -1, 0], X_no_bias[y == -1, 1],
        color="red", label="-1 (Class 2)", alpha=0.7
    )

    axes[1].set_xlabel(feature1)
    axes[1].set_ylabel(feature2)
    axes[1].set_title("Decision Boundary")

    # Extract weights (always 3 because bias column exists)
    w1, w2, wbias = w.flatten()

    # Decision line: w1*x + w2*y + b = 0 → y = -(w1*x + b)/w2
    x_vals = np.linspace(X_no_bias[:, 0].min(), X_no_bias[:, 0].max(), 100)
    y_vals = -(w1 * x_vals + wbias) / w2

    axes[1].plot(x_vals, y_vals, color="green", linewidth=2, label="Decision Boundary")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()

    # Embed figure in Tkinter
    canvas = FigureCanvasTkAgg(fig, master=parent_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)


# =========================================
# 🔹 Main Function to Run Model
# =========================================
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

        # Run chosen model
        if model_type == "Adaline":
            accuracy, final_data, w, X_test_np, y_test_np, cm = Adaline(
                data, species1, species2, feature1, feature2,
                lr, epoch, mse_threshold, is_bias
            )
        else:
            accuracy, final_data, w, X_test_np, y_test_np, cm = Perceptron(
                data, species1, species2, feature1, feature2,
                lr, epoch, mse_threshold, is_bias
            )

        # Update accuracy label
        result_label.config(text=f"{model_type} Accuracy: {accuracy:.2f}%")

        # Plot results inside Tkinter
        plot_results_in_tkinter(plot_frame, X_test_np, y_test_np, w, cm, feature1, feature2, model_type)

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred:\n{e}")


# =========================================
# 🔹 Tkinter GUI Layout
# =========================================
root = tk.Tk()
root.title("Adaline & Perceptron Classifier")
root.geometry("900x700")
root.resizable(True, True)

# --- Title ---
tk.Label(root, text="Penguins Classifier", font=("Arial", 16, "bold")).pack(pady=10)

# --- Inputs Frame ---
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

# Model Type
tk.Label(frame, text="Model Type:").grid(row=4, column=0, padx=5, pady=5, sticky="e")
model_choice = ttk.Combobox(frame, values=["Adaline", "Perceptron"], state="readonly")
model_choice.current(0)
model_choice.grid(row=4, column=1, padx=5, pady=5)

# Learning Rate
tk.Label(frame, text="Learning Rate (lr):").grid(row=5, column=0, padx=5, pady=5, sticky="e")
lr_entry = tk.Entry(frame)
lr_entry.insert(0, "0.01")
lr_entry.grid(row=5, column=1, padx=5, pady=5)

# Epochs
tk.Label(frame, text="Epochs:").grid(row=6, column=0, padx=5, pady=5, sticky="e")
epoch_entry = tk.Entry(frame)
epoch_entry.insert(0, "100")
epoch_entry.grid(row=6, column=1, padx=5, pady=5)

# MSE Threshold
tk.Label(frame, text="MSE Threshold:").grid(row=7, column=0, padx=5, pady=5, sticky="e")
mse_entry = tk.Entry(frame)
mse_entry.insert(0, "0.01")
mse_entry.grid(row=7, column=1, padx=5, pady=5)

# Bias Choice
tk.Label(frame, text="Use Bias:").grid(row=8, column=0, padx=5, pady=5, sticky="e")
bias_choice = ttk.Combobox(frame, values=["True", "False"], state="readonly")
bias_choice.current(0)
bias_choice.grid(row=8, column=1, padx=5, pady=5)

# --- Buttons ---
btn_frame = tk.Frame(root)
btn_frame.pack(pady=10)

run_btn = tk.Button(btn_frame, text="Run Model", bg="#4CAF50", fg="white",
                    font=("Arial", 12, "bold"), command=run_model)
run_btn.grid(row=0, column=0, padx=10)

exit_btn = tk.Button(btn_frame, text="Exit", bg="gray", fg="white",
                     font=("Arial", 10, "bold"), command=root.destroy)
exit_btn.grid(row=0, column=1, padx=10)

# --- Result Label ---
result_label = tk.Label(root, text="Model not run yet", font=("Arial", 12))
result_label.pack(pady=10)

# --- Plot Area ---
plot_frame = tk.Frame(root, bg="white", relief="sunken", bd=2)
plot_frame.pack(fill="both", expand=True, padx=10, pady=10)

root.mainloop()
