import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def create_confusion_matrix(y_true, y_pred, num_classes):
    """Create a confusion matrix from true and predicted labels"""
    cm = np.zeros((num_classes, num_classes), dtype=int)
    
    for true_label, pred_label in zip(y_true, y_pred):
        cm[true_label, pred_label] += 1
    
    return cm

def calculate_accuracy(confusion_matrix):
    """Calculate overall accuracy from confusion matrix"""
    correct_predictions = np.trace(confusion_matrix)
    total_predictions = np.sum(confusion_matrix)
    return correct_predictions / total_predictions

def plot_confusion_matrix(cm, class_names):
    """Plot confusion matrix as heatmap"""
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()

def plot_training_loss(loss_history):
    """Plot training loss over epochs"""
    plt.figure(figsize=(8, 4))
    plt.plot(loss_history)
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_network_architecture(hidden_layers):
    """Plot network architecture diagram"""
    layers = [5] + hidden_layers + [3]  # 5 input features, 3 output classes
    
    plt.figure(figsize=(8, 4))
    plt.bar(range(len(layers)), layers)
    plt.title('Neural Network Architecture')
    plt.xlabel('Layer')
    plt.ylabel('Number of Neurons')
    plt.xticks(range(len(layers)), 
               ['Input'] + [f'Hidden {i+1}' for i in range(len(hidden_layers))] + ['Output'])
    plt.tight_layout()
    plt.show()