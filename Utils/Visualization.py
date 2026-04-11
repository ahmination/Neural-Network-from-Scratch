"""
Visualization utilities for neural network training and evaluation
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
import seaborn as sns


def plot_training_history(history: Dict, save_path: Optional[str] = None) -> None:
    """
    Plot training history (loss and accuracy)
    
    Parameters:
    -----------
    history : dict
        Training history from model.train()
    save_path : str, optional
        Path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot loss
    axes[0].plot(history['loss'], label='Training Loss', linewidth=2)
    if 'val_loss' in history and len(history['val_loss']) > 0:
        axes[0].plot(history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Plot accuracy
    axes[1].plot(history['accuracy'], label='Training Accuracy', linewidth=2)
    if 'val_accuracy' in history and len(history['val_accuracy']) > 0:
        axes[1].plot(history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, class_names: Optional[List[str]] = None, save_path: Optional[str] = None) -> None:
    """
    Plot confusion matrix
    
    Parameters:
    -----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    class_names : list, optional
        Class names for labels
    save_path : str, optional
        Path to save the plot
    """
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    
    if class_names is None:
        class_names = [str(i) for i in range(cm.shape[0])]
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, cbar_kws={'label': 'Count'})
    
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.show()


def plot_sample_predictions(X: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, num_samples: int = 10, save_path: Optional[str] = None) -> None:
    """
    Plot sample predictions
    
    Parameters:
    -----------
    X : np.ndarray
        Input images (flattened or not)
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    num_samples : int
        Number of samples to plot
    save_path : str, optional
        Path to save the plot
    """
    # Reshape if flattened
    if len(X.shape) == 2:
        X_images = X.reshape(-1, 28, 28)
    else:
        X_images = X
    
    # Select random samples
    indices = np.random.choice(len(X), min(num_samples, len(X)), replace=False)
    
    # Create grid
    cols = 5
    rows = (num_samples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, rows * 2.5))
    axes = axes.flatten()
    
    for i, idx in enumerate(indices):
        axes[i].imshow(X_images[idx], cmap='gray')
        axes[i].axis('off')
        
        # Color code: green for correct, red for incorrect
        color = 'green' if y_true[idx] == y_pred[idx] else 'red'
        title = f"True: {y_true[idx]}\nPred: {y_pred[idx]}"
        axes[i].set_title(title, fontsize=10, color=color, fontweight='bold')
    
    # Hide unused subplots
    for i in range(len(indices), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Sample predictions saved to {save_path}")
    
    plt.show()


def plot_misclassified(X: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, num_samples: int = 10, save_path: Optional[str] = None) -> None:
    """
    Plot misclassified samples
    
    Parameters:
    -----------
    X : np.ndarray
        Input images
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    num_samples : int
        Number of samples to plot
    save_path : str, optional
        Path to save the plot
    """
    # Find misclassified samples
    misclassified_idx = np.where(y_true != y_pred)[0]
    
    if len(misclassified_idx) == 0:
        print("No misclassified samples found!")
        return
    
    # Reshape if flattened
    if len(X.shape) == 2:
        X_images = X.reshape(-1, 28, 28)
    else:
        X_images = X
    
    # Select random misclassified samples
    num_to_plot = min(num_samples, len(misclassified_idx))
    selected_idx = np.random.choice(misclassified_idx, num_to_plot, replace=False)
    
    # Create grid
    cols = 5
    rows = (num_to_plot + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, rows * 2.5))
    axes = axes.flatten()
    
    for i, idx in enumerate(selected_idx):
        axes[i].imshow(X_images[idx], cmap='gray')
        axes[i].axis('off')
        title = f"True: {y_true[idx]}\nPred: {y_pred[idx]}"
        axes[i].set_title(title, fontsize=10, color='red', fontweight='bold')
    
    # Hide unused subplots
    for i in range(num_to_plot, len(axes)):
        axes[i].axis('off')
    
    fig.suptitle(f'Misclassified Samples ({len(misclassified_idx)} total)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Misclassified samples saved to {save_path}")
    
    plt.show()


def plot_activation_heatmap(activations: np.ndarray, layer_name: str = "Layer", save_path: Optional[str] = None) -> None:
    """
    Plot heatmap of layer activations
    
    Parameters:
    -----------
    activations : np.ndarray
        Activations from a layer
    layer_name : str
        Name of the layer
    save_path : str, optional
        Path to save the plot
    """
    plt.figure(figsize=(12, 6))
    
    # Show first 100 samples and neurons
    data_to_plot = activations[:100, :min(100, activations.shape[1])]
    
    sns.heatmap(data_to_plot.T, cmap='viridis', cbar_kws={'label': 'Activation'})
    plt.xlabel('Sample', fontsize=12)
    plt.ylabel('Neuron', fontsize=12)
    plt.title(f'{layer_name} Activations', fontsize=14, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Activation heatmap saved to {save_path}")
    
    plt.show()


def compare_models(histories: Dict[str, Dict], save_path: Optional[str] = None) -> None:
    """
    Compare multiple model training histories
    
    Parameters:
    -----------
    histories : dict
        Dictionary of {model_name: history}
    save_path : str, optional
        Path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot losses
    for name, history in histories.items():
        axes[0].plot(history['loss'], label=f'{name} - Train', linewidth=2)
        if 'val_loss' in history and len(history['val_loss']) > 0:
            axes[0].plot(history['val_loss'], label=f'{name} - Val', 
                        linewidth=2, linestyle='--')
    
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Loss Comparison', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)
    
    # Plot accuracies
    for name, history in histories.items():
        axes[1].plot(history['accuracy'], label=f'{name} - Train', linewidth=2)
        if 'val_accuracy' in history and len(history['val_accuracy']) > 0:
            axes[1].plot(history['val_accuracy'], label=f'{name} - Val', linewidth=2, linestyle='--')
    
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Accuracy Comparison', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to {save_path}")
    
    plt.show()


def visualize_weights(weights: np.ndarray, layer_name: str = "Layer", num_neurons: int = 25, save_path: Optional[str] = None) -> None:
    """
    Visualize weights of first layer (useful for 784 input)
    
    Parameters:
    -----------
    weights : np.ndarray
        Weight matrix
    layer_name : str
        Name of the layer
    num_neurons : int
        Number of neurons to visualize
    save_path : str, optional
        Path to save the plot
    """
    if weights.shape[0] != 784:
        print("Weight visualization works best with 784 input features (28x28 images)")
        return
    
    num_to_plot = min(num_neurons, weights.shape[1])
    cols = 5
    rows = (num_to_plot + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(12, rows * 2.5))
    axes = axes.flatten()
    
    for i in range(num_to_plot):
        weight_img = weights[:, i].reshape(28, 28)
        axes[i].imshow(weight_img, cmap='RdBu', vmin=-1, vmax=1)
        axes[i].axis('off')
        axes[i].set_title(f'Neuron {i}', fontsize=8)
    
    # Hide unused subplots
    for i in range(num_to_plot, len(axes)):
        axes[i].axis('off')
    
    fig.suptitle(f'{layer_name} Weights', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Weight visualization saved to {save_path}")
    
    plt.show()