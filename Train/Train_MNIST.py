"""
Training script for Fashion_MNIST digit classification
Step 1 of the project plan
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from Src.feed_forward import Neuralnetwork
from Utils.data_loader import load_Fashion_MNIST, one_hot_encode, prepare_data_splits
from Utils.Visualization import (plot_training_history, plot_confusion_matrix, plot_sample_predictions, plot_misclassified)


def main():
    print("=" * 60)
    print("Fashion_MNIST Digit Classification - From Scratch Neural Network")
    print("=" * 60)
    
    # Load Fashion_MNIST data
    (X_train, y_train), (X_test, y_test) = load_Fashion_MNIST(normalize=True, flatten=True)
    
    # Prepare training and validation sets
    X_train, y_train_encoded, X_val, y_val_encoded = prepare_data_splits(X_train, y_train, val_split=0.1, one_hot=True)
    
    # One-hot encode test labels for evaluation
    y_test_encoded = one_hot_encode(y_test, num_classes=10)
    
    print("\n" + "=" * 60)
    print("Building Neural Network")
    print("=" * 60)
    
    # Create neural network
    # Architecture: 784 -> 128 -> 64 -> 10
    model = Neuralnetwork(
        layer_size=[784, 128, 64, 10],
        learning_rate=0.001,
        activation='relu'
    )
    
    model.summary()
    
    print("\n" + "=" * 60)
    print("Training Network")
    print("=" * 60)
    
    # Train the model
    history = model.train(
        X_train=X_train,
        y_train=y_train_encoded,
        X_val=X_val,
        y_val=y_val_encoded,
        epochs=20,
        batch_size=128,
        verbose=True
    )
    
    print("\n" + "=" * 60)
    print("Evaluating on Test Set")
    print("=" * 60)
    
    # Evaluate on test set
    test_predictions = model.predict(X_test)
    test_accuracy = np.mean(test_predictions == y_test)
    
    print(f"\nTest Accuracy: {test_accuracy * 100:.2f}%")
    
    # Get prediction probabilities for test set
    test_probs = model.predict_proba(X_test)
    test_loss = model.cross_entropy_loss(test_probs, y_test_encoded)
    print(f"Test Loss: {test_loss:.4f}")
    
    # Save the model
    model_path = '../Neural-Network-from-Scratch/models/Fashion_MNIST_model.json'
    os.makedirs('../Neural-Network-from-Scratch/models', exist_ok=True)
    model.save(model_path)
    
    print("\n" + "=" * 60)
    print("Generating Visualizations")
    print("=" * 60)
    
    # Create visualizations directory
    os.makedirs('../Neural-Network-from-Scratch/results/Fashion_MNIST', exist_ok=True)
    
    # Plot training history
    plot_training_history(history, save_path='../Neural-Network-from-Scratch/results/Fashion_MNIST/Fashion_MNIST_training_history.png')
    
    # Plot confusion matrix
    plot_confusion_matrix(y_test, test_predictions,  class_names=[str(i) for i in range(10)], save_path='../Neural-Network-from-Scratch/results/Fashion_MNIST/Fashion_MNIST_confusion_matrix.png')
    
    # Plot sample predictions
    plot_sample_predictions(X_test, y_test, test_predictions, num_samples=20, save_path='../Neural-Network-from-Scratch/results/Fashion_MNIST/Fashion_MNIST_sample_predictions.png')
    
    # Plot misclassified samples
    plot_misclassified(X_test, y_test, test_predictions, num_samples=15, save_path='../Neural-Network-from-Scratch/results/Fashion_MNIST/Fashion_MNIST_misclassified.png')
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Final Test Accuracy: {test_accuracy * 100:.2f}%")
    print(f"Model saved to: {model_path}")
    print(f"Results saved to: ../Neural-Network-from-Scratch/results/Fashion_MNIST/")


if __name__ == "__main__":
    main()