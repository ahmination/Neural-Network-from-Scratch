"""
Data loading utilities for MNIST, MNIST-Corrupted, and EMNIST
"""

import numpy as np
from tensorflow import keras
from typing import Tuple
import os


def load_mnist(normalize: bool = True, flatten: bool = True) -> Tuple[Tuple, Tuple]:
    """
    Load MNIST dataset
    
    Parameters:
    -----------
    normalize : bool
        Normalize pixel values to [0, 1]
    flatten : bool
        Flatten images from 28x28 to 784
    
    Returns:
    --------
    (X_train, y_train), (X_test, y_test)
    """
    print("Loading MNIST dataset...")
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
    
    if flatten:
        X_train = X_train.reshape(-1, 28 * 28)
        X_test = X_test.reshape(-1, 28 * 28)
    
    if normalize:
        X_train = X_train.astype('float32') / 255.0
        X_test = X_test.astype('float32') / 255.0
    
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    print(f"Input shape: {X_train.shape[1:]}")
    
    return (X_train, y_train), (X_test, y_test)


def load_fashion_mnist(normalize: bool = True, flatten: bool = True) -> Tuple[Tuple, Tuple]:
    """
    Load Fashion MNIST dataset (alternative dataset for testing)
    
    Parameters:
    -----------
    normalize : bool
        Normalize pixel values to [0, 1]
    flatten : bool
        Flatten images from 28x28 to 784
    
    Returns:
    --------
    (X_train, y_train), (X_test, y_test)
    """
    print("Loading Fashion MNIST dataset...")
    (X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
    
    if flatten:
        X_train = X_train.reshape(-1, 28 * 28)
        X_test = X_test.reshape(-1, 28 * 28)
    
    if normalize:
        X_train = X_train.astype('float32') / 255.0
        X_test = X_test.astype('float32') / 255.0
    
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    print(f"Input shape: {X_train.shape[1:]}")
    
    return (X_train, y_train), (X_test, y_test)


def one_hot_encode(y: np.ndarray, num_classes: int = 10) -> np.ndarray:
    """
    Convert labels to one-hot encoded format
    
    Parameters:
    -----------
    y : np.ndarray
        Label array
    num_classes : int
        Number of classes
    
    Returns:
    --------
    one_hot : np.ndarray
        One-hot encoded labels
    """
    one_hot = np.zeros((y.shape[0], num_classes))
    one_hot[np.arange(y.shape[0]), y] = 1
    return one_hot


def add_noise(X: np.ndarray, noise_type: str = 'gaussian', intensity: float = 0.2) -> np.ndarray:
    """
    Add noise to images to simulate corrupted data
    
    Parameters:
    -----------
    X : np.ndarray
        Input images
    noise_type : str
        Type of noise ('gaussian', 'salt_pepper', 'blur')
    intensity : float
        Noise intensity
    
    Returns:
    --------
    X_noisy : np.ndarray
        Noisy images
    """
    X_noisy = X.copy()
    
    if noise_type == 'gaussian':
        noise = np.random.normal(0, intensity, X.shape)
        X_noisy = X_noisy + noise
        X_noisy = np.clip(X_noisy, 0, 1)
    
    elif noise_type == 'salt_pepper':
        # Salt and pepper noise
        prob = intensity
        mask = np.random.random(X.shape)
        X_noisy[mask < prob/2] = 0  # Pepper
        X_noisy[mask > 1 - prob/2] = 1  # Salt
    
    elif noise_type == 'blur':
        # Simple blur by averaging with neighbors
        # This is a simplified version for flattened images
        if len(X.shape) == 2:  # Flattened
            # Reshape to 28x28, apply blur, flatten again
            original_shape = X.shape
            X_reshaped = X.reshape(-1, 28, 28)
            X_blurred = np.zeros_like(X_reshaped)
            
            for i in range(X_reshaped.shape[0]):
                for row in range(28):
                    for col in range(28):
                        # Average with neighbors
                        neighbors = []
                        for dr in [-1, 0, 1]:
                            for dc in [-1, 0, 1]:
                                r, c = row + dr, col + dc
                                if 0 <= r < 28 and 0 <= c < 28:
                                    neighbors.append(X_reshaped[i, r, c])
                        X_blurred[i, row, col] = np.mean(neighbors)
            
            X_noisy = X_blurred.reshape(original_shape)
            # Mix with original based on intensity
            X_noisy = (1 - intensity) * X + intensity * X_noisy
    
    return X_noisy


def create_corrupted_mnist(X: np.ndarray, corruption_types: list = None) -> np.ndarray:
    """
    Create corrupted version of MNIST for robustness testing
    
    Parameters:
    -----------
    X : np.ndarray
        Original MNIST images
    corruption_types : list
        List of corruption types to apply
    
    Returns:
    --------
    X_corrupted : np.ndarray
        Corrupted images
    """
    if corruption_types is None:
        corruption_types = ['gaussian', 'salt_pepper', 'blur']
    
    print(f"Creating corrupted dataset with: {corruption_types}")
    
    # Split data into chunks and apply different corruptions
    chunk_size = len(X) // len(corruption_types)
    X_corrupted = []
    
    for i, corruption in enumerate(corruption_types):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size if i < len(corruption_types) - 1 else len(X)
        
        X_chunk = X[start_idx:end_idx]
        X_chunk_corrupted = add_noise(X_chunk, noise_type=corruption, intensity=0.3)
        X_corrupted.append(X_chunk_corrupted)
    
    X_corrupted = np.vstack(X_corrupted)
    
    return X_corrupted


def prepare_data_splits(X: np.ndarray, y: np.ndarray, val_split: float = 0.1, one_hot: bool = True) -> Tuple:
    """
    Prepare train/validation splits
    
    Parameters:
    -----------
    X : np.ndarray
        Features
    y : np.ndarray
        Labels
    val_split : float
        Validation split ratio
    one_hot : bool
        Whether to one-hot encode labels
    
    Returns:
    --------
    X_train, y_train, X_val, y_val
    """
    n_samples = X.shape[0]
    n_val = int(n_samples * val_split)
    
    # Shuffle
    indices = np.random.permutation(n_samples)
    X_shuffled = X[indices]
    y_shuffled = y[indices]
    
    # Split
    X_val = X_shuffled[:n_val]
    y_val = y_shuffled[:n_val]
    X_train = X_shuffled[n_val:]
    y_train = y_shuffled[n_val:]
    
    # One-hot encode if requested
    if one_hot:
        num_classes = len(np.unique(y))
        y_train = one_hot_encode(y_train, num_classes)
        y_val = one_hot_encode(y_val, num_classes)
    
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Validation samples: {X_val.shape[0]}")
    
    return X_train, y_train, X_val, y_val


class DataAugmenter:
    """Simple data augmentation for digit images"""
    
    @staticmethod
    def shift(X: np.ndarray, max_shift: int = 2) -> np.ndarray:
        """Random shift of images"""
        if len(X.shape) == 2:  # Flattened
            X_reshaped = X.reshape(-1, 28, 28)
        else:
            X_reshaped = X
        
        X_shifted = np.zeros_like(X_reshaped)
        
        for i in range(X_reshaped.shape[0]):
            shift_x = np.random.randint(-max_shift, max_shift + 1)
            shift_y = np.random.randint(-max_shift, max_shift + 1)
            
            X_shifted[i] = np.roll(X_reshaped[i], shift_x, axis=1)
            X_shifted[i] = np.roll(X_shifted[i], shift_y, axis=0)
        
        if len(X.shape) == 2:
            X_shifted = X_shifted.reshape(-1, 28 * 28)
        
        return X_shifted
    
    @staticmethod
    def augment_batch(X: np.ndarray, y: np.ndarray, augmentation_factor: int = 2) -> Tuple:
        """
        Augment a batch of data
        
        Parameters:
        -----------
        X : np.ndarray
            Images
        y : np.ndarray
            Labels
        augmentation_factor : int
            How many augmented versions to create per sample
        
        Returns:
        --------
        X_augmented, y_augmented
        """
        X_list = [X]
        y_list = [y]
        
        for _ in range(augmentation_factor - 1):
            X_shifted = DataAugmenter.shift(X)
            X_list.append(X_shifted)
            y_list.append(y)
        
        X_augmented = np.vstack(X_list)
        y_augmented = np.vstack(y_list)
        
        # Shuffle
        indices = np.random.permutation(len(X_augmented))
        
        return X_augmented[indices], y_augmented[indices]