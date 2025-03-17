"""Neural Network model implementation using PyTorch for KIBA prediction."""

import os
import time
import json
import pickle
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Any, Optional, Union, Tuple, List

from kiba_model.modeling.models.base import BaseModel
from kiba_model.config import KIBAConfig

logger = logging.getLogger("kiba_model")

class KIBANeuralNetwork(nn.Module):
    """PyTorch neural network model for KIBA prediction."""
    
    def __init__(self, input_dim: int, hidden_layers: List[int], dropout_rate: float = 0.3, 
                activation: str = 'relu'):
        """Initialize the neural network architecture.
        
        Args:
            input_dim: Number of input features
            hidden_layers: List of hidden layer sizes
            dropout_rate: Dropout rate for regularization
            activation: Activation function ('relu' or 'leaky_relu')
        """
        super(KIBANeuralNetwork, self).__init__()
        
        # Create layers list
        layers = []
        prev_dim = input_dim
        
        # Set activation function
        if activation.lower() == 'relu':
            act_fn = nn.ReLU()
        elif activation.lower() == 'leaky_relu':
            act_fn = nn.LeakyReLU(0.1)
        else:
            act_fn = nn.ReLU()  # Default to ReLU
        
        # Build hidden layers
        for layer_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, layer_dim))
            layers.append(nn.BatchNorm1d(layer_dim))
            layers.append(act_fn)
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = layer_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        
        # Create sequential model
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        """Forward pass through the network."""
        return self.model(x)


class NeuralNetworkModel(BaseModel):
    """PyTorch Neural Network implementation for KIBA prediction."""
    
    def __init__(self, config: KIBAConfig, **kwargs):
        """Initialize the Neural Network model.
        
        Args:
            config: KIBAConfig object with model parameters
            **kwargs: Additional model-specific parameters including:
                - hidden_layers: List of integers defining hidden layer sizes
                - dropout_rate: Dropout rate for regularization
                - learning_rate: Learning rate for optimizer
                - batch_size: Batch size for training
                - activation: Activation function for hidden layers
                - device: Device to use for training ('cpu', 'cuda')
        """
        self.config = config
        self.model = None
        self.history = {"train_loss": [], "val_loss": []}
        self.best_epoch = None
        self.best_val_loss = float('inf')
        self.feature_names = kwargs.get('feature_names', None)
        
        # Neural network parameters
        self.hidden_layers = kwargs.get('hidden_layers', [512, 256, 128, 64])
        self.dropout_rate = kwargs.get('dropout_rate', 0.3)
        self.learning_rate = kwargs.get('learning_rate', 0.001)
        self.batch_size = kwargs.get('batch_size', self.config.batch_size)
        self.activation = kwargs.get('activation', 'relu')
        
        # Set device
        self.device_name = kwargs.get('device', 'cuda' if torch.cuda.is_available() and self.config.gpu_enabled else 'cpu')
        self.device = torch.device(self.device_name)
        
        # Store all parameters for saving/loading
        self.params = {
            'hidden_layers': self.hidden_layers,
            'dropout_rate': self.dropout_rate,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'activation': self.activation,
            'device': self.device_name
        }
        
        logger.info(f"Neural Network initialized: device={self.device_name}, "
                   f"hidden_layers={self.hidden_layers}, dropout={self.dropout_rate}")
    
    def _build_model(self, input_dim: int) -> None:
        """Build the neural network model architecture.
        
        Args:
            input_dim: Number of input features
        """
        # Create the PyTorch model
        self.model = KIBANeuralNetwork(
            input_dim=input_dim,
            hidden_layers=self.hidden_layers,
            dropout_rate=self.dropout_rate,
            activation=self.activation
        ).to(self.device)
        
        # Define optimizer and loss function
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
              **kwargs) -> nn.Module:
        """Train the neural network model.
        
        Args:
            X_train: Training feature matrix
            y_train: Training target vector
            X_val: Optional validation feature matrix
            y_val: Optional validation target vector
            **kwargs: Additional training parameters including:
                - epochs: Number of training epochs
                - patience: Early stopping patience
                - verbose: Verbosity level (0, 1, or 2)
            
        Returns:
            Trained PyTorch model
        """
        logger.info("Training Neural Network model using PyTorch...")
        
        # Process kwargs
        epochs = kwargs.get('epochs', 100)
        patience = kwargs.get('patience', 10)
        verbose = kwargs.get('verbose', 1)
        
        # Reshape y if needed and convert to tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_train_tensor = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32).to(self.device)
        
        # Create validation tensors if provided
        has_validation = X_val is not None and y_val is not None
        if has_validation:
            X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(self.device)
            y_val_tensor = torch.tensor(y_val.reshape(-1, 1), dtype=torch.float32).to(self.device)
        
        # Create datasets and dataloaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        # Build model architecture
        input_dim = X_train.shape[1]
        if self.model is None:
            self._build_model(input_dim)
        
        # Training loop
        start_time = time.time()
        best_val_loss = float('inf')
        best_model_state = None
        no_improve_epochs = 0
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                # Zero the gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                
                # Backward pass and optimization
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item() * batch_X.size(0)
            
            train_loss /= len(train_loader.dataset)
            self.history["train_loss"].append(train_loss)
            
            # Validation phase
            if has_validation:
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(X_val_tensor)
                    val_loss = self.criterion(val_outputs, y_val_tensor).item()
                    self.history["val_loss"].append(val_loss)
                
                # Check for improvement
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = self.model.state_dict().copy()
                    self.best_val_loss = val_loss
                    self.best_epoch = epoch + 1
                    no_improve_epochs = 0
                else:
                    no_improve_epochs += 1
                
                # Early stopping
                if no_improve_epochs >= patience:
                    if verbose > 0:
                        logger.info(f"Early stopping at epoch {epoch+1}")
                    break
                
                # Print progress
                if verbose > 0 and (epoch % 10 == 0 or epoch == epochs - 1):
                    logger.info(f"Epoch {epoch+1}/{epochs} - train_loss: {train_loss:.6f}, val_loss: {val_loss:.6f}")
            else:
                # Print progress without validation
                if verbose > 0 and (epoch % 10 == 0 or epoch == epochs - 1):
                    logger.info(f"Epoch {epoch+1}/{epochs} - train_loss: {train_loss:.6f}")
        
        # Load best model if we have validation data
        if has_validation and best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        training_time = time.time() - start_time
        logger.info(f"Neural Network model trained in {training_time:.2f}s")
        if self.best_epoch and self.best_val_loss:
            logger.info(f"Best epoch: {self.best_epoch}, Best validation loss: {self.best_val_loss:.6f}")
        
        return self.model
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with the trained model.
        
        Args:
            X: Feature matrix to predict on
            
        Returns:
            Array of predictions
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded. Call train() or load() first.")
        
        # Convert to tensor
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        # Make predictions
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_tensor).cpu().numpy()
        
        return predictions.flatten()  # Return 1D array for consistency with XGBoost
    
    def save(self, file_path: str) -> None:
        """Save the model to disk.
        
        Args:
            file_path: Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Save model state dictionary
        model_path = file_path + ".pt"
        torch.save(self.model.state_dict(), model_path)
        
        # Save model architecture and metadata
        metadata = {
            'params': self.params,
            'best_epoch': self.best_epoch,
            'best_val_loss': self.best_val_loss,
            'feature_names': self.feature_names,
            'input_dim': next(self.model.parameters()).shape[1] if len(list(self.model.parameters())) > 0 else None,
            'history': self.history
        }
        
        metadata_path = file_path + ".json"
        with open(metadata_path, 'w') as f:
            # Convert numpy and torch types to Python native types
            def convert_to_serializable(obj):
                if isinstance(obj, (np.integer, np.floating)):
                    return obj.item()
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj
            
            # Use custom serialization
            serializable_metadata = {k: convert_to_serializable(v) for k, v in metadata.items()}
            json.dump(serializable_metadata, f, indent=2)
        
        logger.info(f"PyTorch Neural Network model saved to {model_path} with metadata at {metadata_path}")
    
    def load(self, file_path: str) -> None:
        """Load the model from disk.
        
        Args:
            file_path: Path to load the model from
        """
        model_path = file_path + ".pt"
        metadata_path = file_path + ".json"
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Model metadata not found: {metadata_path}")
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            
        self.params = metadata['params']
        self.best_epoch = metadata.get('best_epoch')
        self.best_val_loss = metadata.get('best_val_loss')
        self.feature_names = metadata.get('feature_names')
        self.history = metadata.get('history', {"train_loss": [], "val_loss": []})
        
        # Restore parameters
        self.hidden_layers = self.params['hidden_layers']
        self.dropout_rate = self.params['dropout_rate']
        self.learning_rate = self.params['learning_rate']
        self.batch_size = self.params['batch_size']
        self.activation = self.params['activation']
        
        # Load device
        self.device_name = self.params.get('device', 'cuda' if torch.cuda.is_available() and self.config.gpu_enabled else 'cpu')
        self.device = torch.device(self.device_name)
        
        # Build model with the same architecture
        input_dim = metadata.get('input_dim')
        if input_dim is None:
            raise ValueError("Input dimension not found in metadata")
            
        self._build_model(input_dim)
        
        # Load model state
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()  # Set to evaluation mode
        
        logger.info(f"PyTorch Neural Network model loaded from {model_path}")
    
    def get_params(self) -> Dict[str, Any]:
        """Get the model parameters.
        
        Returns:
            Dictionary of model parameters
        """
        return self.params
    
    def set_params(self, params: Dict[str, Any]) -> None:
        """Set the model parameters.
        
        Args:
            params: Dictionary of model parameters
        """
        for key, value in params.items():
            if key in self.params:
                self.params[key] = value
                setattr(self, key, value)
        
        # Note: This doesn't rebuild the model - only sets params for next training
        logger.info("Model parameters updated. Model will need to be retrained to apply them.")
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importances if available.
        
        Neural networks don't have native feature importance like tree-based models.
        This implementation returns None as feature importance is not directly available.
        
        Returns:
            None (feature importances not available for neural networks)
        """
        if self.model is None:
            return None
            
        logger.warning("Feature importance not available for neural network models")
        return None