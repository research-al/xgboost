"""Neural network model implementation for KIBA prediction."""

import os
import time
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, r2_score
import pickle
from typing import Dict, Tuple, List, Optional, Union, Any

from kiba_model.modeling.models.base import BaseModel
from kiba_model.config import KIBAConfig

logger = logging.getLogger("kiba_model")

class KIBANeuralNetwork(nn.Module):
    """Neural network model for KIBA prediction."""
    
    def __init__(self, input_dim: int, hidden_layers: List[int] = [512, 256, 128, 64], 
                 dropout: float = 0.3):
        """Initialize neural network with specified architecture.
        
        Args:
            input_dim: Dimension of input features
            hidden_layers: List of hidden layer sizes
            dropout: Dropout probability
        """
        super(KIBANeuralNetwork, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout
        
        # Build layers
        layers = []
        prev_size = input_dim
        
        for i, size in enumerate(hidden_layers):
            # Add linear layer
            layers.append(nn.Linear(prev_size, size))
            
            # Add batch normalization
            layers.append(nn.BatchNorm1d(size))
            
            # Add activation
            layers.append(nn.ReLU())
            
            # Add dropout for regularization (except last layer)
            if i < len(hidden_layers) - 1 or dropout > 0:
                layers.append(nn.Dropout(dropout))
                
            prev_size = size
        
        # Output layer
        layers.append(nn.Linear(prev_size, 1))
        
        # Create sequential model
        self.model = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor (predictions)
        """
        return self.model(x).squeeze()


class NeuralNetTrainer:
    """Class for training and tuning neural network models for KIBA prediction."""
    
    def __init__(self, config):
        """Initialize with configuration.
        
        Args:
            config: KIBAConfig object with model parameters
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() and config.gpu_enabled else "cpu")
        self.model = None
        self.best_model_state = None
        self.history = {"train_loss": [], "val_loss": []}
        
        logger.info(f"Neural Network initialized: device={self.device}, "
                    f"hidden_layers=[512, 256, 128, 64], dropout=0.3")
        
    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        """Convert numpy array to PyTorch tensor.
        
        Args:
            data: NumPy array
            
        Returns:
            PyTorch tensor
        """
        return torch.tensor(data, dtype=torch.float32, device=self.device)
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: np.ndarray, y_val: np.ndarray,
              epochs: int = 100, batch_size: int = 32, 
              learning_rate: float = 0.001) -> Dict[str, List[float]]:
        """Train neural network model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            
        Returns:
            Dictionary with training history
        """
        # Convert data to tensors
        X_train_tensor = self._to_tensor(X_train)
        y_train_tensor = self._to_tensor(y_train)
        X_val_tensor = self._to_tensor(X_val)
        y_val_tensor = self._to_tensor(y_val)
        
        # Create datasets and dataloaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Initialize model
        input_dim = X_train.shape[1]
        self.model = KIBANeuralNetwork(input_dim).to(self.device)
        
        # Loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
        
        # Early stopping
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        logger.info(f"Starting neural network training for {epochs} epochs")
        logger.info(f"Learning rate: {learning_rate}, batch size: {batch_size}")
        
        # Training loop
        start_time = time.time()
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0
            
            for inputs, targets in train_loader:
                # Forward pass
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            # Calculate average loss
            avg_train_loss = total_loss / len(train_loader)
            self.history["train_loss"].append(avg_train_loss)
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor).item()
                self.history["val_loss"].append(val_loss)
            
            # Update learning rate
            scheduler.step(val_loss)
            
            # Log progress every 10 epochs
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.6f}, "
                           f"Val Loss: {val_loss:.6f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.best_model_state = self.model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f}s, best validation loss: {best_val_loss:.6f}")
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        return self.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with model.
        
        Args:
            X: Features
            
        Returns:
            Predictions
        """
        if self.model is None:
            logger.error("Cannot predict: no model has been trained or loaded")
            return None
            
        # Convert to tensor
        X_tensor = self._to_tensor(X)
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_tensor).cpu().numpy()
            
        return predictions


# Add the BaseModel implementation for neural networks
class NeuralNetworkModel(BaseModel):
    """Neural network implementation for KIBA prediction."""
    
    def __init__(self, config: KIBAConfig, **kwargs):
        """Initialize the neural network model.
        
        Args:
            config: KIBAConfig object with model parameters
            **kwargs: Additional model-specific parameters
        """
        self.config = config
        self.model = None
        self.trainer = NeuralNetTrainer(config)
        self.device = torch.device("cuda" if torch.cuda.is_available() and config.gpu_enabled else "cpu")
        self.feature_names = kwargs.get('feature_names', None)
        self.input_dim = kwargs.get('input_dim', None)
        self.hidden_layers = kwargs.get('hidden_layers', [512, 256, 128, 64])
        self.dropout_rate = kwargs.get('dropout_rate', 0.3)
        self.learning_rate = kwargs.get('learning_rate', 0.001)
    
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
                - verbose: Verbosity of training output
            
        Returns:
            Trained neural network model
        """
        logger.info("Training Neural Network model...")
        
        # Get input dimension if not provided
        if self.input_dim is None:
            self.input_dim = X_train.shape[1]
        
        # Check that validation data is provided
        if X_val is None or y_val is None:
            logger.warning("Validation data not provided, using 10% of training data")
            # Split training data into train and validation
            from sklearn.model_selection import train_test_split
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.1, random_state=self.config.random_state
            )
        
        # Train model
        history = self.trainer.train(
            X_train, 
            y_train, 
            X_val, 
            y_val,
            epochs=kwargs.get('epochs', 100),
            batch_size=kwargs.get('batch_size', 32),
            learning_rate=self.learning_rate
        )
        
        # Get the trained model
        self.model = self.trainer.model
        
        return self.model
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with the trained model.
        
        Args:
            X: Feature matrix to predict on
            
        Returns:
            Array of predictions
        """
        if self.model is None:
            self.model = self.trainer.model
            
        if self.model is None:
            raise ValueError("Model not trained or loaded. Call train() or load() first.")
        
        # Convert to tensor
        X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        
        # Make predictions
        with torch.no_grad():
            self.model.eval()
            predictions = self.model(X_tensor).cpu().numpy()
            
        return predictions
    
    def save(self, file_path: str) -> None:
        """Save the model to disk.
        
        Args:
            file_path: Path to save the model
        """
        if self.model is None:
            self.model = self.trainer.model
            
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        # Make sure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Adjust file path to use .pt extension for PyTorch models
        file_path = str(file_path).replace('.json', '.pt')
        
        # Save model state dict
        torch.save(self.model.state_dict(), file_path)
        logger.info(f"Neural network model saved to {file_path}")
    
    def load(self, file_path: str) -> None:
        """Load the model from disk.
        
        Args:
            file_path: Path to load the model from
        """
        # Adjust file path to use .pt extension
        file_path = str(file_path).replace('.json', '.pt')
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Model file not found: {file_path}")
        
        # Create model if input_dim is known
        if self.input_dim is not None:
            self.model = KIBANeuralNetwork(
                self.input_dim, 
                hidden_layers=self.hidden_layers, 
                dropout=self.dropout_rate
            ).to(self.device)
            
            # Load state dict
            self.model.load_state_dict(torch.load(file_path, map_location=self.device))
            self.model.eval()
            logger.info(f"Neural network model loaded from {file_path}")
        else:
            logger.error("Cannot load model: input_dim is not known")
            raise ValueError("Input dimension must be specified before loading model")
    
    def get_params(self) -> Dict[str, Any]:
        """Get the model parameters.
        
        Returns:
            Dictionary of model parameters
        """
        return {
            'input_dim': self.input_dim,
            'hidden_layers': self.hidden_layers,
            'dropout_rate': self.dropout_rate,
            'learning_rate': self.learning_rate
        }
    
    def set_params(self, params: Dict[str, Any]) -> None:
        """Set the model parameters.
        
        Args:
            params: Dictionary of model parameters
        """
        if 'input_dim' in params:
            self.input_dim = params['input_dim']
        if 'hidden_layers' in params:
            self.hidden_layers = params['hidden_layers']
        if 'dropout_rate' in params:
            self.dropout_rate = params['dropout_rate']
        if 'learning_rate' in params:
            self.learning_rate = params['learning_rate']
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importances.
        
        Returns:
            Dictionary mapping feature names/indices to importance values
            
        Note:
            Neural networks don't have built-in feature importance like tree-based models.
            This method returns None or could be implemented with techniques like permutation importance.
        """
        logger.warning("Feature importance not directly available for neural network models")
        return None