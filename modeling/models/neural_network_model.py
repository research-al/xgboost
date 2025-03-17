"""Neural network model for KIBA prediction."""

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
    
    def tune_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray, 
                            X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        """Tune hyperparameters for neural network model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            
        Returns:
            Dictionary with best parameters
        """
        logger.info("Tuning neural network hyperparameters...")
        
        # Define parameter grid
        param_grid = {
            'learning_rate': [0.001, 0.0005],
            'batch_size': [16, 32],
            'dropout': [0.2, 0.3]
        }
        
        best_val_loss = float('inf')
        best_params = None
        
        # Try different parameter combinations
        for lr in param_grid['learning_rate']:
            for batch_size in param_grid['batch_size']:
                for dropout in param_grid['dropout']:
                    logger.info(f"Testing: learning_rate={lr}, batch_size={batch_size}, dropout={dropout}")
                    
                    # Initialize model
                    input_dim = X_train.shape[1]
                    self.model = KIBANeuralNetwork(input_dim, dropout=dropout).to(self.device)
                    
                    # Train model
                    history = self.train(X_train, y_train, X_val, y_val, 
                                         epochs=50, batch_size=batch_size, learning_rate=lr)
                    
                    # Get best validation loss
                    val_loss = min(history['val_loss'])
                    
                    logger.info(f"  Validation loss: {val_loss:.6f}")
                    
                    # Update best parameters
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_params = {
                            'learning_rate': lr,
                            'batch_size': batch_size,
                            'dropout': dropout,
                            'val_loss': val_loss
                        }
        
        logger.info(f"Best parameters: {best_params}")
        return best_params
    
    def train_final_model(self, X_train: np.ndarray, y_train: np.ndarray, 
                          X_val: np.ndarray, y_val: np.ndarray,
                          best_params: Dict[str, Any], epochs: int = 200) -> nn.Module:
        """Train final model with best parameters.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            best_params: Best hyperparameters
            epochs: Number of training epochs
            
        Returns:
            Trained neural network model
        """
        logger.info("Training final neural network model with best parameters...")
        
        # Combine train and validation data
        X_train_full = np.vstack([X_train, X_val])
        y_train_full = np.concatenate([y_train, y_val])
        
        # Convert data to tensors
        X_train_tensor = self._to_tensor(X_train_full)
        y_train_tensor = self._to_tensor(y_train_full)
        
        # Create dataset and dataloader
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=best_params['batch_size'], 
            shuffle=True
        )
        
        # Initialize model with best parameters
        input_dim = X_train_full.shape[1]
        self.model = KIBANeuralNetwork(input_dim, dropout=best_params['dropout']).to(self.device)
        
        # Loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=best_params['learning_rate'])
        
        # Training loop
        logger.info(f"Training final model for {epochs} epochs with best parameters")
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
            
            # Log progress every 20 epochs
            if (epoch + 1) % 20 == 0:
                avg_loss = total_loss / len(train_loader)
                logger.info(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}")
        
        training_time = time.time() - start_time
        logger.info(f"Final model trained in {training_time:.2f}s")
        
        return self.model
    
    def save_model(self, model_path: str) -> None:
        """Save model to file.
        
        Args:
            model_path: Path to save model
        """
        if self.model is None:
            logger.error("Cannot save model: no model has been trained")
            return
            
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            # Save model state dict
            torch.save(self.model.state_dict(), model_path)
            logger.info(f"Model saved to {model_path}")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
    
    def load_model(self, model_path: str, input_dim: int) -> None:
        """Load model from file.
        
        Args:
            model_path: Path to model file
            input_dim: Input dimension for model
        """
        try:
            # Initialize model
            self.model = KIBANeuralNetwork(input_dim).to(self.device)
            
            # Load model state dict
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            
            logger.info(f"Model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            self.model = None
    
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