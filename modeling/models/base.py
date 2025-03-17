"""Base model interfaces for KIBA prediction."""

import os
import logging
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, Tuple

logger = logging.getLogger("kiba_model")

class BaseModel(ABC):
    """Abstract base class for all KIBA prediction models.
    
    This class defines the interface that all model implementations must follow.
    """
    
    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
              **kwargs) -> Any:
        """Train the model on the given data.
        
        Args:
            X_train: Training feature matrix
            y_train: Training target vector
            X_val: Optional validation feature matrix
            y_val: Optional validation target vector
            **kwargs: Additional model-specific training parameters
            
        Returns:
            Trained model object
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with the trained model.
        
        Args:
            X: Feature matrix to predict on
            
        Returns:
            Array of predictions
        """
        pass
    
    @abstractmethod
    def save(self, file_path: str) -> None:
        """Save the model to disk.
        
        Args:
            file_path: Path to save the model
        """
        pass
    
    @abstractmethod
    def load(self, file_path: str) -> None:
        """Load the model from disk.
        
        Args:
            file_path: Path to load the model from
        """
        pass
    
    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """Get the model parameters.
        
        Returns:
            Dictionary of model parameters
        """
        pass
    
    @abstractmethod
    def set_params(self, params: Dict[str, Any]) -> None:
        """Set the model parameters.
        
        Args:
            params: Dictionary of model parameters
        """
        pass
    
    @abstractmethod
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importances if available.
        
        Returns:
            Dictionary mapping feature names/indices to importance values,
            or None if feature importances are not available
        """
        pass


class ModelFactory:
    """Factory class for creating model instances."""
    
    @staticmethod
    def create_model(model_type: str, config: Any = None, **kwargs) -> BaseModel:
        """Create and return a model instance of the specified type."""
        from kiba_model.modeling.models.xgboost_model import XGBoostModel
        from kiba_model.modeling.models.neural_network_model import NeuralNetworkModel
        
        if model_type.lower() == 'xgboost':
            return XGBoostModel(config, **kwargs)
        elif model_type.lower() in ['neural_network', 'nn']:
            return NeuralNetworkModel(config, **kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")