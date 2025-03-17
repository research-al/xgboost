"""XGBoost model implementation for KIBA prediction."""

import os
import time
import logging
import xgboost as xgb
import numpy as np
from typing import Dict, Any, Optional, Union, Tuple

from kiba_model.modeling.models.base import BaseModel
from kiba_model.config import KIBAConfig

logger = logging.getLogger("kiba_model")

class XGBoostModel(BaseModel):
    """XGBoost implementation for KIBA prediction."""
    
    def __init__(self, config: KIBAConfig, **kwargs):
        """Initialize the XGBoost model.
        
        Args:
            config: KIBAConfig object with model parameters
            **kwargs: Additional model-specific parameters
        """
        self.config = config
        self.model = None
        self.best_iteration = None
        self.best_score = None
        self.feature_names = kwargs.get('feature_names', None)
        
        # Default XGBoost parameters from config
        self.params = config.get_xgb_params()
        
        # Update with any additional parameters
        self.params.update({k: v for k, v in kwargs.items() 
                           if k not in ['config', 'feature_names']})
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
              **kwargs) -> xgb.Booster:
        """Train the XGBoost model.
        
        Args:
            X_train: Training feature matrix
            y_train: Training target vector
            X_val: Optional validation feature matrix
            y_val: Optional validation target vector
            **kwargs: Additional training parameters including:
                - num_boost_round: Number of boosting rounds
                - early_stopping_rounds: Early stopping criteria
                - verbose_eval: Verbosity of training output
            
        Returns:
            Trained XGBoost model
        """
        logger.info("Training XGBoost model...")
        
        # Create DMatrix objects for training
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=self.feature_names)
        
        # Create evaluation list
        evals = [(dtrain, 'train')]
        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val, feature_names=self.feature_names)
            evals.append((dval, 'validation'))
        
        # Set training parameters
        num_boost_round = kwargs.get('num_boost_round', 1000)
        early_stopping_rounds = kwargs.get('early_stopping_rounds', 10)
        verbose_eval = kwargs.get('verbose_eval', 25)
        
        # Train model
        start_time = time.time()
        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=num_boost_round,
            evals=evals,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=verbose_eval
        )
        
        # Store best iteration and score
        if hasattr(self.model, 'best_iteration'):
            self.best_iteration = self.model.best_iteration
            self.best_score = self.model.best_score
        
        training_time = time.time() - start_time
        logger.info(f"XGBoost model trained in {training_time:.2f}s")
        if self.best_iteration:
            logger.info(f"Best iteration: {self.best_iteration}, Best score: {self.best_score:.6f}")
        
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
        
        dmatrix = xgb.DMatrix(X, feature_names=self.feature_names)
        return self.model.predict(dmatrix)
    
    def save(self, file_path: str) -> None:
        """Save the model to disk.
        
        Args:
            file_path: Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        self.model.save_model(file_path)
        logger.info(f"XGBoost model saved to {file_path}")
    
    def load(self, file_path: str) -> None:
        """Load the model from disk.
        
        Args:
            file_path: Path to load the model from
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Model file not found: {file_path}")
        
        self.model = xgb.Booster()
        self.model.load_model(file_path)
        logger.info(f"XGBoost model loaded from {file_path}")
    
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
        self.params.update(params)
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importances.
        
        Returns:
            Dictionary mapping feature names/indices to importance values
        """
        if self.model is None:
            return None
        
        try:
            # Get feature importance (default to 'gain')
            importance_type = 'gain'
            feature_importance = self.model.get_score(importance_type=importance_type)
            return feature_importance
        except Exception as e:
            logger.warning(f"Could not get feature importance: {str(e)}")
            return None
