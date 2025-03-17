"""Model training for KIBA prediction."""

import time
import logging
import numpy as np
import pickle
import os
from typing import Dict, Tuple, List, Optional, Union, Any

from kiba_model.config import KIBAConfig
from kiba_model.modeling.models.base import BaseModel, ModelFactory
from kiba_model.modeling.models.xgboost_model import XGBoostModel
from kiba_model.modeling.models.neural_network_model import NeuralNetTrainer
from sklearn.model_selection import train_test_split

logger = logging.getLogger("kiba_model")

class ModelTrainer:
    """Trains and tunes models for KIBA prediction.
    
    This class handles data splitting, model training, and hyperparameter tuning.
    """
    
    def __init__(self, config: KIBAConfig, model_type: str = 'xgboost'):
        """Initialize with configuration.
        
        Args:
            config: KIBAConfig object with model parameters
            model_type: Type of model to use ('xgboost', 'neural_network', etc.)
        """
        self.config = config
        self.model_type = model_type
        self.model = None
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.strata_train = None
        self.strata_val = None
        self.strata_test = None
        self.initial_model = None
        self.final_model = None
        self.best_params = None
        self.feature_names = None
        
    def _initialize_model(self):
        """Initialize the model based on the specified model type."""
        if self.model_type == 'xgboost':
            self.model = XGBoostModel(self.config)
        elif self.model_type in ['neural_network', 'nn']:
            self.model = NeuralNetTrainer(self.config)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def split_data(self, X: np.ndarray, y: np.ndarray, 
                  strata_array: Optional[np.ndarray] = None,
                  feature_names: Optional[List[str]] = None) -> None:
        """Split data into training, validation, and test sets.
        
        Args:
            X: Feature matrix
            y: Target vector
            strata_array: Optional array of stratification labels
            feature_names: Optional list of feature names
        """
        logger.info("Splitting data into train, validation, and test sets...")
        
        # Store feature names if provided
        self.feature_names = feature_names
        
        # Use stratification if available and enabled
        if strata_array is not None and self.config.use_stratification:
            logger.info("Using stratified sampling")
            
            # First split: train+val vs test
            X_train_val, self.X_test, y_train_val, self.y_test, strata_train_val, self.strata_test = train_test_split(
                X, y, strata_array, 
                test_size=self.config.test_size, 
                random_state=self.config.random_state,
                stratify=strata_array
            )
            
            # Second split: train vs val
            self.X_train, self.X_val, self.y_train, self.y_val, self.strata_train, self.strata_val = train_test_split(
                X_train_val, y_train_val, strata_train_val,
                test_size=self.config.val_size,
                random_state=self.config.random_state,
                stratify=strata_train_val
            )
            
            # Verify strata distribution
            if logger.isEnabledFor(logging.DEBUG):
                for split_name, strata in [
                    ("Train", self.strata_train), 
                    ("Validation", self.strata_val), 
                    ("Test", self.strata_test)
                ]:
                    import pandas as pd
                    strata_counts = pd.Series(strata).value_counts(normalize=True) * 100
                    logger.debug(f"{split_name} set strata distribution (%):\n{strata_counts}")
        else:
            # Regular splitting without stratification
            if strata_array is not None and not self.config.use_stratification:
                logger.info("Stratification data available but not used (disabled in config)")
            else:
                logger.info("Using random sampling (no stratification data available)")
            
            # First split: train+val vs test
            X_train_val, self.X_test, y_train_val, self.y_test = train_test_split(
                X, y, 
                test_size=self.config.test_size, 
                random_state=self.config.random_state
            )
            
            # Second split: train vs val
            self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
                X_train_val, y_train_val,
                test_size=self.config.val_size,
                random_state=self.config.random_state
            )
            
            self.strata_train = None
            self.strata_val = None
            self.strata_test = None
        
        logger.info(f"Training set: {self.X_train.shape[0]} samples")
        logger.info(f"Validation set: {self.X_val.shape[0]} samples")
        logger.info(f"Test set: {self.X_test.shape[0]} samples")
    
    def _clean_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Clean data by removing NaN, Inf, and other problematic values.
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            Tuple of cleaned (X, y)
        """
        # Fix NaN and Inf values in features
        X_clean = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Fix problematic values in targets
        y_clean = np.clip(y, -1e3, 1e3)
        
        # Check if any values are still problematic
        if np.isnan(X_clean).any() or np.isinf(X_clean).any() or np.isnan(y_clean).any() or np.isinf(y_clean).any():
            logger.warning("There are still NaN or Inf values after cleaning. Results may be unpredictable.")
        
        return X_clean, y_clean
    
    def train_initial_model(self) -> BaseModel:
        """Train initial model with default parameters.
        
        Returns:
            Trained model
            
        Raises:
            ValueError: If data hasn't been split yet
        """
        if self.X_train is None or self.y_train is None:
            error_msg = "Data not split. Call split_data() first."
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        logger.info(f"Training initial {self.model_type} model...")
        
        # Clean data if needed
        X_train_clean, y_train_clean = self._clean_data(self.X_train, self.y_train)
        X_val_clean, y_val_clean = self._clean_data(self.X_val, self.y_val)
        
        # Initialize model
        self._initialize_model()
        self.initial_model = self.model

        
        # Train model
        start_time = time.time()
        
        # Set model-specific training parameters
        train_kwargs = {}
        if self.model_type.lower() == 'xgboost':
            train_kwargs = {
                'num_boost_round': 1000,
                'early_stopping_rounds': 10,
                'verbose_eval': 25
            }
        elif self.model_type.lower() in ['neural_network', 'nn']:
            train_kwargs = {
                'epochs': 100,
                'patience': 10,
                'verbose': 1
            }
        
        self.initial_model.train(
            X_train_clean, 
            y_train_clean, 
            X_val_clean, 
            y_val_clean,
            **train_kwargs
        )
        
        training_time = time.time() - start_time
        
        # Save initial model
        initial_model_file = self._get_model_file_path(is_initial=True)
        self.initial_model.save(initial_model_file)
        
        logger.info(f"Initial model trained in {training_time:.2f}s")
        logger.info(f"Initial model saved to {initial_model_file}")
        
        return self.initial_model
    
    def _get_model_file_path(self, is_initial: bool = False, hyperparams: bool = False) -> str:
        """Get file path for model saving/loading.
        
        Args:
            is_initial: Whether this is the initial model or final model
            hyperparams: Whether this is the hyperparameter file
        
        Returns:
            File path string
        """
        # Create directory if it doesn't exist
        os.makedirs(self.config.models_dir, exist_ok=True)
        
        # Base name depends on model type and transformation
        transform_suffix = "log10" if self.config.use_log10_transform else "ln"
        model_prefix = "initial" if is_initial else "final"
        model_suffix = f"{self.model_type}_{transform_suffix}"
        
        if hyperparams:
            return os.path.join(self.config.models_dir, f"best_params_{self.model_type}_{transform_suffix}.pkl")
        else:
            return os.path.join(self.config.models_dir, f"{model_prefix}_model_{model_suffix}")
    
    def tune_hyperparameters(self) -> Dict[str, Any]:
        """Tune hyperparameters for model.
        
        Returns:
            Dictionary with best parameters
            
        Raises:
            ValueError: If initial model hasn't been trained
        """
        if self.initial_model is None:
            error_msg = "Initial model not trained. Call train_initial_model() first."
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        logger.info(f"Tuning hyperparameters for {self.model_type} model...")
        
        # Clean data if needed
        X_train_clean, y_train_clean = self._clean_data(self.X_train, self.y_train)
        X_val_clean, y_val_clean = self._clean_data(self.X_val, self.y_val)
        
        # Model-specific hyperparameter tuning
        if self.model_type.lower() == 'xgboost':
            self.best_params = self._tune_xgboost_hyperparameters(X_train_clean, y_train_clean, X_val_clean, y_val_clean)
        elif self.model_type.lower() in ['neural_network', 'nn']:
            self.best_params = self._tune_nn_hyperparameters(X_train_clean, y_train_clean, X_val_clean, y_val_clean)
        else:
            logger.warning(f"Hyperparameter tuning not implemented for model type: {self.model_type}")
            # Use default parameters
            self.best_params = self.initial_model.get_params()
        
        # Save best parameters
        best_params_file = self._get_model_file_path(hyperparams=True)
        with open(best_params_file, 'wb') as f:
            pickle.dump(self.best_params, f)
        logger.info(f"Best parameters saved to {best_params_file}")
            
        return self.best_params
    
    def _tune_xgboost_hyperparameters(self, X_train, y_train, X_val, y_val) -> Dict[str, Any]:
        """Tune hyperparameters for XGBoost model.
        
        Args:
            X_train: Training feature matrix
            y_train: Training target vector
            X_val: Validation feature matrix
            y_val: Validation target vector
            
        Returns:
            Dictionary with best parameters
        """
        import xgboost as xgb
        
        # Define parameter grid for simplified tuning
        param_grid = {
            'max_depth': [3, 6, 8],
            'learning_rate': [0.01, 0.1],
            'subsample': [0.7, 0.9]
        }
        
        # Try key parameter combinations
        tuning_results = []
        
        for max_depth in param_grid['max_depth']:
            for learning_rate in param_grid['learning_rate']:
                for subsample in param_grid['subsample']:
                    logger.info(f"Testing: max_depth={max_depth}, "
                                f"learning_rate={learning_rate}, subsample={subsample}")
                    
                    # Set parameters
                    params = self.config.get_xgb_params()
                    params['eta'] = learning_rate
                    params['max_depth'] = max_depth
                    params['subsample'] = subsample
                    
                    # Create model
                    model = ModelFactory.create_model('xgboost', self.config, **params)
                    
                    # Train model
                    start_time = time.time()
                    model.train(
                        X_train, 
                        y_train, 
                        X_val, 
                        y_val,
                        num_boost_round=500,
                        early_stopping_rounds=10,
                        verbose_eval=False
                    )
                    
                    # Evaluate on validation set
                    y_pred = model.predict(X_val)
                    val_rmse = np.sqrt(np.mean((y_val - y_pred) ** 2))
                    
                    tuning_time = time.time() - start_time
                    
                    # Save result
                    tuning_results.append({
                        'max_depth': max_depth,
                        'learning_rate': learning_rate,
                        'subsample': subsample,
                        'val_rmse': val_rmse,
                        'tuning_time': tuning_time
                    })
                    
                    logger.info(f"  Validation RMSE: {val_rmse:.6f}, "
                                f"time: {tuning_time:.2f}s")
        
        # Find best parameters
        best_params = min(tuning_results, key=lambda x: x['val_rmse'])
        logger.info(f"Best XGBoost parameters: {best_params}")
        
        # Convert to proper format for model
        xgb_params = self.config.get_xgb_params()
        xgb_params['eta'] = best_params['learning_rate']
        xgb_params['max_depth'] = best_params['max_depth']
        xgb_params['subsample'] = best_params['subsample']
        
        return {
            'model_params': xgb_params,
            'best_iteration': 500,  # Placeholder, real value would come from early stopping
            'val_rmse': best_params['val_rmse']
        }
    
    def _tune_nn_hyperparameters(self, X_train, y_train, X_val, y_val) -> Dict[str, Any]:
        """Tune hyperparameters for Neural Network model.
        
        Args:
            X_train: Training feature matrix
            y_train: Training target vector
            X_val: Validation feature matrix
            y_val: Validation target vector
            
        Returns:
            Dictionary with best parameters
        """
        # Define parameter grid
        param_grid = {
            'hidden_layers': [
                [512, 256, 128, 64],
                [256, 128, 64],
                [128, 64, 32]
            ],
            'dropout_rate': [0.2, 0.3, 0.4],
            'learning_rate': [0.01, 0.001]
        }
        
        # Try key parameter combinations
        tuning_results = []
        
        for hidden_layers in param_grid['hidden_layers']:
            for dropout_rate in param_grid['dropout_rate']:
                for learning_rate in param_grid['learning_rate']:
                    logger.info(f"Testing: hidden_layers={hidden_layers}, "
                                f"dropout_rate={dropout_rate}, learning_rate={learning_rate}")
                    
                    # Create model
                    model = ModelFactory.create_model(
                        'neural_network', 
                        self.config,
                        hidden_layers=hidden_layers,
                        dropout_rate=dropout_rate,
                        learning_rate=learning_rate,
                        feature_names=self.feature_names
                    )
                    
                    # Train model
                    start_time = time.time()
                    model.train(
                        X_train, 
                        y_train, 
                        X_val, 
                        y_val,
                        epochs=50,  # Reduced for tuning
                        patience=5,
                        verbose=0
                    )
                    
                    # Evaluate on validation set
                    y_pred = model.predict(X_val)
                    val_rmse = np.sqrt(np.mean((y_val - y_pred) ** 2))
                    
                    tuning_time = time.time() - start_time
                    
                    # Save result
                    tuning_results.append({
                        'hidden_layers': hidden_layers,
                        'dropout_rate': dropout_rate,
                        'learning_rate': learning_rate,
                        'val_rmse': val_rmse,
                        'tuning_time': tuning_time
                    })
                    
                    logger.info(f"  Validation RMSE: {val_rmse:.6f}, "
                                f"time: {tuning_time:.2f}s")
        
        # Find best parameters
        best_params = min(tuning_results, key=lambda x: x['val_rmse'])
        logger.info(f"Best Neural Network parameters: {best_params}")
        
        # Convert to proper format
        nn_params = {
            'hidden_layers': best_params['hidden_layers'],
            'dropout_rate': best_params['dropout_rate'],
            'learning_rate': best_params['learning_rate'],
            'batch_size': self.config.batch_size,
            'activation': 'relu'  # Default
        }
        
        return {
            'model_params': nn_params,
            'val_rmse': best_params['val_rmse']
        }
    
    def train_final_model(self) -> BaseModel:
        """Train final model with best parameters on combined training+validation data.
        
        Returns:
            Final trained model
            
        Raises:
            ValueError: If best parameters haven't been determined
        """
        if self.best_params is None:
            error_msg = "Best parameters not found. Call tune_hyperparameters() first."
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        logger.info(f"Training final {self.model_type} model with best parameters...")
        
        # Combine train and validation data
        X_train_full = np.vstack([self.X_train, self.X_val])
        y_train_full = np.concatenate([self.y_train, self.y_val])
        
        # Clean data
        X_train_full_clean, y_train_full_clean = self._clean_data(X_train_full, y_train_full)
        X_test_clean, y_test_clean = self._clean_data(self.X_test, self.y_test)
        
        # Initialize model
        self._initialize_model()
        
        # Set training parameters based on model type
        train_kwargs = {}
        if self.model_type.lower() == 'xgboost':
            # Use best iteration from tuning if available
            num_boost_round = self.best_params.get('best_iteration', 500)
            train_kwargs = {
                'num_boost_round': num_boost_round,
                'verbose_eval': 25
            }
        elif self.model_type.lower() in ['neural_network', 'nn']:
            train_kwargs = {
                'epochs': 100,
                'patience': 10,
                'verbose': 1
            }
        
        # Train final model
        start_time = time.time()
        self.final_model.train(
            X_train_full_clean, 
            y_train_full_clean,
            X_test_clean,  # Use test set as validation
            y_test_clean,
            **train_kwargs
        )
        
        training_time = time.time() - start_time
        
        # Save final model
        final_model_file = self._get_model_file_path(is_initial=False)
        self.final_model.save(final_model_file)
        
        logger.info(f"Final model trained in {training_time:.2f}s")
        logger.info(f"Final model saved to {final_model_file}")
        
        return self.final_model
