"""Model training for KIBA prediction."""

import time
import logging
import numpy as np
import xgboost as xgb
from typing import Dict, Tuple, List, Optional, Union, Any
from sklearn.model_selection import train_test_split
import pickle

from kiba_model.config import KIBAConfig

logger = logging.getLogger("kiba_model")



class ModelTrainer:
    """Trains and tunes XGBoost models for KIBA prediction.
    
    This class handles data splitting, model training, and hyperparameter tuning.
    """
    
    def __init__(self, config: KIBAConfig):
        """Initialize with configuration.
        
        Args:
            config: KIBAConfig object with model parameters
        """
        self.config = config
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
        
    def split_data(self, X: np.ndarray, y: np.ndarray, 
                  strata_array: Optional[np.ndarray] = None) -> None:
        """Split data into training, validation, and test sets.
        
        Args:
            X: Feature matrix
            y: Target vector
            strata_array: Optional array of stratification labels
        """
        logger.info("Splitting data into train, validation, and test sets...")
        
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
        
    def train_initial_model(self) -> xgb.Booster:
        """Train initial XGBoost model with default parameters.
        
        Returns:
            Trained XGBoost model
            
        Raises:
            ValueError: If data hasn't been split yet
        """
        if self.X_train is None or self.y_train is None:
            error_msg = "Data not split. Call split_data() first."
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        logger.info("Training initial XGBoost model...")
        
        # Create DMatrix objects with error handling
        try:
            dtrain = xgb.DMatrix(self.X_train, label=self.y_train)
            dval = xgb.DMatrix(self.X_val, label=self.y_val)
            logger.info("Successfully created DMatrix objects")
        except xgb.core.XGBoostError as e:
            logger.error(f"Error creating DMatrix: {str(e)}")
            logger.info("Performing additional cleanup and trying again...")
            
            # Clean up data
            X_train_clean = np.nan_to_num(self.X_train, nan=0.0, posinf=0.0, neginf=0.0)
            y_train_clean = np.clip(self.y_train, -1e3, 1e3)
            
            X_val_clean = np.nan_to_num(self.X_val, nan=0.0, posinf=0.0, neginf=0.0)
            y_val_clean = np.clip(self.y_val, -1e3, 1e3)
            
            try:
                dtrain = xgb.DMatrix(X_train_clean, label=y_train_clean)
                dval = xgb.DMatrix(X_val_clean, label=y_val_clean)
                logger.info("Successfully created DMatrix objects after cleanup")
                
                # Update our variables
                self.X_train, self.y_train = X_train_clean, y_train_clean
                self.X_val, self.y_val = X_val_clean, y_val_clean
            except xgb.core.XGBoostError as e2:
                error_msg = f"Still failed after cleanup: {str(e2)}"
                logger.error(error_msg)
                raise ValueError(error_msg)
        
        # Set parameters
        params = self.config.get_xgb_params()
        logger.info(f"XGBoost parameters: {params}")
        
        # Train model with early stopping
        start_time = time.time()
        self.initial_model = xgb.train(
            params,
            dtrain,
            num_boost_round=1000,
            evals=[(dtrain, 'train'), (dval, 'validation')],
            early_stopping_rounds=10,
            verbose_eval=25
        )
        training_time = time.time() - start_time
        
        # Save initial model
        self.initial_model.save_model(str(self.config.initial_model_file))
        logger.info(f"Initial model trained in {training_time:.2f}s, "
                   f"best iteration: {self.initial_model.best_iteration}, "
                   f"best score: {self.initial_model.best_score:.6f}")
        logger.info(f"Initial model saved to {self.config.initial_model_file}")
        
        return self.initial_model
    
    def tune_hyperparameters(self) -> Dict[str, Any]:
        """Tune hyperparameters for XGBoost model.
        
        Returns:
            Dictionary with best parameters
            
        Raises:
            ValueError: If initial model hasn't been trained
        """
        if self.initial_model is None:
            error_msg = "Initial model not trained. Call train_initial_model() first."
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        logger.info("Tuning hyperparameters...")
        
        # Create DMatrix objects
        dtrain = xgb.DMatrix(self.X_train, label=self.y_train)
        dval = xgb.DMatrix(self.X_val, label=self.y_val)
        
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
                    
                    # Train model with early stopping
                    start_time = time.time()
                    model_tuning = xgb.train(
                        params,
                        dtrain,
                        num_boost_round=500,
                        evals=[(dval, 'validation')],
                        early_stopping_rounds=10,
                        verbose_eval=False
                    )
                    tuning_time = time.time() - start_time
                    
                    # Get validation RMSE
                    val_rmse = model_tuning.best_score
                    
                    # Save result
                    tuning_results.append({
                        'max_depth': max_depth,
                        'learning_rate': learning_rate,
                        'subsample': subsample,
                        'val_rmse': val_rmse,
                        'best_iteration': model_tuning.best_iteration,
                        'tuning_time': tuning_time
                    })
                    
                    logger.info(f"  Validation RMSE: {val_rmse:.6f}, "
                                f"best iteration: {model_tuning.best_iteration}, "
                                f"time: {tuning_time:.2f}s")
        
        # Find best parameters
        self.best_params = min(tuning_results, key=lambda x: x['val_rmse'])
        logger.info(f"Best parameters: {self.best_params}")
        
        # Save best parameters
        with open(self.config.best_params_file, 'wb') as f:
            pickle.dump(self.best_params, f)
        logger.info(f"Best parameters saved to {self.config.best_params_file}")
            
        return self.best_params
    
    def train_final_model(self) -> xgb.Booster:
        """Train final model with best parameters on combined training+validation data.
        
        Returns:
            Final trained XGBoost model
            
        Raises:
            ValueError: If best parameters haven't been determined
        """
        if self.best_params is None:
            error_msg = "Best parameters not found. Call tune_hyperparameters() first."
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        logger.info("Training final model with best parameters...")
        
        # Combine train and validation data
        X_train_full = np.vstack([self.X_train, self.X_val])
        y_train_full = np.concatenate([self.y_train, self.y_val])
        
        # Create DMatrix objects
        dtrain_full = xgb.DMatrix(X_train_full, label=y_train_full)
        dtest = xgb.DMatrix(self.X_test, label=self.y_test)
        
        # Set parameters
        final_params = self.config.get_xgb_params()
        final_params['eta'] = self.best_params['learning_rate']
        final_params['max_depth'] = self.best_params['max_depth']
        final_params['subsample'] = self.best_params['subsample']
        
        logger.info(f"Final model parameters: {final_params}")
        logger.info(f"Training with {self.best_params['best_iteration']} boosting rounds")
        
        # Train final model
        start_time = time.time()
        self.final_model = xgb.train(
            final_params,
            dtrain_full,
            num_boost_round=self.best_params['best_iteration'],
            evals=[(dtrain_full, 'train'), (dtest, 'test')],
            verbose_eval=25
        )
        training_time = time.time() - start_time
        
        # Save final model
        self.final_model.save_model(str(self.config.final_model_file))
        logger.info(f"Final model trained in {training_time:.2f}s")
        logger.info(f"Final model saved to {self.config.final_model_file}")
        
        return self.final_model
