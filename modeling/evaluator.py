"""Model evaluation for KIBA prediction."""

import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from typing import Dict, Tuple, List, Optional, Union, Any
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import torch

from kiba_model.config import KIBAConfig
from kiba_model.modeling.models.base import BaseModel
from kiba_model.modeling.models.neural_network_model import NeuralNetTrainer

logger = logging.getLogger("kiba_model")

class ModelEvaluator:
    """Evaluates model performance and generates visualizations.
    
    This class handles model evaluation, metrics calculation, and visualization generation.
    """
    
    def __init__(self, config: KIBAConfig):
        """Initialize with configuration.
        
        Args:
            config: KIBAConfig object with paths and settings
        """
        self.config = config
        self.metrics = {
            'log_scale': {},
            'original_scale': {}
        }
        self.model_type = getattr(config, 'model_type', 'xgboost')
        self.device = torch.device("cuda" if torch.cuda.is_available() and config.gpu_enabled else "cpu")
    
    def evaluate_model(self, model: Union[xgb.Booster, torch.nn.Module], 
                       X_test: np.ndarray, y_test: np.ndarray, 
                       strata_test: Optional[np.ndarray] = None) -> Dict[str, Dict[str, float]]:
        """Evaluate model performance on test data.
        
        Args:
            model: Trained model (any BaseModel implementation)
            X_test: Test feature matrix
            y_test: Test target vector
            strata_test: Optional test strata for subgroup evaluation
            
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info("Evaluating model on test set...")
        
        # Make predictions based on model type
        if self.model_type == 'neural_network' or isinstance(model, NeuralNetTrainer):
            # Use the predict method from NeuralNetTrainer
            y_pred = model.predict(X_test)
        elif hasattr(model, 'predict'):
            # Standard predict method for most model types
            y_pred = model.predict(X_test)
        else:
            raise ValueError(f"Unknown model type: {type(model)}")
        
        # Convert from log scale back to original scale
        if self.config.use_log10_transform:
            y_test_orig = (10**y_test) - 1
            y_pred_orig = (10**y_pred) - 1
            log_label = "log10"
        else:
            y_test_orig = np.exp(y_test) - 1e-6
            y_pred_orig = np.exp(y_pred) - 1e-6
            log_label = "ln"
        
        # Calculate metrics on log scale
        rmse_log = np.sqrt(mean_squared_error(y_test, y_pred))
        mae_log = mean_absolute_error(y_test, y_pred)
        r2_log = r2_score(y_test, y_pred)
        
        # Calculate metrics on original scale
        rmse_orig = np.sqrt(mean_squared_error(y_test_orig, y_pred_orig))
        mae_orig = mean_absolute_error(y_test_orig, y_pred_orig)
        r2_orig = r2_score(y_test_orig, y_pred_orig)
        
        # Store metrics
        self.metrics['log_scale'] = {
            'rmse': rmse_log,
            'mae': mae_log,
            'r2': r2_log
        }
        
        self.metrics['original_scale'] = {
            'rmse': rmse_orig,
            'mae': mae_orig,
            'r2': r2_orig
        }
        
        # Log metrics
        logger.info(f"{log_label} scale metrics:")
        logger.info(f"  RMSE: {rmse_log:.4f}")
        logger.info(f"  MAE: {mae_log:.4f}")
        logger.info(f"  R²: {r2_log:.4f}")
        
        logger.info(f"Original scale metrics:")
        logger.info(f"  RMSE: {rmse_orig:.4f}")
        logger.info(f"  MAE: {mae_orig:.4f}")
        logger.info(f"  R²: {r2_orig:.4f}")
        
        # Evaluate on subgroups if strata information is available
        if strata_test is not None:
            logger.info("Evaluating on different data subgroups:")
            subgroup_metrics = {}
            
            # Create masks for different groups
            exp_mask = np.array([s.startswith('exp') for s in strata_test])
            est_mask = np.array([s.startswith('est') for s in strata_test])
            low_mask = np.array([s.endswith('low') for s in strata_test])
            med_mask = np.array([s.endswith('med') for s in strata_test])
            high_mask = np.array([s.endswith('high') for s in strata_test])
            
            # Evaluate on each group
            for name, mask in [
                ("Experimental", exp_mask),
                ("Estimated", est_mask),
                ("Low values (<25)", low_mask),
                ("Medium values (25-100)", med_mask),
                ("High values (>100)", high_mask)
            ]:
                if np.sum(mask) > 0:
                    # Log scale metrics
                    subset_rmse_log = np.sqrt(mean_squared_error(y_test[mask], y_pred[mask]))
                    subset_mae_log = mean_absolute_error(y_test[mask], y_pred[mask])
                    subset_r2_log = r2_score(y_test[mask], y_pred[mask])
                    
                    # Original scale
                    y_test_orig_subset = y_test_orig[mask]
                    y_pred_orig_subset = y_pred_orig[mask]
                    subset_rmse_orig = np.sqrt(mean_squared_error(y_test_orig_subset, y_pred_orig_subset))
                    subset_mae_orig = mean_absolute_error(y_test_orig_subset, y_pred_orig_subset)
                    subset_r2_orig = r2_score(y_test_orig_subset, y_pred_orig_subset)
                    
                    # Store subgroup metrics
                    subgroup_metrics[name] = {
                        'log_scale': {
                            'rmse': subset_rmse_log,
                            'mae': subset_mae_log,
                            'r2': subset_r2_log,
                            'count': int(np.sum(mask))
                        },
                        'original_scale': {
                            'rmse': subset_rmse_orig,
                            'mae': subset_mae_orig,
                            'r2': subset_r2_orig,
                            'count': int(np.sum(mask))
                        }
                    }
                    
                    # Log subgroup metrics
                    logger.info(f"\n{name} subset ({np.sum(mask)} samples):")
                    logger.info(f"  {log_label} scale - RMSE: {subset_rmse_log:.4f}, MAE: {subset_mae_log:.4f}, "
                                f"R²: {subset_r2_log:.4f}")
                    logger.info(f"  Original scale - RMSE: {subset_rmse_orig:.4f}, MAE: {subset_mae_orig:.4f}, "
                                f"R²: {subset_r2_orig:.4f}")
            
            # Add subgroup metrics to main metrics
            self.metrics['subgroups'] = subgroup_metrics
        
        # Save metrics
        with open(self.config.metrics_file, 'wb') as f:
            pickle.dump(self.metrics, f)
            
        logger.info(f"Metrics saved to {self.config.metrics_file}")
        
        return self.metrics
    
    def generate_visualizations(self, model: Union[xgb.Booster, torch.nn.Module], 
                                X_test: np.ndarray, y_test: np.ndarray) -> None:
        """Generate visualizations for model evaluation.
        
        Args:
            model: Trained model (any BaseModel implementation)
            X_test: Test feature matrix
            y_test: Test target vector
        """
        logger.info("Generating visualizations...")
        
            # Make predictions based on model type
        if self.model_type == 'neural_network' or type(model).__name__ == 'NeuralNetTrainer':
            # Use predict method for neural networks
            y_pred = model.predict(X_test)
        elif hasattr(model, 'predict'):
            # Standard predict method for most model types
            y_pred = model.predict(X_test)
        else:
            raise ValueError(f"Unknown model type: {type(model)}")
        
        # Convert from log scale back to original scale
        if self.config.use_log10_transform:
            y_test_orig = (10**y_test) - 1
            y_pred_orig = (10**y_pred) - 1
            log_label = "log10"
        else:
            y_test_orig = np.exp(y_test) - 1e-6
            y_pred_orig = np.exp(y_pred) - 1e-6
            log_label = "ln"
        
        # Set a common style for all plots
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # 1. Actual vs Predicted plot
        plt.figure(figsize=(10, 8))
        plt.scatter(y_test, y_pred, alpha=0.3)
        plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
        plt.xlabel(f'Actual {log_label}(KIBA Score)')
        plt.ylabel(f'Predicted {log_label}(KIBA Score)')
        plt.title(f'Actual vs Predicted {log_label}(KIBA Score)')
        plt.tight_layout()
        plt.savefig(self.config.results_dir / f'actual_vs_predicted_{log_label}.png', 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Feature importance plot (XGBoost only)
        if self.model_type == 'xgboost':
            feature_importance = model.get_feature_importance()
            if feature_importance:  # Check if there are any features with importance
                indices = np.argsort(list(feature_importance.values()))[::-1]
                feature_names = list(feature_importance.keys())
                
                plt.figure(figsize=(12, 8))
                plt.barh(range(min(20, len(indices))), 
                        [list(feature_importance.values())[i] for i in indices[:20]], 
                        align='center')
                plt.yticks(range(min(20, len(indices))), [feature_names[i] for i in indices[:20]])
                plt.xlabel('Relative Importance')
                plt.title('Top 20 Feature Importance')
                plt.tight_layout()
                plt.savefig(self.config.results_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
                plt.close()
            else:
                logger.warning("No feature importance available")
        
        # 3. Distribution of predictions
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.hist(y_test, bins=50, alpha=0.5, label='Actual')
        plt.hist(y_pred, bins=50, alpha=0.5, label='Predicted')
        plt.xlabel(f'{log_label}(KIBA Score)')
        plt.ylabel('Frequency')
        plt.title(f'Distribution of {log_label}(KIBA Scores)')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.hist(y_test_orig, bins=50, alpha=0.5, label='Actual')
        plt.hist(y_pred_orig, bins=50, alpha=0.5, label='Predicted')
        plt.xlabel('KIBA Score')
        plt.ylabel('Frequency')
        plt.title('Distribution of KIBA Scores')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(self.config.results_dir / 'prediction_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Error heatmap
        plt.figure(figsize=(10, 8))
        plt.hexbin(y_test, y_pred, gridsize=50, cmap='viridis')
        plt.colorbar(label='Count')
        plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
        plt.xlabel(f'Actual {log_label}(KIBA Score)')
        plt.ylabel(f'Predicted {log_label}(KIBA Score)')
        plt.title(f'Actual vs Predicted {log_label}(KIBA Score) with Density Heatmap')
        plt.tight_layout()
        plt.savefig(self.config.results_dir / 'error_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. Error distribution
        errors = y_pred - y_test
        plt.figure(figsize=(10, 6))
        plt.hist(errors, bins=50)
        plt.xlabel(f'Prediction Error ({log_label} scale)')
        plt.ylabel('Frequency')
        plt.title(f'Distribution of Prediction Errors ({log_label} scale)')
        plt.axvline(x=0, color='r', linestyle='--')
        plt.tight_layout()
        plt.savefig(self.config.results_dir / 'error_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 6. Residual plot (error vs. predicted)
        plt.figure(figsize=(10, 6))
        plt.scatter(y_pred, errors, alpha=0.3)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel(f'Predicted {log_label}(KIBA Score)')
        plt.ylabel(f'Residual (Predicted - Actual)')
        plt.title('Residual Plot')
        plt.tight_layout()
        plt.savefig(self.config.results_dir / 'residual_plot.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualizations saved to {self.config.results_dir}")
