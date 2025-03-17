"""Configuration class for KIBA model pipeline."""

import os
import logging
import shutil
import pickle
from pathlib import Path
from typing import Dict, Any
from datetime import datetime


logger = logging.getLogger("kiba_model")

class KIBAConfig:
    """Configuration manager for KIBA model pipeline."""
    
    def __init__(
        self, 
        data_dir: str = 'data',
        models_dir: str = 'models',
        results_dir: str = 'results',
        logs_dir: str = 'logs',
        kiba_score_threshold: float = 100.0,
        protein_min_length: int = 100,
        protein_max_length: int = 2000,
        smiles_max_length: int = 200,
        use_log10_transform: bool = True,
        random_state: int = 42,
        gpu_enabled: bool = True,
        use_stratification: bool = True,
        batch_size: int = 32,
        test_size: float = 0.2,
        val_size: float = 0.25,
        backup_existing: bool = True,
        allow_empty_results: bool = False,
        min_valid_interactions: int = 50,
        fallback_to_lenient_filtering: bool = True
    ):
        """Initialize KIBA model configuration."""
        # Store runtime configuration
        self.kiba_score_threshold = kiba_score_threshold
        self.protein_min_length = protein_min_length
        self.protein_max_length = protein_max_length
        self.smiles_max_length = smiles_max_length
        self.use_log10_transform = use_log10_transform
        self.random_state = random_state
        self.gpu_enabled = gpu_enabled
        self.use_stratification = use_stratification
        self.batch_size = batch_size
        self.test_size = test_size
        self.val_size = val_size
        self.backup_existing = backup_existing
        self.allow_empty_results = allow_empty_results
        self.min_valid_interactions = min_valid_interactions
        self.fallback_to_lenient_filtering = fallback_to_lenient_filtering
        
        # Setup directories
        self.data_dir = Path(data_dir)
        self.models_dir = Path(models_dir)
        self.results_dir = Path(results_dir)
        self.logs_dir = Path(logs_dir)
        
        # Create directories if they don't exist
        for directory in [self.data_dir, self.models_dir, self.results_dir, self.logs_dir]:
            directory.mkdir(exist_ok=True, parents=True)
            
        # Source file paths (to be set later)
        self.kiba_file = None
        self.protein_file = None
        self.compound_file = None
        
        # Processed data file paths
        self.filtered_interactions_file = self.data_dir / 'filtered_interactions.csv'
        self.filtered_proteins_file = self.data_dir / 'filtered_proteins.csv'
        self.filtered_compounds_file = self.data_dir / 'filtered_compounds.csv'
        
        # Embedding file paths
        self.protein_embeddings_file = self.data_dir / 'esm_embeddings.h5'
        self.compound_embeddings_file = self.data_dir / 'chemberta_embeddings.h5'
        
        # Feature matrices
        self.X_features_file = self.data_dir / 'X_features.npy'
        self.y_target_file = self.data_dir / 'y_target.npy'
        self.strata_array_file = self.data_dir / 'strata_array.npy'
        
        # Model files
        suffix = "log10" if use_log10_transform else "ln"
        self.initial_model_file = self.models_dir / f'initial_model_{suffix}.json'
        self.final_model_file = self.models_dir / f'final_model_{suffix}.json'
        self.best_params_file = self.models_dir / f'best_params_{suffix}.pkl'
        
        # Results files
        self.metrics_file = self.results_dir / f'metrics_{suffix}.pkl'
        
        logger.info(f"Initialized KIBA model configuration")
        logger.info(f"Data directory: {self.data_dir}")
        logger.info(f"Model directory: {self.models_dir}")
        logger.info(f"Results directory: {self.results_dir}")
        logger.info(f"Protein sequence length range: {self.protein_min_length}-{self.protein_max_length}")
        logger.info(f"Using {'log10' if use_log10_transform else 'natural log'} transformation")
        
    def set_file_paths(self, kiba_file: str, protein_file: str, compound_file: str) -> None:
        """Set paths for input data files."""
        self.kiba_file = kiba_file
        self.protein_file = protein_file
        self.compound_file = compound_file
        
        logger.info(f"Set input file paths:")
        logger.info(f"  KIBA file: {self.kiba_file}")
        logger.info(f"  Protein file: {self.protein_file}")
        logger.info(f"  Compound file: {self.compound_file}")
        
    def backup_files(self) -> None:
        """Create backups of existing output files if they exist."""
        # Implementation of backup_files method here
        # (Copy from original implementation)
        
    def get_xgb_params(self) -> Dict[str, Any]:
        """Get default XGBoost parameters."""
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'eta': 0.1,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8
        }
        
        if self.gpu_enabled:
            try:
                # Add GPU parameters if enabled
                params['tree_method'] = 'gpu_hist'
                params['gpu_id'] = 0
            except Exception as e:
                logger.warning(f"Could not configure GPU: {e}")
                logger.info("Falling back to CPU training")
                
        return params