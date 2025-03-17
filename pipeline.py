"""Main pipeline for KIBA prediction model."""

import logging
import time
import traceback
import pandas as pd
import numpy as np
import xgboost as xgb
from typing import Dict, Tuple, List, Optional, Union, Any
import torch


from kiba_model.config import KIBAConfig
from kiba_model.data.loader import DataLoader
from kiba_model.data.preprocessor import DataPreprocessor
from kiba_model.features.engineering import FeatureEngineering
from kiba_model.modeling.trainer import ModelTrainer
from kiba_model.modeling.evaluator import ModelEvaluator
from kiba_model.modeling.predictor import Predictor

logger = logging.getLogger("kiba_model")

class KIBAModelPipeline:
    """Orchestrates the KIBA model pipeline.
    
    This class coordinates the entire pipeline from data loading to model evaluation.
    """
    
    def __init__(self, config: KIBAConfig, model_type: str = None):
        """Initialize with configuration.
        
        Args:
            config: KIBAConfig object with settings
            model_type: Type of model to use ('xgboost', 'neural_network', etc.)
        """
        self.config = config
        # Use model_type from config if not explicitly provided
        self.model_type = model_type if model_type is not None else getattr(config, 'model_type', 'xgboost')
        self.data_loader = DataLoader(config)
        self.data_preprocessor = DataPreprocessor(config)
        self.feature_engineering = FeatureEngineering(config)
        self.model_trainer = ModelTrainer(config, model_type=self.model_type)
        self.model_evaluator = ModelEvaluator(config)
        self.predictor = Predictor(config, model_type=self.model_type)
        
        # Create backup of existing files if configured
        self.config.backup_files()
        
    def run_preprocessing_pipeline(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Run the preprocessing pipeline from raw data to filtered data.
        
        Returns:
            Tuple of filtered (interactions, proteins, compounds) DataFrames
        """
        logger.info("=== Running preprocessing pipeline ===")
        
        # Load raw data
        kiba_data, protein_data, compound_data = self.data_loader.load_data()
        
        # Validate data quality
        self.data_loader.validate_data()
        
        # Preprocess and filter data
        valid_interactions, valid_proteins, valid_compounds = self.data_preprocessor.preprocess_data(
            kiba_data, protein_data, compound_data
        )
        
        logger.info("=== Preprocessing pipeline completed ===")
        
        return valid_interactions, valid_proteins, valid_compounds
    
    def run_feature_engineering_pipeline(self, interactions: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Run the feature engineering pipeline to create feature matrices.
        
        Args:
            interactions: DataFrame with filtered interaction data
            
        Returns:
            Tuple containing (X, y, strata_array)
        """
        logger.info("=== Running feature engineering pipeline ===")
        
        # Load embeddings
        self.feature_engineering.load_embeddings()
        
        # Create feature matrix
        X, y, strata_array = self.feature_engineering.create_feature_matrix(interactions)
        
        logger.info("=== Feature engineering pipeline completed ===")
        
        return X, y, strata_array
    
    def run_modeling_pipeline(self, X: np.ndarray, y: np.ndarray, 
                         strata_array: Optional[np.ndarray] = None) -> Union[xgb.Booster, torch.nn.Module]:
        """Run the modeling pipeline to train and evaluate models.
        
        Args:
            X: Feature matrix
            y: Target vector
            strata_array: Optional strata for stratified sampling
            
        Returns:
            Trained final model
        """
        logger.info("=== Running modeling pipeline ===")
        
        # Split data
        self.model_trainer.split_data(X, y, strata_array)
        
        # Train initial model
        self.model_trainer.train_initial_model()
        
        # Tune hyperparameters
        self.model_trainer.tune_hyperparameters()
        
        # Train final model
        final_model = self.model_trainer.train_final_model()
        
        # Evaluate model
        self.model_evaluator.evaluate_model(
            final_model, 
            self.model_trainer.X_test, 
            self.model_trainer.y_test,
            self.model_trainer.strata_test
        )
        
        # Generate visualizations
        self.model_evaluator.generate_visualizations(
            final_model,
            self.model_trainer.X_test,
            self.model_trainer.y_test
        )
        
        logger.info("=== Modeling pipeline completed ===")
        
        return final_model
    
    def run_full_pipeline(self) -> Union[xgb.Booster, torch.nn.Module]:
        """Run the full pipeline from raw data to trained model.
        
        Returns:
            Trained final model
        """
        logger.info("====== Starting full KIBA modeling pipeline ======")
        start_time = time.time()
        
        try:
            # Preprocessing
            valid_interactions, _, _ = self.run_preprocessing_pipeline()
            
            # Feature engineering
            X, y, strata_array = self.run_feature_engineering_pipeline(valid_interactions)
            
            # Modeling
            final_model = self.run_modeling_pipeline(X, y, strata_array)
            
            total_time = time.time() - start_time
            logger.info(f"====== Pipeline completed successfully in {total_time:.2f}s ======")
            
            return final_model
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            logger.debug(traceback.format_exc())
            raise  # Re-raise the exception to ensure proper error handling upstream
    
    def setup_for_prediction(self) -> None:
        """Set up the pipeline for making predictions.
        """
        logger.info("Setting up for prediction...")
        
        # Load model and embeddings
        self.predictor.load_model_and_embeddings()
        
        logger.info("Ready for predictions")
    
    def predict(self, protein_id: str, compound_id: str, 
               is_experimental: bool = False) -> Optional[Dict[str, Any]]:
        """Make a prediction for a protein-compound pair.
        
        Args:
            protein_id: UniProt ID
            compound_id: PubChem CID
            is_experimental: Whether prediction is for experimental data
            
        Returns:
            Dictionary with prediction results
        """
        return self.predictor.predict(protein_id, compound_id, is_experimental)
    
    def predict_batch(self, protein_ids: List[str], compound_ids: List[str],
                    is_experimental: Union[bool, List[bool]] = False) -> List[Optional[Dict[str, Any]]]:
        """Make predictions for multiple protein-compound pairs.
        
        Args:
            protein_ids: List of UniProt IDs
            compound_ids: List of PubChem CIDs
            is_experimental: Whether predictions are for experimental data
            
        Returns:
            List of prediction results
        """
        return self.predictor.predict_batch(protein_ids, compound_ids, is_experimental)
    

    def predict_by_id(self, uniprot_id: str, pubchem_id: str, is_experimental: bool = False) -> Optional[Dict[str, Any]]:
        """
        Make a prediction by UniProt ID and PubChem ID.
        
        Args:
            uniprot_id: UniProt ID of protein
            pubchem_id: PubChem CID of compound
            is_experimental: Whether prediction is for experimental data
            
        Returns:
            Dictionary with prediction results
        """
        self.setup_for_prediction()
        return self.predictor.predict_by_id(uniprot_id, pubchem_id, is_experimental)
