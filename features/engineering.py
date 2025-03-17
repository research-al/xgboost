"""Feature engineering for KIBA prediction model."""

import os
import time
import logging
import traceback
import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm
from typing import Dict, Tuple, List, Optional, Union, Any

from kiba_model.config import KIBAConfig

logger = logging.getLogger("kiba_model")



class FeatureEngineering:
    """Generates and manages embeddings for proteins and compounds.
    
    This class handles loading pre-computed embeddings and creating feature matrices
    for training machine learning models.
    """
    
    def __init__(self, config: KIBAConfig):
        """Initialize with configuration.
        
        Args:
            config: KIBAConfig object with paths and settings
        """
        self.config = config
        self.protein_embeddings = None
        self.protein_ids = None
        self.compound_embeddings = None
        self.compound_ids = None
        self.protein_id_to_idx = {}
        self.cid_to_idx = {}
        
    def load_embeddings(self) -> None:
        """Load protein and compound embeddings from H5 files.
        
        Raises:
            FileNotFoundError: If embedding files don't exist
            ValueError: If embeddings can't be loaded properly
        """
        # Check if embedding files exist
        protein_embeddings_exist = os.path.exists(self.config.protein_embeddings_file)
        compound_embeddings_exist = os.path.exists(self.config.compound_embeddings_file)
        
        if not protein_embeddings_exist:
            error_msg = f"Protein embeddings file not found: {self.config.protein_embeddings_file}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
            
        if not compound_embeddings_exist:
            error_msg = f"Compound embeddings file not found: {self.config.compound_embeddings_file}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        logger.info("Loading embeddings from disk...")
        
        try:
            # Load protein embeddings
            start_time = time.time()
            with h5py.File(self.config.protein_embeddings_file, 'r') as f:
                self.protein_embeddings = f['embeddings'][:]
                protein_ids_bytes = f['protein_ids'][:]
                
                # Convert bytes to strings if needed
                self.protein_ids = []
                for pid_bytes in protein_ids_bytes:
                    if isinstance(pid_bytes, bytes):
                        self.protein_ids.append(pid_bytes.decode('utf-8'))
                    else:
                        self.protein_ids.append(str(pid_bytes))
            
            load_time = time.time() - start_time
            logger.info(f"Loaded {len(self.protein_ids)} protein embeddings with "
                        f"{self.protein_embeddings.shape[1]} dimensions in {load_time:.2f}s")
            
            # Create protein ID to index mapping
            self.protein_id_to_idx = {pid: i for i, pid in enumerate(self.protein_ids)}
            
        except Exception as e:
            error_msg = f"Error loading protein embeddings: {str(e)}"
            logger.error(error_msg)
            logger.debug(traceback.format_exc())
            raise ValueError(error_msg)
        
        try:
            # Load compound embeddings
            start_time = time.time()
            with h5py.File(self.config.compound_embeddings_file, 'r') as f:
                self.compound_embeddings = f['embeddings'][:]
                compound_ids_bytes = f['cids'][:]
                
                # Convert bytes to strings if needed
                self.compound_ids = []
                for cid_bytes in compound_ids_bytes:
                    if isinstance(cid_bytes, bytes):
                        self.compound_ids.append(cid_bytes.decode('utf-8'))
                    else:
                        self.compound_ids.append(str(cid_bytes))
            
            load_time = time.time() - start_time
            logger.info(f"Loaded {len(self.compound_ids)} compound embeddings with "
                        f"{self.compound_embeddings.shape[1]} dimensions in {load_time:.2f}s")
            
            # Create compound ID to index mapping
            self.cid_to_idx = {cid: i for i, cid in enumerate(self.compound_ids)}
            
        except Exception as e:
            error_msg = f"Error loading compound embeddings: {str(e)}"
            logger.error(error_msg)
            logger.debug(traceback.format_exc())
            raise ValueError(error_msg)
    
    def create_feature_matrix(self, interactions: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Create feature matrix from interaction data and embeddings.
        
        Args:
            interactions: DataFrame with interaction data
            
        Returns:
            Tuple containing (X, y, strata_array) where X is the feature matrix,
            y is the target vector, and strata_array is optional stratification labels
            
        Raises:
            ValueError: If embeddings aren't loaded or feature creation fails
        """
        if self.protein_embeddings is None or self.compound_embeddings is None:
            error_msg = "Embeddings not loaded. Call load_embeddings() first."
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        logger.info(f"Creating feature matrix from {len(interactions)} interactions...")
        
        # Get dimensions and initialize arrays
        compound_dim = self.compound_embeddings.shape[1]
        protein_dim = self.protein_embeddings.shape[1]
        total_interactions = len(interactions)
        
        X = np.zeros((total_interactions, compound_dim + protein_dim + 1), dtype=np.float32)
        y = np.zeros(total_interactions, dtype=np.float32)
        
        strata_values = [] if 'strata' in interactions.columns else None
        valid_indices = []
        invalid_count = 0
        not_found_proteins = set()
        not_found_compounds = set()
        
        # Process each interaction
        for i, (idx, row) in enumerate(tqdm(interactions.iterrows(), 
                                           total=len(interactions),
                                           desc="Creating features")):
            try:
                # Get IDs, ensuring they're treated as strings
                protein_id = str(row['UniProt_ID_str'])
                compound_id = str(row['pubchem_cid_str'])
                
                # Skip if protein not in embeddings
                if protein_id not in self.protein_id_to_idx:
                    if protein_id not in not_found_proteins:  # Only log each missing ID once
                        not_found_proteins.add(protein_id)
                        logger.debug(f"Protein ID {protein_id} not found in embeddings")
                    invalid_count += 1
                    continue
                
                # Skip if compound not in embeddings
                if compound_id not in self.cid_to_idx:
                    if compound_id not in not_found_compounds:  # Only log each missing ID once
                        not_found_compounds.add(compound_id)
                        logger.debug(f"Compound ID {compound_id} not found in embeddings")
                    invalid_count += 1
                    continue
                
                # Get embedding indices
                protein_idx = self.protein_id_to_idx[protein_id]
                compound_idx = self.cid_to_idx[compound_id]
                
                # Add protein embedding
                X[i, :protein_dim] = self.protein_embeddings[protein_idx]
                
                # Add compound embedding
                X[i, protein_dim:protein_dim+compound_dim] = self.compound_embeddings[compound_idx]
                
                # Add experimental flag
                X[i, -1] = row['is_experimental']
                
                # Get KIBA score
                kiba_score = row['kiba_score']
                
                # Handle problematic values
                if np.isnan(kiba_score) or np.isinf(kiba_score) or kiba_score <= 0:
                    logger.debug(f"Skipping interaction {idx} with problematic KIBA score: {kiba_score}")
                    invalid_count += 1
                    continue
                
                # Transform KIBA score based on configuration
                if self.config.use_log10_transform:
                    # Log10 transform with +1 offset
                    log_kiba = np.log10(kiba_score + 1)
                else:
                    # Natural log transform with small epsilon
                    log_kiba = np.log(kiba_score + 1e-6)
                
                # Check if log result is valid
                if np.isnan(log_kiba) or np.isinf(log_kiba):
                    logger.debug(f"Skipping interaction {idx} with invalid log transform: {log_kiba}")
                    invalid_count += 1
                    continue
                
                # Set target and record index as valid
                y[i] = log_kiba
                valid_indices.append(i)
                
                # Add strata if available
                if strata_values is not None and 'strata' in row:
                    strata_values.append(row['strata'])
                
            except Exception as e:
                logger.warning(f"Error processing interaction {idx}: {str(e)}")
                logger.debug(traceback.format_exc())
                invalid_count += 1
        
        # Log summary of missing IDs
        if not_found_proteins:
            logger.warning(f"{len(not_found_proteins)} unique protein IDs not found in embeddings")
        if not_found_compounds:
            logger.warning(f"{len(not_found_compounds)} unique compound IDs not found in embeddings")
        
        # Trim arrays to only valid entries
        if valid_indices:
            X = X[valid_indices]
            y = y[valid_indices]
            strata_array = np.array(strata_values) if strata_values else None
            
            logger.info(f"Created features for {len(valid_indices)} valid interactions")
            logger.info(f"Skipped {invalid_count} interactions with problematic values or missing embeddings")
            
            # Final validation check and cleanup
            if np.isnan(X).any() or np.isinf(X).any():
                logger.warning("Feature matrix contains NaN or Inf values. Fixing...")
                X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
                
            if np.isnan(y).any() or np.isinf(y).any():
                logger.warning("Target vector contains NaN or Inf values. Fixing...")
                valid_mask = ~(np.isnan(y) | np.isinf(y))
                X = X[valid_mask]
                y = y[valid_mask]
                if strata_array is not None:
                    strata_array = strata_array[valid_mask]
                logger.info(f"After removing NaN/Inf in target: {len(y)} valid interactions")
            
            # Save feature matrices
            np.save(self.config.X_features_file, X)
            np.save(self.config.y_target_file, y)
            if strata_array is not None:
                np.save(self.config.strata_array_file, strata_array)
                
            logger.info(f"Saved feature matrices to {self.config.data_dir}")
            logger.info(f"Final feature matrix shape: {X.shape}")
            
            # Log distribution statistics
            y_orig = self._convert_to_original_scale(y)
            logger.info(f"Target distribution (original scale): min={y_orig.min():.2f}, "
                        f"max={y_orig.max():.2f}, mean={y_orig.mean():.2f}, median={np.median(y_orig):.2f}")
            logger.info(f"Target distribution (log scale): min={y.min():.2f}, "
                        f"max={y.max():.2f}, mean={y.mean():.2f}, median={np.median(y):.2f}")
            
            return X, y, strata_array
        else:
            error_msg = "No valid interactions after feature creation"
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    def _convert_to_original_scale(self, y_log: np.ndarray) -> np.ndarray:
        """Convert log-transformed KIBA scores back to original scale.
        
        Args:
            y_log: Log-transformed KIBA scores
            
        Returns:
            KIBA scores in original scale
        """
        if self.config.use_log10_transform:
            return (10 ** y_log) - 1
        else:
            return np.exp(y_log) - 1e-6

    # In kiba_model/features/engineering.py

def load_embeddings(self) -> None:
    """Load protein and compound embeddings from H5 files."""
    # Check if embedding files exist
    protein_embeddings_exist = os.path.exists(self.config.protein_embeddings_file)
    compound_embeddings_exist = os.path.exists(self.config.compound_embeddings_file)
    
    # If either file is missing, try to generate embeddings
    if not protein_embeddings_exist or not compound_embeddings_exist:
        from kiba_model.features.embedding_generator import EmbeddingGenerator
        generator = EmbeddingGenerator(self.config)
        
        # Load raw data for embedding generation
        if not protein_embeddings_exist:
            try:
                logger.info(f"Protein embeddings file not found: {self.config.protein_embeddings_file}")
                logger.info("Attempting to generate protein embeddings...")
                
                # Try to load protein data
                if os.path.exists(self.config.filtered_proteins_file):
                    proteins_df = pd.read_csv(self.config.filtered_proteins_file)
                    generator.generate_protein_embeddings(proteins_df=proteins_df)
                else:
                    logger.error("Cannot generate protein embeddings: no filtered proteins file found")
                    raise FileNotFoundError(f"Protein embeddings file not found: {self.config.protein_embeddings_file}")
            except Exception as e:
                logger.error(f"Failed to generate protein embeddings: {str(e)}")
                raise
                
        if not compound_embeddings_exist:
            try:
                logger.info(f"Compound embeddings file not found: {self.config.compound_embeddings_file}")
                logger.info("Attempting to generate compound embeddings...")
                
                # Try to load compound data
                if os.path.exists(self.config.filtered_compounds_file):
                    compounds_df = pd.read_csv(self.config.filtered_compounds_file)
                    generator.generate_compound_embeddings(compounds_df=compounds_df)
                else:
                    logger.error("Cannot generate compound embeddings: no filtered compounds file found")
                    raise FileNotFoundError(f"Compound embeddings file not found: {self.config.compound_embeddings_file}")
            except Exception as e:
                logger.error(f"Failed to generate compound embeddings: {str(e)}")
                raise
    
    # Now try to load the embeddings (they should exist now)
    try:
        # Load protein embeddings
        start_time = time.time()
        with h5py.File(self.config.protein_embeddings_file, 'r') as f:
            self.protein_embeddings = f['embeddings'][:]
            protein_ids_bytes = f['protein_ids'][:]
            
            # Convert bytes to strings if needed
            self.protein_ids = []
            for pid_bytes in protein_ids_bytes:
                if isinstance(pid_bytes, bytes):
                    self.protein_ids.append(pid_bytes.decode('utf-8'))
                else:
                    self.protein_ids.append(str(pid_bytes))
        
        load_time = time.time() - start_time
        logger.info(f"Loaded {len(self.protein_ids)} protein embeddings with "
                    f"{self.protein_embeddings.shape[1]} dimensions in {load_time:.2f}s")
        
        # Create protein ID to index mapping
        self.protein_id_to_idx = {pid: i for i, pid in enumerate(self.protein_ids)}
        
    except Exception as e:
        error_msg = f"Error loading protein embeddings: {str(e)}"
        logger.error(error_msg)
        logger.debug(traceback.format_exc())
        raise ValueError(error_msg)
    
    try:
        # Load compound embeddings
        start_time = time.time()
        with h5py.File(self.config.compound_embeddings_file, 'r') as f:
            self.compound_embeddings = f['embeddings'][:]
            compound_ids_bytes = f['cids'][:]
            
            # Convert bytes to strings if needed
            self.compound_ids = []
            for cid_bytes in compound_ids_bytes:
                if isinstance(cid_bytes, bytes):
                    self.compound_ids.append(cid_bytes.decode('utf-8'))
                else:
                    self.compound_ids.append(str(cid_bytes))
        
        load_time = time.time() - start_time
        logger.info(f"Loaded {len(self.compound_ids)} compound embeddings with "
                    f"{self.compound_embeddings.shape[1]} dimensions in {load_time:.2f}s")
        
        # Create compound ID to index mapping
        self.cid_to_idx = {cid: i for i, cid in enumerate(self.compound_ids)}
        
    except Exception as e:
        error_msg = f"Error loading compound embeddings: {str(e)}"
        logger.error(error_msg)
        logger.debug(traceback.format_exc())
        raise ValueError(error_msg)
