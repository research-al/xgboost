"""Prediction for KIBA model."""

import os
import logging
import traceback
import numpy as np
import h5py
import torch
from typing import Dict, Tuple, List, Optional, Union, Any

from kiba_model.config import KIBAConfig
from kiba_model.modeling.models.base import ModelFactory

logger = logging.getLogger("kiba_model")

class Predictor:
    """Makes predictions using trained models.
    
    This class handles loading models and embeddings for making predictions
    on new protein-compound pairs.
    """
    
    def __init__(self, config: KIBAConfig, model_type: str = None):
        """Initialize with configuration.
        
        Args:
            config: KIBAConfig object with paths and settings
            model_type: Type of model ('xgboost', 'neural_network', etc.)
        """
        self.config = config
        self.model = None
        self.model_type = model_type if model_type else getattr(config, 'model_type', 'xgboost')
        self.device = torch.device("cuda" if torch.cuda.is_available() and config.gpu_enabled else "cpu")
        self.protein_embeddings = None
        self.protein_ids = None
        self.compound_embeddings = None
        self.compound_ids = None
        self.protein_id_to_idx = {}
        self.cid_to_idx = {}
        
    def load_model_and_embeddings(self) -> None:
        """Load trained model and embeddings for prediction.
        
        Raises:
            FileNotFoundError: If model or embedding files don't exist
            ValueError: If model or embeddings can't be loaded properly
        """
        logger.info("Loading model and embeddings for prediction...")
        
        # Load model based on model type
        if self.model_type == 'neural_network':
            model_file = str(self.config.final_model_file).replace('.json', '.pt')
            
            if not os.path.exists(model_file):
                error_msg = f"Neural network model file not found: {model_file}"
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)
                
            try:
                # First load embeddings to get input dimension
                self._load_embeddings()
                
                # Initialize neural network model
                input_dim = self.protein_embeddings.shape[1] + self.compound_embeddings.shape[1] + 1
                self.model = ModelFactory.create_model('neural_network', self.config, input_dim=input_dim).to(self.device)
                
                # Load model weights
                self.model.load_state_dict(torch.load(model_file, map_location=self.device))
                self.model.eval()  # Set to evaluation mode
                
                logger.info(f"Loaded neural network model from {model_file}")
            except Exception as e:
                error_msg = f"Error loading neural network model: {str(e)}"
                logger.error(error_msg)
                logger.debug(traceback.format_exc())
                raise ValueError(error_msg)
        else:
            # Load XGBoost model
            model_file = self.config.final_model_file
            if not os.path.exists(model_file):
                error_msg = f"XGBoost model file not found: {model_file}"
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)
                
            try:
                self.model = ModelFactory.create_model('xgboost', self.config)
                self.model.load(model_file)
                logger.info(f"Loaded XGBoost model from {model_file}")
                
                # Load embeddings
                self._load_embeddings()
            except Exception as e:
                error_msg = f"Error loading XGBoost model: {str(e)}"
                logger.error(error_msg)
                logger.debug(traceback.format_exc())
                raise ValueError(error_msg)
    
    def _load_embeddings(self) -> None:
        """Load protein and compound embeddings."""
        # Load protein embeddings
        protein_file = self.config.protein_embeddings_file
        if not os.path.exists(protein_file):
            error_msg = f"Protein embeddings file not found: {protein_file}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
            
        try:
            with h5py.File(protein_file, 'r') as f:
                self.protein_embeddings = f['embeddings'][:]
                protein_ids_bytes = f['protein_ids'][:]
                
                # Convert bytes to strings if needed
                self.protein_ids = []
                for pid_bytes in protein_ids_bytes:
                    if isinstance(pid_bytes, bytes):
                        self.protein_ids.append(pid_bytes.decode('utf-8'))
                    else:
                        self.protein_ids.append(str(pid_bytes))
                        
            # Create protein ID to index mapping
            self.protein_id_to_idx = {pid: i for i, pid in enumerate(self.protein_ids)}
            logger.info(f"Loaded {len(self.protein_ids)} protein embeddings")
        except Exception as e:
            error_msg = f"Error loading protein embeddings: {str(e)}"
            logger.error(error_msg)
            logger.debug(traceback.format_exc())
            raise ValueError(error_msg)
            
        # Load compound embeddings
        compound_file = self.config.compound_embeddings_file
        if not os.path.exists(compound_file):
            error_msg = f"Compound embeddings file not found: {compound_file}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
            
        try:
            with h5py.File(compound_file, 'r') as f:
                self.compound_embeddings = f['embeddings'][:]
                compound_ids_bytes = f['cids'][:]
                
                # Convert bytes to strings if needed
                self.compound_ids = []
                for cid_bytes in compound_ids_bytes:
                    if isinstance(cid_bytes, bytes):
                        self.compound_ids.append(cid_bytes.decode('utf-8'))
                    else:
                        self.compound_ids.append(str(cid_bytes))
                        
            # Create compound ID to index mapping
            self.cid_to_idx = {cid: i for i, cid in enumerate(self.compound_ids)}
            logger.info(f"Loaded {len(self.compound_ids)} compound embeddings")
        except Exception as e:
            error_msg = f"Error loading compound embeddings: {str(e)}"
            logger.error(error_msg)
            logger.debug(traceback.format_exc())
            raise ValueError(error_msg)
    
    def predict(self, protein_id: str, compound_id: str, 
                is_experimental: bool = False) -> Optional[Dict[str, Any]]:
        """Predict KIBA score for a protein-compound pair.
        
        Args:
            protein_id: UniProt ID
            compound_id: PubChem CID
            is_experimental: Whether prediction is for experimental data
            
        Returns:
            Dictionary with prediction results or None if prediction fails
        """
        if self.model is None or self.protein_embeddings is None or self.compound_embeddings is None:
            error_msg = "Model or embeddings not loaded. Call load_model_and_embeddings() first."
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        try:
            # Convert IDs to strings
            protein_id = str(protein_id)
            compound_id = str(compound_id)
            
            # Check if IDs are in embeddings
            if protein_id not in self.protein_id_to_idx:
                logger.warning(f"Protein ID {protein_id} not found in embeddings")
                return None
            
            if compound_id not in self.cid_to_idx:
                logger.warning(f"Compound ID {compound_id} not found in embeddings")
                return None
            
            # Get embedding indices
            protein_idx = self.protein_id_to_idx[protein_id]
            compound_idx = self.cid_to_idx[compound_id]
            
            # Get dimensions
            protein_dim = self.protein_embeddings.shape[1]
            compound_dim = self.compound_embeddings.shape[1]
            
            # Create feature vector
            X = np.zeros(protein_dim + compound_dim + 1, dtype=np.float32)
            X[:protein_dim] = self.protein_embeddings[protein_idx]
            X[protein_dim:protein_dim+compound_dim] = self.compound_embeddings[compound_idx]
            X[-1] = int(is_experimental)  # Experimental flag
            
            # Clean up feature vector
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Make prediction based on model type
            if self.model_type == 'neural_network':
                # Convert to PyTorch tensor
                X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
                
                # Make prediction
                with torch.no_grad():
                    self.model.eval()
                    y_pred_log = self.model(X_tensor.unsqueeze(0)).item()
            else:
                # Make prediction
                y_pred_log = self.model.predict(X)[0]
            
            # Convert to original scale
            if self.config.use_log10_transform:
                y_pred = (10**y_pred_log) - 1
            else:
                y_pred = np.exp(y_pred_log) - 1e-6
            
            # Return prediction results
            return {
                'protein_id': protein_id,
                'compound_id': compound_id,
                'is_experimental': is_experimental,
                'kiba_score_log': float(y_pred_log),
                'kiba_score': float(y_pred)
            }
        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            logger.debug(traceback.format_exc())
            return None
    
    def predict_by_id(self, uniprot_id: str, pubchem_id: str) -> Optional[Dict[str, Any]]:
        """
        Predict KIBA score directly from UniProt ID and PubChem ID.
        
        Args:
            uniprot_id: UniProt ID of the protein
            pubchem_id: PubChem CID of the compound
            
        Returns:
            Dictionary with prediction results or None if prediction fails
        """
        logger.info(f"Predicting for UniProt ID {uniprot_id} and PubChem CID {pubchem_id}")
        
        # Check if IDs are in embeddings
        protein_found = uniprot_id in self.protein_id_to_idx if self.protein_id_to_idx else False
        compound_found = pubchem_id in self.cid_to_idx if self.cid_to_idx else False
        
        if not protein_found or not compound_found:
            # Need to generate embeddings
            from kiba_model.features.embedding_generator import EmbeddingGenerator
            generator = EmbeddingGenerator(self.config)
            
            if not protein_found:
                logger.info(f"Protein {uniprot_id} not found in embeddings, fetching and generating...")
                try:
                    # Fetch and generate
                    protein_embeddings, protein_ids = generator.generate_protein_embeddings(
                        uniprot_ids=[uniprot_id]
                    )
                    
                    # Update embeddings in memory
                    if len(protein_ids) > 0:
                        idx = len(self.protein_ids)
                        self.protein_embeddings = np.vstack([self.protein_embeddings, protein_embeddings])
                        self.protein_ids.append(protein_ids[0])
                        self.protein_id_to_idx[protein_ids[0]] = idx
                        protein_found = True
                except Exception as e:
                    logger.error(f"Failed to generate protein embedding: {str(e)}")
                    return None
            
            if not compound_found:
                logger.info(f"Compound {pubchem_id} not found in embeddings, fetching and generating...")
                try:
                    # Fetch and generate
                    compound_embeddings, compound_ids = generator.generate_compound_embeddings(
                        compound_ids=[pubchem_id]
                    )
                    
                    # Update embeddings in memory
                    if len(compound_ids) > 0:
                        idx = len(self.compound_ids)
                        self.compound_embeddings = np.vstack([self.compound_embeddings, compound_embeddings])
                        self.compound_ids.append(compound_ids[0])
                        self.cid_to_idx[compound_ids[0]] = idx
                        compound_found = True
                except Exception as e:
                    logger.error(f"Failed to generate compound embedding: {str(e)}")
                    return None
        
        # Now predict using the embeddings
        if protein_found and compound_found:
            return self.predict(uniprot_id, pubchem_id)
        else:
            if not protein_found:
                logger.error(f"Could not find or generate embedding for protein {uniprot_id}")
            if not compound_found:
                logger.error(f"Could not find or generate embedding for compound {pubchem_id}")
            return None
    
    def predict_batch(self, protein_ids: List[str], compound_ids: List[str], 
                     is_experimental: Union[bool, List[bool]] = False) -> List[Optional[Dict[str, Any]]]:
        """Predict KIBA scores for multiple protein-compound pairs.
        
        Args:
            protein_ids: List of UniProt IDs
            compound_ids: List of PubChem CIDs
            is_experimental: Whether predictions are for experimental data (bool or list of bools)
            
        Returns:
            List of prediction results
            
        Raises:
            ValueError: If input lists have different lengths
        """
        if len(protein_ids) != len(compound_ids):
            error_msg = "Length of protein_ids and compound_ids must be the same"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        # Convert is_experimental to list if it's a single value
        if isinstance(is_experimental, bool):
            is_experimental = [is_experimental] * len(protein_ids)
            
        if len(is_experimental) != len(protein_ids):
            error_msg = "Length of is_experimental must match protein_ids and compound_ids"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.info(f"Predicting KIBA scores for {len(protein_ids)} protein-compound pairs...")
        
        # Make predictions
        results = []
        for i, (protein_id, compound_id) in enumerate(zip(protein_ids, compound_ids)):
            result = self.predict(protein_id, compound_id, is_experimental[i])
            results.append(result)
            
        # Count successful predictions
        successful = sum(1 for r in results if r is not None)
        logger.info(f"Successfully predicted {successful}/{len(protein_ids)} pairs")
            
        return results
