"""Data loading module for KIBA prediction model."""

import os
import time
import logging
import traceback
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Set, List, Optional, Union, Any

from kiba_model.config import KIBAConfig

logger = logging.getLogger("kiba_model")

class DataLoader:
    """Loads and validates raw KIBA dataset files."""
    
    def __init__(self, config: KIBAConfig):
        """Initialize with configuration."""
        self.config = config
        self.kiba_data = None
        self.protein_data = None
        self.compound_data = None
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load raw data files.
        
        Returns:
            Tuple containing (kiba_data, protein_data, compound_data) as pandas DataFrames
        
        Raises:
            FileNotFoundError: If input files don't exist
            ValueError: If files are empty or have invalid format
        """
        logger.info("Loading raw data files...")
        
        # Check if files exist
        for file_path, file_type in [
            (self.config.kiba_file, "KIBA"),
            (self.config.protein_file, "protein"),
            (self.config.compound_file, "compound")
        ]:
            if not os.path.exists(file_path):
                error_msg = f"{file_type} file not found: {file_path}"
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)
        
        # Load KIBA data
        try:
            start_time = time.time()
            self.kiba_data = pd.read_csv(self.config.kiba_file)
            
            # Fix pubchem_cid data type to ensure consistent ID matching
            if 'pubchem_cid' in self.kiba_data.columns:
                try:
                    # Convert to integer if possible (removing decimal part)
                    self.kiba_data['pubchem_cid'] = self.kiba_data['pubchem_cid'].astype(int)
                    logger.info("Converted pubchem_cid to integer for better ID matching")
                except Exception as e:
                    logger.warning(f"Could not convert pubchem_cid to integer: {str(e)}")
            
            logger.info(f"Loaded KIBA data: {len(self.kiba_data)} rows in {time.time() - start_time:.2f}s")
            
            # Check if empty
            if len(self.kiba_data) == 0:
                raise ValueError("KIBA data file is empty")
                
            # Check required columns
            required_columns = ['UniProt_ID', 'pubchem_cid', 'kiba_score', 'kiba_score_estimated']
            missing_columns = [col for col in required_columns if col not in self.kiba_data.columns]
            if missing_columns:
                raise ValueError(f"KIBA data missing required columns: {missing_columns}")
                
        except Exception as e:
            logger.error(f"Error loading KIBA data: {str(e)}")
            logger.debug(traceback.format_exc())
            raise
            
        # Load protein data
        try:
            start_time = time.time()
            self.protein_data = pd.read_csv(self.config.protein_file)
            logger.info(f"Loaded protein data: {len(self.protein_data)} rows in {time.time() - start_time:.2f}s")
            
            # Check if empty
            if len(self.protein_data) == 0:
                raise ValueError("Protein data file is empty")
                
            # Check required columns
            required_columns = ['UniProt_ID', 'Protein_Sequence']
            missing_columns = [col for col in required_columns if col not in self.protein_data.columns]
            if missing_columns:
                raise ValueError(f"Protein data missing required columns: {missing_columns}")
                
        except Exception as e:
            logger.error(f"Error loading protein data: {str(e)}")
            logger.debug(traceback.format_exc())
            raise
            
        # Load compound data
        try:
            start_time = time.time()
            self.compound_data = pd.read_csv(self.config.compound_file)
            logger.info(f"Loaded compound data: {len(self.compound_data)} rows in {time.time() - start_time:.2f}s")
            
            # Check if empty
            if len(self.compound_data) == 0:
                raise ValueError("Compound data file is empty")
                
            # Check required columns
            required_columns = ['cid', 'smiles']
            missing_columns = [col for col in required_columns if col not in self.compound_data.columns]
            if missing_columns:
                raise ValueError(f"Compound data missing required columns: {missing_columns}")
                
        except Exception as e:
            logger.error(f"Error loading compound data: {str(e)}")
            logger.debug(traceback.format_exc())
            raise
            
        # Log sample data
        logger.debug(f"KIBA data sample:\n{self.kiba_data.head()}")
        logger.debug(f"Protein data sample:\n{self.protein_data.head()}")
        logger.debug(f"Compound data sample:\n{self.compound_data.head()}")
        
        return self.kiba_data, self.protein_data, self.compound_data
        
    def validate_data(self) -> Dict[str, Dict[str, Any]]:
        """Check data quality and report issues.
        
        Returns:
            Dictionary containing validation statistics
        
        Raises:
            ValueError: If data has not been loaded yet
        """
        if self.kiba_data is None or self.protein_data is None or self.compound_data is None:
            error_msg = "Data not loaded. Call load_data() first."
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        logger.info("Validating data quality...")
        
        # Calculate protein sequence lengths if not already done
        if 'seq_length' not in self.protein_data.columns:
            self.protein_data['seq_length'] = self.protein_data['Protein_Sequence'].apply(len)
            
        # Calculate SMILES complexity if not already done
        if 'smiles_length' not in self.compound_data.columns:
            self.compound_data['smiles_length'] = self.compound_data['smiles'].apply(len)
        
        # Check KIBA data quality
        kiba_stats = {
            'total_rows': len(self.kiba_data),
            'nan_values': np.isnan(self.kiba_data['kiba_score']).sum(),
            'inf_values': np.isinf(self.kiba_data['kiba_score']).sum(),
            'negative_values': (self.kiba_data['kiba_score'] < 0).sum(),
            'zero_values': (self.kiba_data['kiba_score'] == 0).sum(),
            'large_values': (self.kiba_data['kiba_score'] > self.config.kiba_score_threshold).sum(),
            'above_threshold': (self.kiba_data['kiba_score'] >= self.config.kiba_score_threshold).sum(),
            'missing_uniprot': self.kiba_data['UniProt_ID'].isna().sum(),
            'missing_pubchem': self.kiba_data['pubchem_cid'].isna().sum(),
        }
        
        # Check protein data quality
        protein_stats = {
            'total_rows': len(self.protein_data),
            'missing_sequences': self.protein_data['Protein_Sequence'].isna().sum(),
            'short_sequences': (self.protein_data['seq_length'] < self.config.protein_min_length).sum(),
            'min_length': self.protein_data['seq_length'].min(),
            'max_length': self.protein_data['seq_length'].max(),
            'mean_length': self.protein_data['seq_length'].mean(),
            'missing_uniprot': self.protein_data['UniProt_ID'].isna().sum(),
        }
        
        # Check compound data quality
        compound_stats = {
            'total_rows': len(self.compound_data),
            'missing_smiles': self.compound_data['smiles'].isna().sum(),
            'complex_smiles': (self.compound_data['smiles_length'] > self.config.smiles_max_length).sum(),
            'min_length': self.compound_data['smiles_length'].min(),
            'max_length': self.compound_data['smiles_length'].max(),
            'mean_length': self.compound_data['smiles_length'].mean(),
            'missing_cid': self.compound_data['cid'].isna().sum(),
        }
        
        # Check for data alignment issues
        # Get unique identifiers from each dataset
        kiba_uniprot_ids = set(self.kiba_data['UniProt_ID'].dropna().astype(str).unique())
        kiba_pubchem_ids = set(self.kiba_data['pubchem_cid'].dropna().astype(str).unique())
        protein_uniprot_ids = set(self.protein_data['UniProt_ID'].dropna().astype(str).unique())
        compound_cids = set(self.compound_data['cid'].dropna().astype(str).unique())
        
        # Calculate intersection and missing sets
        uniprot_missing_in_proteins = kiba_uniprot_ids - protein_uniprot_ids
        pubchem_missing_in_compounds = kiba_pubchem_ids - compound_cids
        
        # Calculate potential valid interactions
        potential_valid_proteins = len(kiba_uniprot_ids.intersection(protein_uniprot_ids))
        potential_valid_compounds = len(kiba_pubchem_ids.intersection(compound_cids))
        
        # Add alignment stats
        alignment_stats = {
            'unique_uniprot_in_kiba': len(kiba_uniprot_ids),
            'unique_pubchem_in_kiba': len(kiba_pubchem_ids),
            'unique_uniprot_in_proteins': len(protein_uniprot_ids),
            'unique_cid_in_compounds': len(compound_cids),
            'uniprot_missing_in_proteins': len(uniprot_missing_in_proteins),
            'pubchem_missing_in_compounds': len(pubchem_missing_in_compounds),
            'potential_valid_proteins': potential_valid_proteins,
            'potential_valid_compounds': potential_valid_compounds,
            'uniprot_id_match_percent': 
                (len(kiba_uniprot_ids.intersection(protein_uniprot_ids)) / len(kiba_uniprot_ids) * 100 
                 if len(kiba_uniprot_ids) > 0 else 0),
            'pubchem_id_match_percent': 
                (len(kiba_pubchem_ids.intersection(compound_cids)) / len(kiba_pubchem_ids) * 100
                 if len(kiba_pubchem_ids) > 0 else 0),
        }
        
        # Data types
        data_types = {
            'kiba_uniprot_type': str(self.kiba_data['UniProt_ID'].dtype),
            'kiba_pubchem_type': str(self.kiba_data['pubchem_cid'].dtype),
            'protein_uniprot_type': str(self.protein_data['UniProt_ID'].dtype),
            'compound_cid_type': str(self.compound_data['cid'].dtype)
        }
        
        # Combine all validation stats
        validation_stats = {
            'kiba': kiba_stats,
            'protein': protein_stats,
            'compound': compound_stats,
            'alignment': alignment_stats,
            'data_types': data_types
        }
        
        # Log summary
        logger.info(f"KIBA data issues: {kiba_stats['nan_values']} NaN, {kiba_stats['inf_values']} Inf, "
                    f"{kiba_stats['negative_values']} negative, {kiba_stats['above_threshold']} above threshold")
        logger.info(f"Protein sequences below min length: {protein_stats['short_sequences']} "
                    f"(threshold: {self.config.protein_min_length})")
        logger.info(f"Complex SMILES strings: {compound_stats['complex_smiles']} "
                    f"(threshold: {self.config.smiles_max_length})")
        logger.info(f"ID matching: {alignment_stats['uniprot_id_match_percent']:.1f}% of UniProt IDs match, "
                    f"{alignment_stats['pubchem_id_match_percent']:.1f}% of PubChem IDs match")
        
        # Log ID type information
        logger.info(f"Data types - KIBA: UniProt={data_types['kiba_uniprot_type']}, "
                    f"PubChem={data_types['kiba_pubchem_type']}")
        logger.info(f"Data types - Protein: UniProt={data_types['protein_uniprot_type']}")
        logger.info(f"Data types - Compound: CID={data_types['compound_cid_type']}")
        
        # Warn about potential filtering issues
        if (alignment_stats['uniprot_id_match_percent'] < 50 or 
            alignment_stats['pubchem_id_match_percent'] < 50):
            logger.warning("Less than 50% of IDs match between datasets. "
                          "This may lead to excessive filtering of interactions.")
        
        return validation_stats