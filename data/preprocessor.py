import logging
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Optional, Union, Any

from kiba_model.config import KIBAConfig

logger = logging.getLogger("kiba_model")


class DataPreprocessor:
    """Preprocesses and filters KIBA data.
    
    This class handles data cleaning, filtering, and preparation for feature engineering.
    """
    
    def __init__(self, config: KIBAConfig):
        """Initialize with configuration.
        
        Args:
            config: KIBAConfig object with filtering parameters
        """
        self.config = config
        
    def preprocess_data(
        self, 
        kiba_data: pd.DataFrame, 
        protein_data: pd.DataFrame, 
        compound_data: pd.DataFrame,
        strict_filtering: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Preprocess and filter data based on configured thresholds.
        
        Args:
            kiba_data: DataFrame with KIBA interaction data
            protein_data: DataFrame with protein sequence data
            compound_data: DataFrame with compound SMILES data
            strict_filtering: Whether to use strict filtering criteria
            
        Returns:
            Tuple of filtered (interactions, proteins, compounds) DataFrames
        """
        filtering_mode = "strict" if strict_filtering else "lenient"
        logger.info(f"Preprocessing and filtering data (mode: {filtering_mode})...")
        
        # Track original counts for reporting
        original_interactions = len(kiba_data)
        original_proteins = len(protein_data)
        original_compounds = len(compound_data)
        
        # Make copies to avoid modifying original data
        kiba_data = kiba_data.copy()
        protein_data = protein_data.copy()
        compound_data = compound_data.copy()
        
        # 1. Calculate lengths if not already done
        if 'seq_length' not in protein_data.columns:
            protein_data['seq_length'] = protein_data['Protein_Sequence'].apply(len)
            logger.info(f"Added protein sequence length column")
            
        if 'smiles_length' not in compound_data.columns:
            compound_data['smiles_length'] = compound_data['smiles'].apply(len)
            logger.info(f"Added SMILES length column")
        
        # 2. Apply protein sequence length filters
        # In lenient mode, adjust the thresholds if needed
        protein_min_length = self.config.protein_min_length
        protein_max_length = self.config.protein_max_length
        
        if not strict_filtering:
            # Try more lenient thresholds if we'd have too few proteins
            if len(protein_data[(protein_data['seq_length'] >= protein_min_length) & 
                                (protein_data['seq_length'] <= protein_max_length)]) < 10:
                protein_min_length = max(50, protein_min_length // 2)
                protein_max_length = min(5000, protein_max_length * 2)
                logger.warning(f"Adjusting protein length range to {protein_min_length}-{protein_max_length} in lenient mode")
        
        # Apply both min and max filters
        valid_proteins = protein_data[
            (protein_data['seq_length'] >= protein_min_length) & 
            (protein_data['seq_length'] <= protein_max_length)
        ].copy()
        
        logger.info(f"Filtered proteins by length ({protein_min_length}-{protein_max_length}): "
                    f"kept {len(valid_proteins)}/{original_proteins} "
                    f"({len(valid_proteins)/original_proteins*100:.2f}%)")
        
        # Log protein length distribution after filtering
        if len(valid_proteins) > 0:
            logger.info(f"Protein length after filtering - min: {valid_proteins['seq_length'].min()}, "
                        f"max: {valid_proteins['seq_length'].max()}, "
                        f"mean: {valid_proteins['seq_length'].mean():.1f}, "
                        f"median: {valid_proteins['seq_length'].median()}")
        
        # 3. Apply SMILES complexity filter
        # In lenient mode, adjust the threshold upward if needed
        smiles_max_length = self.config.smiles_max_length
        if not strict_filtering and len(compound_data[compound_data['smiles_length'] <= smiles_max_length]) < 10:
            # Try a more lenient threshold if we'd have too few compounds
            smiles_max_length = min(300, smiles_max_length * 2)
            logger.warning(f"Adjusting SMILES max length to {smiles_max_length} in lenient mode")
            
        valid_compounds = compound_data[compound_data['smiles_length'] <= smiles_max_length].copy()
        logger.info(f"Filtered compounds: kept {len(valid_compounds)}/{original_compounds} "
                    f"({len(valid_compounds)/original_compounds*100:.2f}%)")
    
        # 4. Clean KIBA data
        interactions_clean = kiba_data.copy()
        
        # 4.1 Convert data to appropriate types first to avoid comparison issues
        # Handle missing values
        interactions_clean = interactions_clean.dropna(subset=['kiba_score', 'UniProt_ID', 'pubchem_cid'])
        logger.info(f"Dropped rows with missing values: kept {len(interactions_clean)}/{original_interactions} "
                   f"({len(interactions_clean)/original_interactions*100:.2f}%)")
        
        # 4.2 Add experimental flag to KIBA data
        # Make sure kiba_score_estimated is boolean
        if interactions_clean['kiba_score_estimated'].dtype == 'object':
            # Convert string representations to boolean
            interactions_clean['kiba_score_estimated'] = interactions_clean['kiba_score_estimated'].map(
                {'True': True, 'False': False, True: True, False: False})
            
        # Handle case where it might be numeric
        if interactions_clean['kiba_score_estimated'].dtype in ['int64', 'float64']:
            interactions_clean['kiba_score_estimated'] = interactions_clean['kiba_score_estimated'].astype(bool)
            
        # Now create is_experimental flag (1 for experimental, 0 for estimated)
        interactions_clean['is_experimental'] = (~interactions_clean['kiba_score_estimated']).astype(int)
        logger.info(f"Added is_experimental flag: "
                    f"{interactions_clean['is_experimental'].sum()} experimental, "
                    f"{len(interactions_clean) - interactions_clean['is_experimental'].sum()} estimated")
            
        # 4.3 Remove rows with invalid KIBA scores
        pre_count = len(interactions_clean)
        
        # Handle NaN and Inf values (should be redundant after dropna above, but being thorough)
        interactions_clean = interactions_clean[~np.isnan(interactions_clean['kiba_score'])]
        interactions_clean = interactions_clean[~np.isinf(interactions_clean['kiba_score'])]
        
        # Log the filtering
        post_count = len(interactions_clean)
        if pre_count > post_count:
            logger.info(f"Removed {pre_count - post_count} rows with NaN or Inf KIBA scores")
        
        # 4.4 Clip negative values to small positive number
        negative_mask = interactions_clean['kiba_score'] < 0
        if negative_mask.sum() > 0:
            interactions_clean.loc[negative_mask, 'kiba_score'] = 1e-4
            logger.info(f"Clipped {negative_mask.sum()} negative KIBA scores to 1e-4")
        
        # 4.5 Apply KIBA score threshold
        if self.config.kiba_score_threshold is not None:
            kiba_threshold = self.config.kiba_score_threshold
            pre_count = len(interactions_clean)
            interactions_clean = interactions_clean[interactions_clean['kiba_score'] < kiba_threshold]
            post_count = len(interactions_clean)
            logger.info(f"Applied KIBA score threshold <{kiba_threshold}: removed {pre_count - post_count} rows "
                        f"({(pre_count - post_count)/pre_count*100:.2f}% of the data)")
            
        # 5. Convert ID columns to string for consistent matching
        # The key improvement here is to ensure all IDs are treated as strings
        # to avoid type mismatch in joins
        # Convert IDs to string with matching formats, ensuring consistency
        try:
            # Convert pubchem_cid to integer first (removes decimal part) then to string
            interactions_clean['pubchem_cid_str'] = interactions_clean['pubchem_cid'].astype(int).astype(str)
        except Exception as e:
            logger.warning(f"Error converting pubchem_cid: {str(e)}. Trying alternate approach.")
            # If that fails, try a different approach
            interactions_clean['pubchem_cid_str'] = interactions_clean['pubchem_cid'].fillna(-1).astype(float).astype(int).astype(str)
            # Replace -1 with empty string
            interactions_clean.loc[interactions_clean['pubchem_cid_str'] == '-1', 'pubchem_cid_str'] = ''

        interactions_clean['UniProt_ID_str'] = interactions_clean['UniProt_ID'].astype(str)
        valid_proteins['UniProt_ID_str'] = valid_proteins['UniProt_ID'].astype(str)
        valid_compounds['cid_str'] = valid_compounds['cid'].astype(str)

        # Check for matches before filtering
        comp_ids_in_kiba = set(interactions_clean['pubchem_cid_str'])
        comp_ids_in_compounds = set(valid_compounds['cid_str'])
        overlap = comp_ids_in_kiba.intersection(comp_ids_in_compounds)
        logger.info(f"CID matching: {len(overlap)} out of {len(comp_ids_in_kiba)} IDs in KIBA file match compound IDs ({len(overlap)/len(comp_ids_in_kiba)*100:.1f}%)")

        logger.info(f"Converted ID columns to string type for consistent matching")
        
        # 6. Filter interactions to only include valid proteins and compounds
        pre_count = len(interactions_clean)
        
        # Check if proteins or compounds might be empty after filtering
        if len(valid_proteins) == 0:
            logger.error("No valid proteins after filtering!")
            if strict_filtering and self.config.fallback_to_lenient_filtering:
                logger.info("Will retry with lenient filtering")
                return self.preprocess_data(kiba_data, protein_data, compound_data, strict_filtering=False)
            elif self.config.allow_empty_results:
                logger.warning("Continuing with empty results as configured")
                valid_proteins = protein_data.head(0).copy()  # Empty DataFrame with same structure
            else:
                raise ValueError("No valid proteins after filtering")
                
        if len(valid_compounds) == 0:
            logger.error("No valid compounds after filtering!")
            if strict_filtering and self.config.fallback_to_lenient_filtering:
                logger.info("Will retry with lenient filtering")
                return self.preprocess_data(kiba_data, protein_data, compound_data, strict_filtering=False)
            elif self.config.allow_empty_results:
                logger.warning("Continuing with empty results as configured")
                valid_compounds = compound_data.head(0).copy()  # Empty DataFrame with same structure
            else:
                raise ValueError("No valid compounds after filtering")
        
        # Count matches before filtering to identify potential issues
        valid_protein_ids = set(valid_proteins['UniProt_ID_str'])
        valid_compound_ids = set(valid_compounds['cid_str'])
        
        protein_matches = sum(interactions_clean['UniProt_ID_str'].isin(valid_protein_ids))
        compound_matches = sum(interactions_clean['pubchem_cid_str'].isin(valid_compound_ids))
        
        logger.info(f"Potential matches: {protein_matches} proteins, {compound_matches} compounds")
        
        # Apply the filtering
        valid_interactions = interactions_clean[
            interactions_clean['UniProt_ID_str'].isin(valid_protein_ids) & 
            interactions_clean['pubchem_cid_str'].isin(valid_compound_ids)
        ].copy()
        
        # Log the filtering results
        post_count = len(valid_interactions)
        logger.info(f"Filtered interactions: kept {post_count}/{pre_count} " 
                    f"({post_count/pre_count*100:.2f}% of clean interactions, "
                    f"{post_count/original_interactions*100:.2f}% of original interactions)")
        
        # Check if we have too few valid interactions
        if post_count < self.config.min_valid_interactions:
            logger.warning(f"Only {post_count} valid interactions after filtering "
                          f"(minimum required: {self.config.min_valid_interactions})")
            
            if strict_filtering and self.config.fallback_to_lenient_filtering:
                logger.info("Trying again with lenient filtering...")
                return self.preprocess_data(kiba_data, protein_data, compound_data, strict_filtering=False)
                
            elif not self.config.allow_empty_results:
                raise ValueError(f"Too few valid interactions after filtering: {post_count}")
        
        # 7. Create strata if stratification is enabled
        if self.config.use_stratification:
            valid_interactions['strata'] = valid_interactions.apply(self._create_strata, axis=1)
            strata_counts = valid_interactions['strata'].value_counts()
            logger.info(f"Created {len(strata_counts)} strata for balanced sampling")
            logger.debug(f"Strata distribution:\n{strata_counts}")
            
        # 8. Save filtered data
        valid_proteins.to_csv(self.config.filtered_proteins_file, index=False)
        valid_compounds.to_csv(self.config.filtered_compounds_file, index=False)
        valid_interactions.to_csv(self.config.filtered_interactions_file, index=False)
        
        logger.info(f"Saved filtered data to {self.config.data_dir}")
        logger.info(f"  {len(valid_interactions)} interactions")
        logger.info(f"  {len(valid_proteins)} proteins")
        logger.info(f"  {len(valid_compounds)} compounds")
        
        return valid_interactions, valid_proteins, valid_compounds
    
    def _create_strata(self, row: pd.Series) -> str:
        """Create stratification label based on experimental status and KIBA score range.
        
        Args:
            row: DataFrame row with 'is_experimental' and 'kiba_score'
            
        Returns:
            String label for stratification
        """
        # Determine if experimental
        is_exp = "exp" if row['is_experimental'] == 1 else "est"
        
        # Determine score range
        if row['kiba_score'] < 25:
            score_range = "low"
        elif row['kiba_score'] < 100:
            score_range = "med"
        else:
            score_range = "high"
        
        return f"{is_exp}_{score_range}"
