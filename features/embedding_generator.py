import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import h5py
import requests
import pandas as pd
import os
from tqdm import tqdm
import logging
import time
import gc
from typing import List, Dict, Tuple, Optional, Union

logger = logging.getLogger("kiba_model")

class EmbeddingGenerator:
    """Class to generate embeddings for proteins and compounds with batch processing support."""
    
    def __init__(self, config):
        """Initialize with configuration."""
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() and self.config.gpu_enabled else "cpu")
        logger.info(f"Using device: {self.device}")
        
    def generate_protein_embeddings(self, proteins_df=None, uniprot_ids=None, batch_size=50, resume=True):
        """
        Generate ESM embeddings for proteins in batches with checkpoint support.
        
        Args:
            proteins_df: DataFrame with UniProt_ID and Protein_Sequence columns
            uniprot_ids: List of UniProt IDs to fetch and embed (if proteins_df not provided)
            batch_size: Number of proteins to process in each batch
            resume: Whether to resume from existing checkpoints
            
        Returns:
            Tuple of (embeddings array, protein_ids list)
        """
        logger.info("Generating protein embeddings with ESM in batches")
        
        # If no DataFrame provided but UniProt IDs given, fetch sequences
        if proteins_df is None and uniprot_ids is not None:
            proteins_df = self._fetch_protein_sequences(uniprot_ids)
        
        if proteins_df is None or len(proteins_df) == 0:
            raise ValueError("No protein data provided for embedding generation")
        
        # Setup checkpoint directory
        checkpoint_dir = os.path.join(os.path.dirname(self.config.protein_embeddings_file), "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Check for existing final output file
        if os.path.exists(self.config.protein_embeddings_file) and resume:
            logger.info(f"Final protein embeddings file already exists: {self.config.protein_embeddings_file}")
            try:
                with h5py.File(self.config.protein_embeddings_file, 'r') as f:
                    embeddings = f['embeddings'][:]
                    protein_ids_bytes = f['protein_ids'][:]
                    
                    # Convert bytes to strings if needed
                    protein_ids = []
                    for pid_bytes in protein_ids_bytes:
                        if isinstance(pid_bytes, bytes):
                            protein_ids.append(pid_bytes.decode('utf-8'))
                        else:
                            protein_ids.append(str(pid_bytes))
                
                logger.info(f"Loaded {len(protein_ids)} existing protein embeddings")
                return embeddings, protein_ids
            except Exception as e:
                logger.warning(f"Error loading existing embeddings file: {str(e)}")
                logger.info("Will regenerate embeddings")
        
        # Look for checkpoints
        processed_proteins = set()
        all_embeddings = []
        all_protein_ids = []
        
        if resume:
            checkpoint_pattern = os.path.join(checkpoint_dir, "protein_batch_*.h5")
            checkpoint_files = sorted([f for f in os.listdir(checkpoint_dir) 
                                     if f.startswith("protein_batch_") and f.endswith(".h5")])
            
            if checkpoint_files:
                logger.info(f"Found {len(checkpoint_files)} checkpoint files")
                
                # Load checkpoints
                for checkpoint_file in checkpoint_files:
                    try:
                        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
                        logger.info(f"Loading checkpoint: {checkpoint_path}")
                        
                        with h5py.File(checkpoint_path, 'r') as f:
                            batch_embeddings = f['embeddings'][:]
                            batch_ids = [pid.decode('utf-8') if isinstance(pid, bytes) else str(pid) 
                                        for pid in f['protein_ids'][:]]
                            
                            all_embeddings.append(batch_embeddings)
                            all_protein_ids.extend(batch_ids)
                            processed_proteins.update(batch_ids)
                        
                        logger.info(f"Loaded {len(batch_ids)} proteins from checkpoint")
                    except Exception as e:
                        logger.warning(f"Error loading checkpoint {checkpoint_file}: {str(e)}")
        
        # Filter out already processed proteins
        if processed_proteins:
            logger.info(f"Already processed {len(processed_proteins)} proteins")
            remaining_df = proteins_df[~proteins_df['UniProt_ID'].astype(str).isin(processed_proteins)]
            logger.info(f"Remaining proteins to process: {len(remaining_df)}")
        else:
            remaining_df = proteins_df
            
        # Check if we have any proteins left to process
        if len(remaining_df) == 0:
            if all_embeddings:
                logger.info("All proteins already processed, combining checkpoints")
                # Combine all checkpoint data
                combined_embeddings = np.vstack(all_embeddings)
                
                # Save to final output file
                with h5py.File(self.config.protein_embeddings_file, 'w') as f:
                    f.create_dataset('embeddings', data=combined_embeddings)
                    dt = h5py.special_dtype(vlen=str)
                    protein_id_dataset = f.create_dataset('protein_ids', shape=(len(all_protein_ids),), dtype=dt)
                    protein_id_dataset[:] = all_protein_ids
                
                logger.info(f"Saved combined embeddings for {len(all_protein_ids)} proteins")
                return combined_embeddings, all_protein_ids
            else:
                logger.warning("No proteins to process and no checkpoints found")
                return np.array([]), []
        
        # Load ESM model
        logger.info("Loading ESM model")
        model_name = "facebook/esm2_t12_35M_UR50D"  # Medium-sized model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model = model.to(self.device)
        model.eval()
        
        # Process remaining proteins in batches
        start_time = time.time()
        total_processed = len(processed_proteins)
        total_to_process = len(remaining_df)
        batch_counter = len(all_embeddings)
        
        for batch_start in range(0, len(remaining_df), batch_size):
            batch_end = min(batch_start + batch_size, len(remaining_df))
            batch_df = remaining_df.iloc[batch_start:batch_end]
            
            logger.info(f"Processing batch {batch_counter+1}: proteins {batch_start}-{batch_end} of {len(remaining_df)}")
            
            batch_embeddings = []
            batch_ids = []
            
            # Process each protein in the batch
            for _, row in tqdm(batch_df.iterrows(), total=len(batch_df), desc="Processing proteins"):
                try:
                    protein_id = str(row['UniProt_ID'])
                    sequence = row['Protein_Sequence']
                    
                    # Skip if sequence is too long
                    if len(sequence) > 1022:  # ESM-2 context limit
                        logger.warning(f"Protein {protein_id} sequence too long ({len(sequence)}), truncating")
                        sequence = sequence[:1022]
                    
                    # Tokenize and get embeddings
                    inputs = tokenizer(sequence, return_tensors="pt").to(self.device)
                    
                    with torch.no_grad():
                        outputs = model(**inputs)
                    
                    # Average pooling over sequence length
                    embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()[0]
                    
                    batch_embeddings.append(embedding)
                    batch_ids.append(protein_id)
                    
                except Exception as e:
                    logger.error(f"Error processing protein {protein_id}: {str(e)}")
            
            if batch_embeddings:
                # Convert to numpy array
                batch_embeddings_array = np.array(batch_embeddings)
                
                # Save batch checkpoint
                checkpoint_path = os.path.join(checkpoint_dir, f"protein_batch_{batch_counter}.h5")
                with h5py.File(checkpoint_path, 'w') as f:
                    f.create_dataset('embeddings', data=batch_embeddings_array)
                    dt = h5py.special_dtype(vlen=str)
                    batch_id_dataset = f.create_dataset('protein_ids', shape=(len(batch_ids),), dtype=dt)
                    batch_id_dataset[:] = batch_ids
                
                # Add to overall results
                all_embeddings.append(batch_embeddings_array)
                all_protein_ids.extend(batch_ids)
                
                # Update progress
                total_processed += len(batch_ids)
                current_time = time.time()
                elapsed = current_time - start_time
                proteins_per_second = total_processed / elapsed if elapsed > 0 else 0
                estimated_total = total_to_process / proteins_per_second if proteins_per_second > 0 else 0
                remaining = estimated_total - elapsed
                
                logger.info(f"Batch {batch_counter+1} complete: {len(batch_ids)} proteins processed")
                logger.info(f"Progress: {total_processed}/{total_processed + total_to_process} proteins "
                           f"({total_processed/(total_processed + total_to_process)*100:.1f}%)")
                logger.info(f"Time elapsed: {elapsed/60:.1f} minutes, estimated remaining: {remaining/60:.1f} minutes")
                
                batch_counter += 1
                
                # Free up memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
        
        # Combine all batches
        if all_embeddings:
            combined_embeddings = np.vstack(all_embeddings)
            
            # Save to final output file
            with h5py.File(self.config.protein_embeddings_file, 'w') as f:
                f.create_dataset('embeddings', data=combined_embeddings)
                dt = h5py.special_dtype(vlen=str)
                protein_id_dataset = f.create_dataset('protein_ids', shape=(len(all_protein_ids),), dtype=dt)
                protein_id_dataset[:] = all_protein_ids
            
            logger.info(f"Generated and saved embeddings for {len(all_protein_ids)} proteins")
            logger.info(f"Output file: {self.config.protein_embeddings_file}")
            
            return combined_embeddings, all_protein_ids
        else:
            logger.warning("No embeddings were generated")
            return np.array([]), []
    
    def generate_compound_embeddings(self, compounds_df=None, compound_ids=None, batch_size=50, resume=True):
        """
        Generate ChemBERTa embeddings for compounds in batches with checkpoint support.
        
        Args:
            compounds_df: DataFrame with cid and smiles columns
            compound_ids: List of PubChem CIDs to fetch and embed (if compounds_df not provided)
            batch_size: Number of compounds to process in each batch
            resume: Whether to resume from existing checkpoints
            
        Returns:
            Tuple of (embeddings array, compound_ids list)
        """
        logger.info("Generating compound embeddings with ChemBERTa in batches")
        
        # If no DataFrame provided but compound IDs given, fetch SMILES
        if compounds_df is None and compound_ids is not None:
            compounds_df = self._fetch_compound_smiles(compound_ids)
        
        if compounds_df is None or len(compounds_df) == 0:
            raise ValueError("No compound data provided for embedding generation")
        
        # Setup checkpoint directory
        checkpoint_dir = os.path.join(os.path.dirname(self.config.compound_embeddings_file), "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Check for existing final output file
        if os.path.exists(self.config.compound_embeddings_file) and resume:
            logger.info(f"Final compound embeddings file already exists: {self.config.compound_embeddings_file}")
            try:
                with h5py.File(self.config.compound_embeddings_file, 'r') as f:
                    embeddings = f['embeddings'][:]
                    compound_ids_bytes = f['cids'][:]
                    
                    # Convert bytes to strings if needed
                    compound_ids = []
                    for cid_bytes in compound_ids_bytes:
                        if isinstance(cid_bytes, bytes):
                            compound_ids.append(cid_bytes.decode('utf-8'))
                        else:
                            compound_ids.append(str(cid_bytes))
                
                logger.info(f"Loaded {len(compound_ids)} existing compound embeddings")
                return embeddings, compound_ids
            except Exception as e:
                logger.warning(f"Error loading existing embeddings file: {str(e)}")
                logger.info("Will regenerate embeddings")
        
        # Look for checkpoints
        processed_compounds = set()
        all_embeddings = []
        all_compound_ids = []
        
        if resume:
            checkpoint_files = sorted([f for f in os.listdir(checkpoint_dir) 
                                     if f.startswith("compound_batch_") and f.endswith(".h5")])
            
            if checkpoint_files:
                logger.info(f"Found {len(checkpoint_files)} checkpoint files")
                
                # Load checkpoints
                for checkpoint_file in checkpoint_files:
                    try:
                        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
                        logger.info(f"Loading checkpoint: {checkpoint_path}")
                        
                        with h5py.File(checkpoint_path, 'r') as f:
                            batch_embeddings = f['embeddings'][:]
                            batch_ids = [cid.decode('utf-8') if isinstance(cid, bytes) else str(cid) 
                                        for cid in f['cids'][:]]
                            
                            all_embeddings.append(batch_embeddings)
                            all_compound_ids.extend(batch_ids)
                            processed_compounds.update(batch_ids)
                        
                        logger.info(f"Loaded {len(batch_ids)} compounds from checkpoint")
                    except Exception as e:
                        logger.warning(f"Error loading checkpoint {checkpoint_file}: {str(e)}")
        
        # Filter out already processed compounds
        if processed_compounds:
            logger.info(f"Already processed {len(processed_compounds)} compounds")
            remaining_df = compounds_df[~compounds_df['cid'].astype(str).isin(processed_compounds)]
            logger.info(f"Remaining compounds to process: {len(remaining_df)}")
        else:
            remaining_df = compounds_df
            
        # Check if we have any compounds left to process
        if len(remaining_df) == 0:
            if all_embeddings:
                logger.info("All compounds already processed, combining checkpoints")
                # Combine all checkpoint data
                combined_embeddings = np.vstack(all_embeddings)
                
                # Save to final output file
                with h5py.File(self.config.compound_embeddings_file, 'w') as f:
                    f.create_dataset('embeddings', data=combined_embeddings)
                    dt = h5py.special_dtype(vlen=str)
                    cid_dataset = f.create_dataset('cids', shape=(len(all_compound_ids),), dtype=dt)
                    cid_dataset[:] = all_compound_ids
                
                logger.info(f"Saved combined embeddings for {len(all_compound_ids)} compounds")
                return combined_embeddings, all_compound_ids
            else:
                logger.warning("No compounds to process and no checkpoints found")
                return np.array([]), []
        
        # Load ChemBERTa model
        logger.info("Loading ChemBERTa model")
        model_name = "seyonec/ChemBERTa-zinc-base-v1"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model = model.to(self.device)
        model.eval()
        
        # Process remaining compounds in batches
        start_time = time.time()
        total_processed = len(processed_compounds)
        total_to_process = len(remaining_df)
        batch_counter = len(all_embeddings)
        
        for batch_start in range(0, len(remaining_df), batch_size):
            batch_end = min(batch_start + batch_size, len(remaining_df))
            batch_df = remaining_df.iloc[batch_start:batch_end]
            
            logger.info(f"Processing batch {batch_counter+1}: compounds {batch_start}-{batch_end} of {len(remaining_df)}")
            
            batch_embeddings = []
            batch_ids = []
            
            # Process each compound in the batch
            for _, row in tqdm(batch_df.iterrows(), total=len(batch_df), desc="Processing compounds"):
                try:
                    compound_id = str(row['cid'])
                    smiles = row['smiles']
                    
                    # Skip if SMILES is too long
                    if len(smiles) > 512:  # ChemBERTa context limit
                        logger.warning(f"Compound {compound_id} SMILES too long ({len(smiles)}), truncating")
                        smiles = smiles[:512]
                    
                    # Tokenize and get embeddings
                    inputs = tokenizer(smiles, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
                    
                    with torch.no_grad():
                        outputs = model(**inputs)
                    
                    # Get CLS token embedding
                    embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
                    
                    batch_embeddings.append(embedding)
                    batch_ids.append(compound_id)
                    
                except Exception as e:
                    logger.error(f"Error processing compound {compound_id}: {str(e)}")
            
            if batch_embeddings:
                # Convert to numpy array
                batch_embeddings_array = np.array(batch_embeddings)
                
                # Save batch checkpoint
                checkpoint_path = os.path.join(checkpoint_dir, f"compound_batch_{batch_counter}.h5")
                with h5py.File(checkpoint_path, 'w') as f:
                    f.create_dataset('embeddings', data=batch_embeddings_array)
                    dt = h5py.special_dtype(vlen=str)
                    batch_id_dataset = f.create_dataset('cids', shape=(len(batch_ids),), dtype=dt)
                    batch_id_dataset[:] = batch_ids
                
                # Add to overall results
                all_embeddings.append(batch_embeddings_array)
                all_compound_ids.extend(batch_ids)
                
                # Update progress
                total_processed += len(batch_ids)
                current_time = time.time()
                elapsed = current_time - start_time
                compounds_per_second = total_processed / elapsed if elapsed > 0 else 0
                estimated_total = total_to_process / compounds_per_second if compounds_per_second > 0 else 0
                remaining = estimated_total - elapsed
                
                logger.info(f"Batch {batch_counter+1} complete: {len(batch_ids)} compounds processed")
                logger.info(f"Progress: {total_processed}/{total_processed + total_to_process} compounds "
                           f"({total_processed/(total_processed + total_to_process)*100:.1f}%)")
                logger.info(f"Time elapsed: {elapsed/60:.1f} minutes, estimated remaining: {remaining/60:.1f} minutes")
                
                batch_counter += 1
                
                # Free up memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
        
        # Combine all batches
        if all_embeddings:
            combined_embeddings = np.vstack(all_embeddings)
            
            # Save to final output file
            with h5py.File(self.config.compound_embeddings_file, 'w') as f:
                f.create_dataset('embeddings', data=combined_embeddings)
                dt = h5py.special_dtype(vlen=str)
                cid_dataset = f.create_dataset('cids', shape=(len(all_compound_ids),), dtype=dt)
                cid_dataset[:] = all_compound_ids
            
            logger.info(f"Generated and saved embeddings for {len(all_compound_ids)} compounds")
            logger.info(f"Output file: {self.config.compound_embeddings_file}")
            
            return combined_embeddings, all_compound_ids
        else:
            logger.warning("No embeddings were generated")
            return np.array([]), []
    
    def _fetch_protein_sequences(self, uniprot_ids: List[str]) -> pd.DataFrame:
        """Fetch protein sequences from UniProt API."""
        logger.info(f"Fetching {len(uniprot_ids)} protein sequences from UniProt")
        
        sequences = []
        ids = []
        
        for uniprot_id in tqdm(uniprot_ids, desc="Fetching proteins"):
            try:
                url = f"https://www.uniprot.org/uniprot/{uniprot_id}.fasta"
                response = requests.get(url)
                
                if response.status_code == 200:
                    # Parse FASTA
                    lines = response.text.strip().split('\n')
                    sequence = ''.join(lines[1:])  # Skip header
                    
                    sequences.append(sequence)
                    ids.append(uniprot_id)
                else:
                    logger.warning(f"Failed to fetch sequence for {uniprot_id}: {response.status_code}")
            except Exception as e:
                logger.error(f"Error fetching sequence for {uniprot_id}: {str(e)}")
        
        # Create DataFrame
        df = pd.DataFrame({
            'UniProt_ID': ids,
            'Protein_Sequence': sequences
        })
        
        return df
    
    def _fetch_compound_smiles(self, compound_ids: List[str]) -> pd.DataFrame:
        """Fetch compound SMILES from PubChem API."""
        logger.info(f"Fetching {len(compound_ids)} compound SMILES from PubChem")
        
        smiles_list = []
        ids = []
        
        for compound_id in tqdm(compound_ids, desc="Fetching compounds"):
            try:
                url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{compound_id}/property/CanonicalSMILES/JSON"
                response = requests.get(url)
                
                if response.status_code == 200:
                    data = response.json()
                    smiles = data['PropertyTable']['Properties'][0]['CanonicalSMILES']
                    
                    smiles_list.append(smiles)
                    ids.append(compound_id)
                else:
                    logger.warning(f"Failed to fetch SMILES for {compound_id}: {response.status_code}")
            except Exception as e:
                logger.error(f"Error fetching SMILES for {compound_id}: {str(e)}")
        
        # Create DataFrame
        df = pd.DataFrame({
            'cid': ids,
            'smiles': smiles_list
        })
        
        return df