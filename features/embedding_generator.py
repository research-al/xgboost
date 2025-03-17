# kiba_model/features/embedding_generator.py
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import h5py
import requests
import pandas as pd
from tqdm import tqdm
import logging
from typing import List, Dict, Tuple

logger = logging.getLogger("kiba_model")

class EmbeddingGenerator:
    """Class to generate embeddings for proteins and compounds when they are not pre-computed."""
    
    def __init__(self, config):
        """Initialize with configuration."""
        self.config = config
        
    def generate_protein_embeddings(self, proteins_df=None, uniprot_ids=None):
        """
        Generate ESM embeddings for proteins.
        
        Args:
            proteins_df: DataFrame with UniProt_ID and Protein_Sequence columns
            uniprot_ids: List of UniProt IDs to fetch and embed (if proteins_df not provided)
        """
        logger.info("Generating protein embeddings with ESM")
        
        # If no DataFrame provided but UniProt IDs given, fetch sequences
        if proteins_df is None and uniprot_ids is not None:
            proteins_df = self._fetch_protein_sequences(uniprot_ids)
        
        if proteins_df is None or len(proteins_df) == 0:
            raise ValueError("No protein data provided for embedding generation")
            
        # Load ESM model
        model_name = "facebook/esm2_t12_35M_UR50D"  # Medium-sized model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        
        # Use GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() and self.config.gpu_enabled else "cpu")
        model = model.to(device)
        model.eval()
        
        # Generate embeddings
        embeddings = []
        protein_ids = []
        
        for idx, row in tqdm(proteins_df.iterrows(), total=len(proteins_df), desc="Generating protein embeddings"):
            protein_id = str(row['UniProt_ID'])
            sequence = row['Protein_Sequence']
            
            # Skip if sequence is too long (depends on model's max context)
            if len(sequence) > 1022:  # ESM-2 has a context limit
                logger.warning(f"Protein {protein_id} sequence too long ({len(sequence)}), truncating")
                sequence = sequence[:1022]
                
            # Tokenize
            inputs = tokenizer(sequence, return_tensors="pt").to(device)
            
            # Get embeddings
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Average pooling over sequence length (simple approach)
            embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()[0]
            
            embeddings.append(embedding)
            protein_ids.append(protein_id)
        
        # Save to H5 file
        with h5py.File(self.config.protein_embeddings_file, 'w') as f:
            f.create_dataset('embeddings', data=np.array(embeddings))
            dt = h5py.special_dtype(vlen=str)
            protein_id_dataset = f.create_dataset('protein_ids', shape=(len(protein_ids),), dtype=dt)
            protein_id_dataset[:] = protein_ids
            
        logger.info(f"Generated and saved {len(protein_ids)} protein embeddings to {self.config.protein_embeddings_file}")
        
        return np.array(embeddings), protein_ids
    
    def generate_compound_embeddings(self, compounds_df=None, compound_ids=None):
        """
        Generate ChemBERTa embeddings for compounds.
        
        Args:
            compounds_df: DataFrame with cid and smiles columns
            compound_ids: List of PubChem CIDs to fetch and embed (if compounds_df not provided)
        """
        logger.info("Generating compound embeddings with ChemBERTa")
        
        # If no DataFrame provided but compound IDs given, fetch SMILES
        if compounds_df is None and compound_ids is not None:
            compounds_df = self._fetch_compound_smiles(compound_ids)
        
        if compounds_df is None or len(compounds_df) == 0:
            raise ValueError("No compound data provided for embedding generation")
            
        # Load ChemBERTa model
        model_name = "seyonec/ChemBERTa-zinc-base-v1"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        
        # Use GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() and self.config.gpu_enabled else "cpu")
        model = model.to(device)
        model.eval()
        
        # Generate embeddings
        embeddings = []
        compound_ids = []
        
        for idx, row in tqdm(compounds_df.iterrows(), total=len(compounds_df), desc="Generating compound embeddings"):
            compound_id = str(row['cid'])
            smiles = row['smiles']
            
            # Skip if SMILES is too long
            if len(smiles) > 512:  # ChemBERTa context limit
                logger.warning(f"Compound {compound_id} SMILES too long ({len(smiles)}), truncating")
                smiles = smiles[:512]
                
            # Tokenize
            inputs = tokenizer(smiles, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            
            # Get embeddings
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Get CLS token embedding (standard for BERT-like models)
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
            
            embeddings.append(embedding)
            compound_ids.append(compound_id)
        
        # Save to H5 file
        with h5py.File(self.config.compound_embeddings_file, 'w') as f:
            f.create_dataset('embeddings', data=np.array(embeddings))
            dt = h5py.special_dtype(vlen=str)
            cid_dataset = f.create_dataset('cids', shape=(len(compound_ids),), dtype=dt)
            cid_dataset[:] = compound_ids
            
        logger.info(f"Generated and saved {len(compound_ids)} compound embeddings to {self.config.compound_embeddings_file}")
        
        return np.array(embeddings), compound_ids
    
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