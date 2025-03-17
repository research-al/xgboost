#!/usr/bin/env python
"""Script to generate all protein and compound embeddings in batches."""

import os
import sys
import time
import logging
import traceback
from datetime import datetime
import argparse
import pandas as pd
import numpy as np
import h5py
from pathlib import Path

# Add parent directory to sys.path if needed
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from kiba_model.utils.logging import setup_logging
from kiba_model.config import KIBAConfig
from kiba_model.features.embedding_generator import EmbeddingGenerator


def main():
    """Generate embeddings for all proteins and compounds."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate embeddings for all proteins and compounds')
    
    # Required arguments
    parser.add_argument('--protein-file', type=str, required=True, 
                        help='Path to protein sequences file')
    parser.add_argument('--compound-file', type=str, required=True, 
                        help='Path to compound SMILES file')
    
    # Optional arguments
    parser.add_argument('--data-dir', type=str, default='data', 
                        help='Directory for data files')
    parser.add_argument('--logs-dir', type=str, default='logs', 
                        help='Directory for log files')
    parser.add_argument('--batch-size', type=int, default=50, 
                        help='Number of proteins/compounds to process in each batch')
    parser.add_argument('--no-gpu', action='store_true', 
                        help='Disable GPU acceleration')
    parser.add_argument('--no-resume', action='store_true', 
                        help='Do not resume from existing checkpoints')
    parser.add_argument('--only-proteins', action='store_true', 
                        help='Only generate protein embeddings')
    parser.add_argument('--only-compounds', action='store_true', 
                        help='Only generate compound embeddings')
    parser.add_argument('--protein-output', type=str, 
                        help='Custom output file for protein embeddings')
    parser.add_argument('--compound-output', type=str, 
                        help='Custom output file for compound embeddings')
    parser.add_argument('--verbose', action='store_true', 
                        help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logging(args.logs_dir, level=log_level)
    
    # Create config
    config = KIBAConfig(
        data_dir=args.data_dir,
        logs_dir=args.logs_dir,
        gpu_enabled=not args.no_gpu
    )
    
    # Override embedding file paths if custom outputs specified
    if args.protein_output:
        config.protein_embeddings_file = args.protein_output
    
    if args.compound_output:
        config.compound_embeddings_file = args.compound_output
    
    # Log configuration
    logger.info("Embedding Generation Configuration:")
    logger.info(f"  Protein file: {args.protein_file}")
    logger.info(f"  Compound file: {args.compound_file}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Data directory: {args.data_dir}")
    logger.info(f"  GPU enabled: {not args.no_gpu}")
    logger.info(f"  Resume from checkpoints: {not args.no_resume}")
    
    # Create embedding generator
    embedding_generator = EmbeddingGenerator(config)
    
    try:
        # Process proteins if not skipped
        if not args.only_compounds:
            logger.info("======= Starting Protein Embedding Generation =======")
            
            # Load protein data
            logger.info(f"Loading protein data from {args.protein_file}")
            try:
                proteins_df = pd.read_csv(args.protein_file)
                logger.info(f"Loaded {len(proteins_df)} proteins")
                
                # Check required columns
                if "UniProt_ID" not in proteins_df.columns or "Protein_Sequence" not in proteins_df.columns:
                    logger.error("Protein file must contain 'UniProt_ID' and 'Protein_Sequence' columns")
                    sys.exit(1)
                
                # Generate protein embeddings with batch processing
                start_time = time.time()
                embeddings, ids = embedding_generator.generate_protein_embeddings(
                    proteins_df=proteins_df,
                    batch_size=args.batch_size,
                    resume=not args.no_resume
                )
                
                elapsed_time = time.time() - start_time
                logger.info(f"Generated embeddings for {len(ids)} proteins in {elapsed_time:.2f} seconds")
                logger.info(f"Protein embeddings saved to: {config.protein_embeddings_file}")
                
            except Exception as e:
                logger.error(f"Error processing proteins: {str(e)}")
                logger.debug(traceback.format_exc())
                if not args.only_proteins:
                    logger.info("Continuing with compound processing...")
                else:
                    sys.exit(1)
        
        # Process compounds if not skipped
        if not args.only_proteins:
            logger.info("======= Starting Compound Embedding Generation =======")
            
            # Load compound data
            logger.info(f"Loading compound data from {args.compound_file}")
            try:
                compounds_df = pd.read_csv(args.compound_file)
                logger.info(f"Loaded {len(compounds_df)} compounds")
                
                # Check required columns
                if "cid" not in compounds_df.columns or "smiles" not in compounds_df.columns:
                    logger.error("Compound file must contain 'cid' and 'smiles' columns")
                    sys.exit(1)
                
                # Generate compound embeddings with batch processing
                start_time = time.time()
                embeddings, ids = embedding_generator.generate_compound_embeddings(
                    compounds_df=compounds_df,
                    batch_size=args.batch_size,
                    resume=not args.no_resume
                )
                
                elapsed_time = time.time() - start_time
                logger.info(f"Generated embeddings for {len(ids)} compounds in {elapsed_time:.2f} seconds")
                logger.info(f"Compound embeddings saved to: {config.compound_embeddings_file}")
                
            except Exception as e:
                logger.error(f"Error processing compounds: {str(e)}")
                logger.debug(traceback.format_exc())
                sys.exit(1)
        
        logger.info("======= Embedding Generation Complete =======")
        
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        logger.info("Partial results may have been saved as checkpoints")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        logger.debug(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()