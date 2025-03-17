"""Command-line interface for KIBA prediction model."""

import os
import sys
import time
import logging
import traceback
from datetime import datetime
import argparse
import pandas as pd
import numpy as np

from kiba_model.utils.logging import setup_logging
from kiba_model.config import KIBAConfig
from kiba_model.pipeline import KIBAModelPipeline


def main():
    """Main function to run the KIBA model pipeline."""
    import argparse
    import sys

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='KIBA Dataset Analysis and Modeling Pipeline')

    # Create mutually exclusive group for different modes
    mode_group = parser.add_mutually_exclusive_group(required=True)
    
    mode_group.add_argument('--predict-ids', action='store_true', 
                          help='Enter prediction mode using UniProt ID and PubChem ID')
    mode_group.add_argument('--predict-only', action='store_true', 
                          help='Only set up for prediction (no training)')
    mode_group.add_argument('--train', action='store_true',
                          help='Run the full training pipeline')

    parser.add_argument('--kiba-file', type=str, required=True, help='Path to KIBA interactions file')
    parser.add_argument('--protein-file', type=str, required=True, help='Path to protein sequences file')
    parser.add_argument('--compound-file', type=str, required=True, help='Path to compound SMILES file')
    parser.add_argument('--data-dir', type=str, default='data', help='Directory for data files')
    parser.add_argument('--models-dir', type=str, default='models', help='Directory for model files')
    parser.add_argument('--results-dir', type=str, default='results', help='Directory for results files')
    parser.add_argument('--logs-dir', type=str, default='logs', help='Directory for log files')
    parser.add_argument('--kiba-threshold', type=float, default=100.0, help='Upper threshold for KIBA scores')
    parser.add_argument('--protein-min-length', type=int, default=100, help='Minimum protein sequence length')
    parser.add_argument('--protein-max-length', type=int, default=2000, help='Maximum protein sequence length')
    parser.add_argument('--smiles-max-length', type=int, default=200, help='Maximum SMILES string length')
    parser.add_argument('--log10', action='store_true', help='Use log10 transformation (default: natural log)')
    parser.add_argument('--no-gpu', action='store_true', help='Disable GPU acceleration')
    parser.add_argument('--no-stratification', action='store_true', help='Disable stratified sampling')
    parser.add_argument('--no-backup', action='store_true', help='Do not backup existing files')
    parser.add_argument('--allow-empty', action='store_true', help='Allow empty results after filtering')
    parser.add_argument('--min-interactions', type=int, default=50, 
                        help='Minimum number of valid interactions required')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    parser.add_argument('--model-type', type=str, default='xgboost', 
                        choices=['xgboost', 'neural_network'],
                        help='Model type to use (xgboost or neural_network)')
    parser.add_argument('--uniprot-id', type=str, help='UniProt ID for prediction')
    parser.add_argument('--pubchem-id', type=str, help='PubChem CID for prediction')
    parser.add_argument('--is-experimental', action='store_true', help='Indicate if prediction is for experimental data')
    
    args = parser.parse_args()
    
    # Set up logging level
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logging(args.logs_dir, level=log_level)
    
    # Log the arguments
    logger.info("Command line arguments:")
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")
    
    # Create config
    config = KIBAConfig(
        data_dir=args.data_dir,
        models_dir=args.models_dir,
        results_dir=args.results_dir,
        logs_dir=args.logs_dir,
        kiba_score_threshold=args.kiba_threshold,
        protein_min_length=args.protein_min_length,
        protein_max_length=args.protein_max_length,
        smiles_max_length=args.smiles_max_length,
        use_log10_transform=args.log10,
        gpu_enabled=not args.no_gpu,
        use_stratification=not args.no_stratification,
        random_state=args.seed,
        backup_existing=not args.no_backup,
        allow_empty_results=args.allow_empty,
        min_valid_interactions=args.min_interactions,
        model_type=args.model_type  # Add model type to config
    )
    
    # Set file paths
    config.set_file_paths(args.kiba_file, args.protein_file, args.compound_file)
    
    # Create pipeline
    pipeline = KIBAModelPipeline(config)
    
    try:
        if args.predict_ids:
            if args.uniprot_id is None or args.pubchem_id is None:
                print("Error: Both --uniprot-id and --pubchem-id are required for ID-based prediction")
                sys.exit(1)
                
            prediction = pipeline.predict_by_id(args.uniprot_id, args.pubchem_id)
            
            if prediction:
                print("\nPrediction results:")
                print(f"  Protein: {args.uniprot_id}")
                print(f"  Compound: {args.pubchem_id}")
                print(f"  Predicted KIBA score: {prediction['kiba_score']:.4f}")
                print(f"  Predicted log KIBA score: {prediction['kiba_score_log']:.4f}")
            else:
                print("\nPrediction failed. Check logs for details.")
            sys.exit(0)

        elif args.predict_only:
            # Set up for prediction only
            pipeline.setup_for_prediction()
            logger.info("Model loaded and ready for predictions")
            
            # Simple interactive mode for testing predictions
            print("\nEnter 'q' to quit")
            while True:
                protein_id = input("\nEnter protein UniProt ID (or 'q' to quit): ")
                if protein_id.lower() == 'q':
                    break
                    
                compound_id = input("Enter compound PubChem CID: ")
                if compound_id.lower() == 'q':
                    break
                
                is_exp = input("Is this experimental data? (y/n): ").lower() == 'y'
                
                # Make prediction
                prediction = pipeline.predict(protein_id, compound_id, is_exp)
                
                if prediction:
                    print(f"\nPrediction results:")
                    print(f"  Protein ID: {prediction['protein_id']}")
                    print(f"  Compound ID: {prediction['compound_id']}")
                    print(f"  Experimental: {'Yes' if prediction['is_experimental'] else 'No'}")
                    print(f"  Predicted KIBA score: {prediction['kiba_score']:.4f}")
                    print(f"  Predicted log KIBA score: {prediction['kiba_score_log']:.4f}")
                else:
                    print("\nPrediction failed. Check logs for details.")
        else:
            # Run full pipeline
            start_time = time.time()
            final_model = pipeline.run_full_pipeline()
            total_time = time.time() - start_time
            logger.info(f"Pipeline completed in {total_time:.2f} seconds")
            
            # Test prediction
            try:
                # Load filtered interactions to get a sample
                interactions = pd.read_csv(config.filtered_interactions_file)
                
                if len(interactions) > 0:
                    sample = interactions.iloc[0]
                    protein_id = sample['UniProt_ID_str']
                    compound_id = sample['pubchem_cid_str']
                    
                    # Set up predictor
                    pipeline.setup_for_prediction()
                    
                    # Make sample prediction
                    prediction = pipeline.predict(protein_id, compound_id)
                    
                    if prediction:
                        logger.info(f"\nSample prediction for protein {protein_id} and compound {compound_id}:")
                        logger.info(f"  Predicted KIBA score: {prediction['kiba_score']:.4f}")
                        logger.info(f"  Actual KIBA score: {sample['kiba_score']:.4f}")
                        logger.info(f"  Difference: {abs(prediction['kiba_score'] - sample['kiba_score']):.4f}")
                        
                        # Print to console as well
                        print(f"\nSample prediction:")
                        print(f"  Protein ID: {protein_id}, Compound ID: {compound_id}")
                        print(f"  Predicted KIBA score: {prediction['kiba_score']:.4f}")
                        print(f"  Actual KIBA score: {sample['kiba_score']:.4f}")
                        print(f"  Difference: {abs(prediction['kiba_score'] - sample['kiba_score']):.4f}")
            except Exception as e:
                logger.warning(f"Error in sample prediction: {str(e)}")
    
    except Exception as e:
        logger.error(f"Error in pipeline execution: {str(e)}")
        logger.debug(traceback.format_exc())
        print(f"Error: {str(e)}")
        print("Check log files for details.")
        sys.exit(1)
    
    logger.info("Done!")
    print("\nProcessing complete. See logs and results directories for details.")


if __name__ == "__main__":
    main()