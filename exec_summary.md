# KIBA Prediction Model: Executive Summary

## Overview

The KIBA (Kinase Inhibitor BioActivity) Prediction Model is a machine learning system for predicting the binding affinity between proteins (especially kinases) and small molecules (potential drug compounds). This document provides a detailed explanation of the data, approach, model architecture, and training process.

## Data Understanding

### KIBA Dataset

The KIBA dataset integrates binding affinities from multiple sources:

1. **Experimental Data**: Directly measured binding affinities from biochemical assays
2. **Estimated Data**: Computed binding affinities based on structural or chemical similarity

The primary components of the dataset are:

- **KIBA Interactions**: Contains the binding affinity scores between proteins and compounds
  - `UniProt_ID`: Protein identifier
  - `pubchem_cid`: Compound identifier
  - `kiba_score`: Binding affinity score (higher = stronger binding)
  - `kiba_score_estimated`: Flag indicating if the score is experimental or estimated

- **Protein Sequences**: Contains the amino acid sequences for proteins
  - `UniProt_ID`: Protein identifier
  - `Protein_Sequence`: Amino acid sequence

- **Compound SMILES**: Contains the chemical structures of compounds
  - `cid`: Compound identifier
  - `smiles`: Simplified Molecular-Input Line-Entry System representation of chemical structure

### Data Preprocessing

The pipeline applies several preprocessing steps:

1. **Length Filtering**:
   - Proteins: Removes sequences that are too short (<100 amino acids) or too long (>2000 amino acids)
   - Compounds: Removes SMILES strings that are overly complex (>200 characters)

2. **KIBA Score Filtering**: 
   - Excludes interactions with unusually high KIBA scores (>100)
   - Handles negative values by setting to a small positive number
   - Applies log transformation to normalize the distribution

3. **Stratification**:
   - Creates balanced strata based on experimental status and KIBA score range
   - Ensures representative sampling for training/validation/test splits

## Approach

### Feature Engineering

The model uses deep learning for feature extraction:

1. **Protein Embeddings**:
   - Uses ESM-2 (Evolutionary Scale Modeling), a protein language model from Meta AI
   - Processes protein sequences to generate fixed-length vector representations
   - Captures structural and functional information from amino acid sequences

2. **Compound Embeddings**:
   - Uses ChemBERTa, a BERT-based model trained on chemical data
   - Converts SMILES strings into fixed-length vector representations
   - Captures chemical properties and structural information

3. **Combined Features**:
   - Concatenates protein and compound embeddings
   - Adds an experimental flag feature (1 if experimental, 0 if estimated)
   - Creates feature matrices for model training

### Model Architecture

The system supports multiple model architectures:

1. **XGBoost (Default)**:
   - Gradient boosting framework using tree-based models
   - Advantages: Handles non-linear relationships, robust to outliers, good performance with limited data
   - Hyperparameters tuned: max_depth, learning rate, subsample

2. **Neural Network**:
   - Multi-layer perceptron with batch normalization and dropout
   - Advantages: Can learn complex patterns, scalable with more data
   - Hyperparameters tuned: hidden layer sizes, dropout rate, learning rate

3. **Extensible Framework**:
   - Abstract BaseModel interface allows adding new model types
   - Factory pattern for model instantiation
   - Maintains consistent API across model types

## Training Process Under the Hood

When you run the training pipeline, the following processes occur:

### 1. Data Loading and Validation
```
KIBAModelPipeline.run_preprocessing_pipeline()
  ├─ DataLoader.load_data() - Loads raw CSV files
  ├─ DataLoader.validate_data() - Checks data quality
  └─ DataPreprocessor.preprocess_data() - Filters and processes data
```

- Loads the three data files (interactions, proteins, compounds)
- Validates data quality (missing values, distribution, data types)
- Filters out invalid entries based on configuration parameters
- Creates stratification labels for balanced sampling
- Saves processed data to disk

### 2. Feature Engineering
```
KIBAModelPipeline.run_feature_engineering_pipeline()
  ├─ FeatureEngineering.load_embeddings() - Loads or generates embeddings
  │  └─ EmbeddingGenerator.generate_*_embeddings() - Creates embeddings if needed
  └─ FeatureEngineering.create_feature_matrix() - Builds feature matrix
```

- Loads pre-computed embeddings or generates new ones using language models
- For each protein-compound pair:
  - Retrieves the protein embedding (ESM-2 output)
  - Retrieves the compound embedding (ChemBERTa output)
  - Concatenates them with the experimental flag
- Creates feature matrix X and target vector y
- Handles missing values and outliers
- Saves feature matrices to disk

### 3. Model Training
```
KIBAModelPipeline.run_modeling_pipeline()
  ├─ ModelTrainer.split_data() - Creates train/validation/test splits
  ├─ ModelTrainer.train_initial_model() - Trains baseline model
  ├─ ModelTrainer.tune_hyperparameters() - Optimizes model parameters
  └─ ModelTrainer.train_final_model() - Trains final model with best parameters
```

- Splits data into training, validation, and test sets
- For XGBoost:
  - Creates initial model with default parameters
  - Evaluates multiple parameter combinations (max_depth, learning_rate, subsample)
  - Selects best hyperparameters based on validation performance
  - Trains final model on combined training+validation data
  - Uses early stopping to prevent overfitting

- For Neural Network:
  - Builds network with configurable hidden layers
  - Uses batch normalization and dropout for regularization
  - Applies early stopping and learning rate reduction
  - Evaluates multiple architectures and learning parameters
  - Trains final model on combined training+validation data

### 4. Model Evaluation
```
KIBAModelPipeline.run_modeling_pipeline()
  ├─ ModelEvaluator.evaluate_model() - Calculates performance metrics
  └─ ModelEvaluator.generate_visualizations() - Creates evaluation plots
```

- Evaluates model on test set
- Calculates metrics:
  - RMSE (Root Mean Squared Error)
  - MAE (Mean Absolute Error)
  - R² (Coefficient of Determination)
- Reports metrics on both log and original scales
- Evaluates performance on subgroups:
  - Experimental vs. estimated data
  - Low, medium, and high KIBA score ranges
- Generates visualizations:
  - Actual vs. predicted plots
  - Feature importance (for applicable models)
  - Error distribution
  - Residual plots

## Prediction Process

When making predictions for new protein-compound pairs:

```
KIBAModelPipeline.predict_by_id(uniprot_id, pubchem_id)
  └─ Predictor.predict_by_id() - Makes prediction for new pair
    ├─ Check if embeddings exist for protein and compound
    ├─ Generate embeddings if needed
    ├─ Create feature vector
    ├─ Run prediction using loaded model
    └─ Convert log-scale prediction back to original scale
```

- Loads the trained model and embeddings
- For proteins or compounds not in training data:
  - Fetches sequences from UniProt or PubChem
  - Generates new embeddings using pre-trained language models
- Creates feature vector by concatenating embeddings
- Makes prediction using the loaded model
- Returns both log-scale and original-scale KIBA scores

## Performance Considerations

### Computational Requirements

- **Embedding Generation**: Most computationally intensive step
  - GPU acceleration recommended (4+ GB VRAM)
  - Can take several hours for large datasets
  - Pre-computed embeddings reduce runtime significantly

- **Model Training**:
  - XGBoost: Moderate resource requirements, can run on CPU
  - Neural Network: Benefits from GPU, but can run on CPU for smaller datasets
  - Tuning process tests multiple parameter combinations

### Accuracy Factors

Model performance depends on several factors:

1. **Data Quality and Quantity**:
   - More training examples improve generalization
   - Balance between experimental and estimated data impacts reliability

2. **Sequence Features**:
   - Protein language models capture evolutionary information
   - Chemical language models encode molecular structure

3. **Model Selection**:
   - XGBoost: Better for datasets with <10,000 examples
   - Neural Networks: Scale better with larger datasets

4. **Hyperparameter Tuning**:
   - Grid search identifies optimal parameters
   - Early stopping prevents overfitting

## Customization Options

The pipeline offers several customization options:

1. **Data Filtering**:
   - Adjust protein sequence length thresholds
   - Modify KIBA score filtering criteria

2. **Feature Engineering**:
   - Change embedding models
   - Add additional features

3. **Model Selection**:
   - Choose between XGBoost, Neural Network, or custom models
   - Configure model-specific hyperparameters

4. **Evaluation**:
   - Set test/validation split sizes
   - Add custom evaluation metrics

## Usage Examples

### Training with XGBoost (Default)

```bash
python -m kiba_model.main --train \
  --kiba-file data/kiba_interactions.csv \
  --protein-file data/proteins.csv \
  --compound-file data/compounds.csv
```

### Training with Neural Network

```bash
python -m kiba_model.main --train \
  --kiba-file data/kiba_interactions.csv \
  --protein-file data/proteins.csv \
  --compound-file data/compounds.csv \
  --model-type neural_network
```

### Making Predictions

```bash
python -m kiba_model.main --predict-ids \
  --kiba-file data/kiba_interactions.csv \
  --protein-file data/proteins.csv \
  --compound-file data/compounds.csv \
  --uniprot-id P00533 \
  --pubchem-id 176870 \
  --is-experimental
```

## Conclusion

The KIBA Prediction Model provides a flexible pipeline for predicting protein-ligand binding affinities. By leveraging modern protein and chemical language models, it creates powerful feature representations that enable accurate predictions. The modular architecture allows for easy customization and extension with new model types, making it adaptable to different research needs and datasets.