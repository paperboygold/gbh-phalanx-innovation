import torch
from torch.utils.data import DataLoader
from train import TransformerModel, train, UnitSolutionDataset, InterconnectorSolutionDataset, RegionSolutionDataset, ConstraintSolutionDataset, validate
from preprocess import read_and_preprocess
import optuna
import os

# Mapping from keys to dataset classes
dataset_key_to_class = {
    'UNITSOLUTION': UnitSolutionDataset,
    'INTERCONNECTORSOLN': InterconnectorSolutionDataset,
    'REGIONSOLUTION': RegionSolutionDataset,
    'CONSTRAINTSOLUTION': ConstraintSolutionDataset
}

def load_data(dataset_key):
    # Define the path to the processed data
    processed_file_path = f'output/{dataset_key.lower()}.csv'
    
    # Check if the processed data file exists
    if not os.path.exists(processed_file_path):
        # If the file does not exist, process the data
        print(f"Processed data for {dataset_key} not found, processing now...")
        processed_file_path = read_and_preprocess(directory='data', key=dataset_key, nrows=None)[0]
    
    # Load the dataset using the appropriate class
    dataset_class = dataset_key_to_class[dataset_key]
    full_dataset = dataset_class(processed_file_path)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    return train_dataset, val_dataset

def objective(trial):
    # Define fixed model dimensions that are multiples of common head counts
    model_dim_options = [128, 256, 512, 1024]  # Common dimensions that are multiples of many possible head counts

    # Suggest model dimension first
    model_dim = trial.suggest_categorical('model_dim', model_dim_options)

    # Suggest number of heads that are divisors of the chosen model_dim
    possible_heads = [h for h in range(2, 13) if model_dim % h == 0]
    num_heads = trial.suggest_categorical('num_heads', possible_heads)

    # Other parameters remain the same
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    num_encoder_layers = trial.suggest_int('num_encoder_layers', 1, 8)
    num_decoder_layers = trial.suggest_int('num_decoder_layers', 1, 8)
    dropout_rate = trial.suggest_float('dropout_rate', 0.01, 0.9)
    l1_lambda = trial.suggest_float('l1_lambda', 1e-5, 1e-1, log=True)
    l2_lambda = trial.suggest_float('l2_lambda', 1e-5, 1e-1, log=True)

    # Training control parameters
    epochs = 20  # Number of training epochs
    patience = 5  # Patience for early stopping
    use_cross_validation = True  # Whether to use cross-validation
    n_splits = 5  # Number of splits for cross-validation

    # DataLoader parameters
    dataloader_params = {
        'batch_size': batch_size,
        'shuffle': True
    }

    # Model setup based on dataset
    dataset_key = 'REGIONSOLUTION'
    input_dim = {'UNITSOLUTION': 27, 'INTERCONNECTORSOLN': 16, 'REGIONSOLUTION': 43, 'CONSTRAINTSOLUTION': 6}[dataset_key]
    output_dim = {'UNITSOLUTION': 2, 'INTERCONNECTORSOLN': 2, 'REGIONSOLUTION': 1, 'CONSTRAINTSOLUTION': 3}[dataset_key]
    
    # Load and prepare the dataset
    dataset_class = dataset_key_to_class[dataset_key]
    processed_file_path = f'output/{dataset_key.lower()}.csv'
    full_dataset = dataset_class(processed_file_path)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    # Create DataLoaders for train and validation datasets
    train_dataloader = DataLoader(train_dataset, **dataloader_params)
    val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size)

    model = TransformerModel(input_dim=input_dim, model_dim=model_dim, num_heads=num_heads, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers, output_dim=output_dim, dropout_rate=dropout_rate)
    
    # Train the model using the train function
    train(model, dataset_class, train_dataloader, val_dataloader, epochs, l1_lambda, l2_lambda, lr, patience, use_cross_validation, n_splits, input_dim, model_dim, num_heads, num_encoder_layers, num_decoder_layers, output_dim, dropout_rate, dataloader_params)
    
    # Validation loop to calculate total validation loss
    total_val_loss = validate(model, val_dataloader)

    # Composite objective
    return total_val_loss  # Minimize validation loss

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

print("Best hyperparameters:", study.best_params)