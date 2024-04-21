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
        processed_file_path = read_and_preprocess(directory='data', key=dataset_key, nrows=100)[0]
    
    # Load the dataset using the appropriate class
    dataset_class = dataset_key_to_class[dataset_key]
    full_dataset = dataset_class(processed_file_path)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    return train_dataset, val_dataset

def objective(trial):
    # Determine the model configuration based on the trial number
    if trial.number % 3 == 0:
        # Configuration for Model 1
        model_dim_options = [256, 512]
        num_heads_options = [2, 4]
        num_encoder_layers_options = [4, 6]
        num_decoder_layers_options = [4, 6]
        dropout_rate_options = [0.1, 0.2]
    elif trial.number % 3 == 1:
        # Configuration for Model 2
        model_dim_options = [512, 1024]
        num_heads_options = [4, 8]
        num_encoder_layers_options = [6, 8]
        num_decoder_layers_options = [6, 8]
        dropout_rate_options = [0.3, 0.4]
    else:
        # Configuration for Model 3
        model_dim_options = [128, 256]
        num_heads_options = [2, 4]
        num_encoder_layers_options = [2, 4]
        num_decoder_layers_options = [2, 4]
        dropout_rate_options = [0.5, 0.6]

    # Suggest model dimension
    model_dim = trial.suggest_categorical('model_dim', model_dim_options)
    num_heads = trial.suggest_categorical('num_heads', [h for h in num_heads_options if model_dim % h == 0])
    num_encoder_layers = trial.suggest_categorical('num_encoder_layers', num_encoder_layers_options)
    num_decoder_layers = trial.suggest_categorical('num_decoder_layers', num_decoder_layers_options)
    dropout_rate = trial.suggest_categorical('dropout_rate', dropout_rate_options)

    # Common parameters for all models
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    l1_lambda = trial.suggest_float('l1_lambda', 1e-5, 1e-1, log=True)
    l2_lambda = trial.suggest_float('l2_lambda', 1e-5, 1e-1, log=True)

    # DataLoader and dataset setup
    dataloader_params = {'batch_size': batch_size, 'shuffle': True}
    dataset_key = 'REGIONSOLUTION'
    input_dim = 43  # From train.py for RegionSolutionDataset
    output_dim = 1   # From train.py for RegionSolutionDataset
    
    # Load and prepare the dataset
    train_dataset, val_dataset = load_data(dataset_key)
    train_dataloader = DataLoader(train_dataset, **dataloader_params)
    val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size)

    # Training Parameters
    epochs = 50
    patience = 5
    splits = 5
    use_cross_validation = True
    nrows = 100

    # Create model
    model = TransformerModel(input_dim=input_dim, model_dim=model_dim, num_heads=num_heads, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers, output_dim=output_dim, dropout_rate=dropout_rate)
    
    # Train the model using the train function
    train(model, RegionSolutionDataset, train_dataloader, val_dataloader, epochs, l1_lambda, l2_lambda, lr, patience, use_cross_validation, splits, input_dim, model_dim, num_heads, num_encoder_layers, num_decoder_layers, output_dim, dropout_rate, dataloader_params, nrows)
    
    # Validation loop to calculate total validation loss
    total_val_loss = validate(model, val_dataloader)

    # Composite objective
    return total_val_loss  # Minimize validation loss

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

print("Best hyperparameters:", study.best_params)