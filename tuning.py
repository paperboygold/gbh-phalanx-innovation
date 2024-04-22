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
    # Suggest model configuration parameters
    model_dim = trial.suggest_categorical('model_dim', [128, 256, 512, 1024])
    # Ensure num_heads is a divisor of model_dim
    possible_heads = [h for h in [2, 4, 8, 12, 16] if model_dim % h == 0]
    num_heads = trial.suggest_categorical('num_heads', possible_heads)
    num_encoder_layers = trial.suggest_categorical('num_encoder_layers', [2, 4, 6, 8, 12, 16])
    num_decoder_layers = trial.suggest_categorical('num_decoder_layers', [2, 4, 6, 8, 12, 16])
    dropout_rate = trial.suggest_float('dropout_rate', 0.01, 0.99, log=True)
    gamma = trial.suggest_float('gamma', 0.55, 3.0, log=False)  # Suggest gamma value for learning rate decay

    # Suggest common parameters
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
    lr = trial.suggest_float('lr', 1e-4, 1e-1, log=True)
    l1_lambda = trial.suggest_float('l1_lambda', 1e-4, 1e-1, log=True)
    l2_lambda = trial.suggest_float('l2_lambda', 1e-5, 1e-2, log=False)

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
    use_cross_validation = False
    nrows = 10000

    # Create model
    model = TransformerModel(input_dim=input_dim, model_dim=model_dim, num_heads=num_heads, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers, output_dim=output_dim, dropout_rate=dropout_rate)
    
    # Train the model using the train function with all parameters including gamma
    train(
        model, RegionSolutionDataset, train_dataloader, val_dataloader, epochs, l1_lambda, l2_lambda, lr, patience, use_cross_validation, splits, input_dim, model_dim, num_heads, num_encoder_layers, num_decoder_layers, output_dim, dropout_rate, dataloader_params, nrows, gamma
    )

    # Validation loop to calculate total validation loss
    total_val_loss = validate(model, val_dataloader)

    # Return the objective value
    return total_val_loss  # Minimize validation loss

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

print("Best hyperparameters:", study.best_params)