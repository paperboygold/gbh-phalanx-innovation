import torch
from torch.utils.data import DataLoader
from train import TransformerModel, train_for_tuning, validate_for_tuning, UnitSolutionDataset
from preprocess import read_and_preprocess
import optuna
import os
from torch.optim import Adam
from torch.nn import L1Loss

def load_data():
    # Define the path to the processed data
    processed_file_path = 'output/unitsolution.csv'
    
    # Check if the processed data file exists
    if not os.path.exists(processed_file_path):
        # If the file does not exist, process the data
        print("Processed data not found, processing now...")
        processed_file_path = read_and_preprocess(directory='data', key='UNITSOLUTION', nrows=None)[0]
    
    # Load the dataset
    full_dataset = UnitSolutionDataset(processed_file_path)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    return train_dataset, val_dataset

def objective(trial):
    # Data preparation steps
    processed_file_path = read_and_preprocess(directory='data', key='UNITSOLUTION', nrows=None)  # Adjust as needed
    full_dataset = UnitSolutionDataset(processed_file_path[0])

    # Splitting the dataset into train and validation sets
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    # Create DataLoaders for train and validation datasets
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Define the hyperparameters to tune
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
    num_layers = trial.suggest_int('num_layers', 1, 6)
    model_dim = trial.suggest_categorical('model_dim', [128, 256, 512])
    dropout_rate = trial.suggest_uniform('dropout_rate', 0.1, 0.3)

    # Model setup
    model = TransformerModel(input_dim=11, model_dim=model_dim, num_heads=8, num_encoder_layers=num_layers, num_decoder_layers=num_layers, output_dim=2, dropout_rate=dropout_rate)
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = L1Loss()

    # Define a balance factor between train and validation loss
    alpha = 0.5

    # Training loop (simplified)
    total_train_loss = 0
    total_val_loss = 0
    for epoch in range(10):
        train_loss = train_for_tuning(model, train_dataloader, optimizer, criterion)
        val_loss = validate_for_tuning(model, val_dataloader, criterion)
        total_train_loss += train_loss
        total_val_loss += val_loss

    # Calculate average losses
    avg_train_loss = total_train_loss / 10
    avg_val_loss = total_val_loss / 10

    # Composite objective
    return alpha * avg_train_loss + (1 - alpha) * avg_val_loss

# Load data once before starting the optimization
train_dataset, val_dataset = load_data()

# Create DataLoaders for train and validation datasets
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

print("Best hyperparameters:", study.best_params)