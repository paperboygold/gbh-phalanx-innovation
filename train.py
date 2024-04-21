import torch
from torch.utils.data import Dataset, DataLoader
from preprocess import read_and_preprocess
from torch.optim import Adam
import pandas as pd
from joblib import load
import torch.nn as nn
import math
import numpy as np
import logging
from sklearn.model_selection import KFold
import gc


logging.basicConfig(level=logging.INFO)

class SingletonMeta(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]

class InterconnectorSolutionDataset(Dataset):
    key = "INTERCONNECTORSOLN"

    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.targets = self.data[['MWFLOW', 'MARGINALVALUE']]
        self.features = self.data.drop(columns=['MWFLOW', 'MARGINALVALUE', 'INTERVAL_DATETIME'])

        # Load the scaler
        self.scaler = load('output/interconnectorsoln_scaler.joblib')

        # Define numeric columns for scaling
        numerical_cols = [
            'METEREDMWFLOW', 'MWLOSSES', 'VIOLATIONDEGREE', 
            'EXPORTLIMIT', 'IMPORTLIMIT', 'MARGINALLOSS', 'FCASEXPORTLIMIT', 'FCASIMPORTLIMIT'
        ]

        # Scale only the numeric columns
        if any(col in self.features.columns for col in numerical_cols):
            self.features[numerical_cols] = self.scaler.transform(self.features[numerical_cols])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features = torch.tensor(self.features.iloc[idx].values, dtype=torch.float32)
        target = torch.tensor(self.targets.iloc[idx].values, dtype=torch.float32)
        return features, target

class RegionSolutionDataset(Dataset):
    key = "REGIONSOLUTION"

    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.targets = self.data['RRP']  # Assuming 'RRP' is the target variable
        self.features = self.data.drop(columns=['RRP', 'INTERVAL_DATETIME'])  # Assuming these columns are not features

        # Load the scalar
        self.scaler = load('output/regionsolution_scaler.joblib')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        features = torch.tensor(self.features.iloc[idx].values, dtype=torch.float32)
        target = torch.tensor(self.targets.iloc[idx], dtype=torch.float32)
        target = target.unsqueeze(-1)  # Add an extra dimension to match the output shape of the model
        return features, target
    
class ConstraintSolutionDataset(Dataset):
    key = "CONSTRAINTSOLUTION"

    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        # Assuming 'RHS', 'MARGINALVALUE', and 'VIOLATIONDEGREE' are targets based on their potential predictive value
        self.targets = self.data[['MARGINALVALUE', 'VIOLATIONDEGREE', 'RHS']]
        # Exclude targets and 'INTERVAL_DATETIME' from features
        self.features = self.data.drop(columns=['RHS', 'MARGINALVALUE', 'VIOLATIONDEGREE', 'INTERVAL_DATETIME'])

        # Load the scaler
        self.scaler = load('output/constraintsolution_scaler.joblib')

        # Define numeric columns for scaling
        # Assuming other columns are either categorical (already encoded) or not relevant for scaling
        numerical_cols = ['LHS']

        # Scale only the numeric columns
        if any(col in self.features.columns for col in numerical_cols):
            self.features[numerical_cols] = self.scaler.transform(self.features[numerical_cols])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features = torch.tensor(self.features.iloc[idx].values, dtype=torch.float32)
        target = torch.tensor(self.targets.iloc[idx].values, dtype=torch.float32)
        return features, target

class UnitSolutionDataset(Dataset):
    key = "UNITSOLUTION"
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.targets = self.data[['TOTALCLEARED', 'AVAILABILITY']]
        # Exclude 'INTERVAL_DATETIME' from features
        self.features = self.data.drop(columns=['TOTALCLEARED', 'AVAILABILITY', 'INTERVAL_DATETIME'])
        
        # Load the scaler
        self.scaler = load('output/unit_solution_scaler.joblib')
        
        # Define numeric columns for scaling
        numerical_cols = [
            'INITIALMW', 'RAMPDOWNRATE', 'RAMPUPRATE', 'LOWER5MIN', 'LOWER60SEC', 
            'LOWER6SEC', 'RAISE5MIN', 'RAISE60SEC', 'RAISE6SEC', 'LOWERREG', 'RAISEREG', 
            'SEMIDISPATCHCAP'
        ]
        
        # Scale only the numeric columns
        if any(col in self.features.columns for col in numerical_cols):
            self.features[numerical_cols] = self.scaler.transform(self.features[numerical_cols])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features = torch.tensor(self.features.iloc[idx].values, dtype=torch.float32)
        target = torch.tensor(self.targets.iloc[idx].values, dtype=torch.float32)
        return features, target
    
class TransformerModel(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_encoder_layers, num_decoder_layers, output_dim, dropout_rate=0.1):
        super(TransformerModel, self).__init__()
        self.transformer = nn.Transformer(d_model=model_dim, nhead=num_heads, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers, dropout=dropout_rate, batch_first=True)
        self.input_linear = nn.Linear(input_dim, model_dim)
        self.output_linear = nn.Linear(model_dim, output_dim)

    def forward(self, src):
        # Ensure src is at least 3D (batch_size, seq_len, features)
        if src.dim() == 2:
            src = src.unsqueeze(1)  # Add a sequence length of 1 if it's missing
        src = self.input_linear(src)
        max_len = src.size(1)
        d_model = src.size(2)
        positional_encoding = self.create_positional_encoding(max_len, d_model)
        src = src + positional_encoding[:max_len, :].unsqueeze(0).repeat(src.size(0), 1, 1)
        output = self.transformer(src, src)
        output = self.output_linear(output[:, -1, :])
        return output

    @staticmethod
    def create_positional_encoding(max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe
    
def train(model, dataset_class, train_dataloader, val_dataloader, epochs, l1_lambda, l2_lambda, lr, patience, use_cross_validation, n_splits, input_dim, model_dim, num_heads, num_encoder_layers, num_decoder_layers, output_dim, dropout_rate, dataloader_params, nrows):
    # Load and prepare the dataset
    processed_file_path = read_and_preprocess(directory='data', key=dataset_class.key, nrows=nrows)
    full_dataset = dataset_class(processed_file_path[0])

    # Splitting the dataset into train and validation sets
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    # Create DataLoaders for train and validation datasets
    train_dataloader = DataLoader(train_dataset, **dataloader_params)
    val_dataloader = DataLoader(val_dataset, **dataloader_params)

    global_best_val_loss = float('inf')
    global_best_model_info = {}

    if use_cross_validation:
        kf = KFold(n_splits=n_splits)
        for fold, (train_idx, val_idx) in enumerate(kf.split(full_dataset)):
            logging.info(f"Fold {fold+1}: Train indices range from {train_idx[0]} to {train_idx[-1]}, Validation indices range from {val_idx[0]} to {val_idx[-1]}")
            
            # Reinitialize model at the start of each fold using parameters passed to the function
            model = TransformerModel(input_dim=input_dim, model_dim=model_dim, num_heads=num_heads, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers, output_dim=output_dim, dropout_rate=dropout_rate)

            train_subset = torch.utils.data.Subset(full_dataset, train_idx)
            val_subset = torch.utils.data.Subset(full_dataset, val_idx)

            train_dataloader_fold = DataLoader(train_subset, batch_size=32, shuffle=True)
            val_dataloader_fold = DataLoader(val_subset, batch_size=32, shuffle=False)
            
            fold_best_val_loss, fold_best_model_state = train_single_fold(model, train_dataloader_fold, val_dataloader_fold, epochs, l1_lambda, l2_lambda, lr, patience)
            
            if fold_best_val_loss < global_best_val_loss:
                global_best_val_loss = fold_best_val_loss
                global_best_model_info = {'fold': fold+1, 'state_dict': fold_best_model_state}
                torch.save(global_best_model_info['state_dict'], f'best_model_fold_{global_best_model_info["fold"]}_val_loss_{global_best_val_loss:.4f}.pth')
                logging.info(f"New best model saved with Validation Loss: {global_best_val_loss} from Fold {global_best_model_info['fold']}")
    else:
        train_single_fold(model, train_dataloader, val_dataloader, epochs, l1_lambda, l2_lambda, lr, patience)

    logging.info(f"Overall best model from Fold {global_best_model_info.get('fold', 'N/A')} with Validation Loss: {global_best_val_loss}")

def train_single_fold(model, train_dataloader, val_dataloader, epochs, l1_lambda, l2_lambda, lr, patience):
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=l2_lambda)
    criterion = torch.nn.L1Loss()
    best_val_loss = float('inf')
    best_model_state = None
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for features, targets in train_dataloader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, targets)
            
            # Calculate L1 regularization loss
            l1_loss = sum(p.abs().sum() for p in model.parameters())
            
            # Total loss includes L1 regularization
            total_loss = loss + l1_lambda * l1_loss
            total_loss.backward()
            optimizer.step()
            epoch_loss += total_loss.item()
        
        validation_loss = validate(model, val_dataloader)
        logging.info(f"Epoch {epoch+1}, Validation Loss: {validation_loss}")
        
        if validation_loss < best_val_loss:
            best_val_loss = validation_loss
            best_model_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve > patience:
                logging.info(f"Stopping early after {epoch+1} epochs.")
                break

    return best_val_loss, best_model_state

def transform_new_data(new_data_file):
    df = pd.read_csv(new_data_file)
    scaler = load('output/unit_solution_scaler.joblib')  # Updated path
    features_to_scale = ['INITIALMW', 'TOTALCLEARED', 'RAMPDOWNRATE', 'RAMPUPRATE', 'AVAILABILITY', 'LOWER5MIN', 'LOWER60SEC', 'LOWER6SEC', 'RAISE5MIN', 'RAISE60SEC', 'RAISE6SEC']
    df_transformed = scaler.transform(df[features_to_scale])
    return df_transformed

def validate(model, dataloader):
    model.eval()
    total_loss = 0
    criterion = torch.nn.L1Loss()
    with torch.no_grad():
        for i, (features, targets) in enumerate(dataloader):
            try:
                outputs = model(features)
                loss = criterion(outputs, targets)
                total_loss += loss.item()
            except IndexError as e:
                print(f"Error at index {i}, DataLoader size: {len(dataloader.dataset)}")
                raise e
    return total_loss / len(dataloader)


# Define model configurations
model_configs = [
    {
        'model_dim': 256,
        'num_heads': 2,
        'num_encoder_layers': 4,
        'num_decoder_layers': 6,
        'dropout_rate': 0.5631189875373072,
        'l1_lambda': 2.804138558687802e-05,
        'l2_lambda': 0.037841455862383855,
        'lr': 0.00130961600938406,
        'batch_size': 16
    },
    {
        'model_dim': 256,
        'num_heads': 4,
        'num_encoder_layers': 4,
        'num_decoder_layers': 8,
        'dropout_rate': 0.6,
        'l1_lambda': 2.804138558687802e-05,
        'l2_lambda': 0.04,
        'lr': 0.001,
        'batch_size': 16
    },
    {
        'model_dim': 512,
        'num_heads': 4,
        'num_encoder_layers': 6,
        'num_decoder_layers': 6,
        'dropout_rate': 0.5,
        'l1_lambda': 3e-05,
        'l2_lambda': 0.035,
        'lr': 0.0012,
        'batch_size': 16
    }
]

if __name__ == "__main__":
    # Mapping from keys used in preprocess.py to dataset class names in train.py
    dataset_key_to_class_name = {
        'INTERCONNECTORSOLN': 'InterconnectorSolutionDataset',
        'UNITSOLUTION': 'UnitSolutionDataset',
        'CONSTRAINTSOLUTION': 'ConstraintSolutionDataset',
        'REGIONSOLUTION': 'RegionSolutionDataset'
    }

    # Parameters for training
    epochs = 50
    patience = 5
    use_cross_validation = True
    n_splits = 5
    nrows = 100

    # List of dataset keys as used in preprocess.py
    dataset_keys = ['REGIONSOLUTION']

    # Loop through each dataset key
    for key in dataset_keys:
        logging.info(f"Starting training for dataset key: {key}")
        # Preprocess and read data
        processed_file_path = read_and_preprocess(directory='data', key=key, nrows=nrows)
        dataset_class_name = dataset_key_to_class_name[key]
        dataset_class = globals()[dataset_class_name]  # Dynamically get the dataset class
        full_dataset = dataset_class(processed_file_path[0])

        # Determine the input and output dimensions based on the dataset
        input_dim = {'INTERCONNECTORSOLN': 16, 'UNITSOLUTION': 27, 'CONSTRAINTSOLUTION': 6, 'REGIONSOLUTION': 43}[key]
        output_dim = {'CONSTRAINTSOLUTION': 3, 'INTERCONNECTORSOLN': 2, 'UNITSOLUTION': 2, 'REGIONSOLUTION': 1}[key]

        # Splitting the dataset into train and validation sets
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

        # Create DataLoaders for train and validation datasets
        train_dataloader = DataLoader(train_dataset, batch_size=model_configs[0]['batch_size'], shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=model_configs[0]['batch_size'], shuffle=False)

        # Train each model variant
        for config in model_configs:
            model = TransformerModel(
                input_dim=input_dim,
                model_dim=config['model_dim'],
                num_heads=config['num_heads'],
                num_encoder_layers=config['num_encoder_layers'],
                num_decoder_layers=config['num_decoder_layers'],
                output_dim=output_dim,
                dropout_rate=config['dropout_rate']
            )

            # Train the model
            train(
                model, dataset_class, train_dataloader, val_dataloader, epochs, config['l1_lambda'], config['l2_lambda'], config['lr'], patience, use_cross_validation, n_splits, input_dim, config['model_dim'], config['num_heads'], config['num_encoder_layers'], config['num_decoder_layers'], output_dim, config['dropout_rate'], {'batch_size': config['batch_size'], 'shuffle': True}, nrows
            )

            # Clear memory for the next iteration
            del model
            gc.collect()
            logging.info(f"Finished training model variant with {config['num_decoder_layers']} decoder layers and cleared memory.")
            logging.info(f"Model configuration: {config}")
    logging.info("All model training completed.")