import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from train import TransformerModel
from sklearn.metrics import mean_squared_error, r2_score


# Assuming the model and necessary functions are in train.py
from train import TransformerModel

def load_model(model_path):
    # Initialize the model with the architecture parameters used during training
    model = TransformerModel(
        input_dim=43,  # From the 'REGIONSOLUTION' dataset
        model_dim=128,  # From best_hyperparams
        num_heads=8,  # From best_hyperparams
        num_encoder_layers=5,  # From best_hyperparams
        num_decoder_layers=3,  # From best_hyperparams
        output_dim=1,  # From 'REGIONSOLUTION' dataset
        dropout_rate=0.5372829108067698  # From best_hyperparams
    )
    
    # Load the model state dictionary
    model_state = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(model_state)
    
    return model

def load_preprocessed_data(data_file):
    # Load the preprocessed data
    df = pd.read_csv(data_file)
    features = df.drop(columns=['RRP', 'INTERVAL_DATETIME'])
    targets = df['RRP']  # Assuming 'RRP' is the target column

    # Convert all columns to float32 explicitly
    features = features.astype(np.float32)
    targets = targets.astype(np.float32)

    return features.values, targets.values


def run_model_on_new_data(model_path, preprocessed_data_file):
    # Load the model
    model = load_model(model_path)
    
    # Load preprocessed new data and targets
    new_data, actual_targets = load_preprocessed_data(preprocessed_data_file)
    
    # Convert to tensor and create a DataLoader
    features_tensor = torch.tensor(new_data, dtype=torch.float32)
    targets_tensor = torch.tensor(actual_targets, dtype=torch.float32).unsqueeze(1)
    dataset = TensorDataset(features_tensor, targets_tensor)
    dataloader = DataLoader(dataset, batch_size=16)
    
    # Run the model on new data
    model.eval()
    predictions = []
    with torch.no_grad():
        for features, targets in dataloader:
            outputs = model(features)
            predictions.extend(outputs.numpy().flatten())

    # Calculate metrics
    mse = mean_squared_error(actual_targets, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(actual_targets, predictions)

    # Save results to CSV
    results_df = pd.DataFrame({'Actual': actual_targets, 'Predicted': predictions})
    results_df.to_csv('model_predictions.csv', index=False)
    print(f"Results saved to model_predictions.csv")
    print(f"MSE: {mse}, RMSE: {rmse}, R^2: {r2}")

    return predictions, actual_targets, mse, rmse, r2

if __name__ == "__main__":
    model_path = 'best_model_fold_5_val_loss_0.9349.pth'
    preprocessed_data_file = 'output/regionsolution.csv'  # Ensure this file is preprocessed similarly to training data
    predictions = run_model_on_new_data(model_path, preprocessed_data_file)
    print(predictions)

