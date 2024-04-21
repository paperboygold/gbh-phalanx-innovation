import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from train import TransformerModel
from sklearn.metrics import mean_squared_error, r2_score

# Define model configurations for the ensemble
model_configs = [
    {
        'input_dim': 43,  # Assuming all models use the same input dimension
        'model_dim': 256,
        'num_heads': 2,
        'num_encoder_layers': 4,
        'num_decoder_layers': 6,
        'output_dim': 1,  # Assuming all models predict the same output dimension
        'dropout_rate': 0.5631189875373072
    },
    {
        'input_dim': 43,
        'model_dim': 256,
        'num_heads': 4,
        'num_encoder_layers': 4,
        'num_decoder_layers': 8,
        'output_dim': 1,
        'dropout_rate': 0.6
    },
    {
        'input_dim': 43,
        'model_dim': 512,
        'num_heads': 4,
        'num_encoder_layers': 6,
        'num_decoder_layers': 6,
        'output_dim': 1,
        'dropout_rate': 0.5
    }
]

def load_models(model_paths, model_configs):
    models = []
    for model_path, config in zip(model_paths, model_configs):
        model = TransformerModel(
            input_dim=config['input_dim'],
            model_dim=config['model_dim'],
            num_heads=config['num_heads'],
            num_encoder_layers=config['num_encoder_layers'],
            num_decoder_layers=config['num_decoder_layers'],
            output_dim=config['output_dim'],
            dropout_rate=config['dropout_rate']
        )
        model_state = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(model_state)
        model.eval()  # Set the model to evaluation mode
        models.append(model)
    return models

def load_preprocessed_data(data_file):
    df = pd.read_csv(data_file)
    features = df.drop(columns=['RRP', 'INTERVAL_DATETIME'])
    targets = df['RRP']  # Assuming 'RRP' is the target column
    features = features.astype(np.float32)
    targets = targets.astype(np.float32)
    return features.values, targets.values

def run_ensemble_on_new_data(model_paths, preprocessed_data_file):
    models = load_models(model_paths)
    new_data, actual_targets = load_preprocessed_data(preprocessed_data_file)
    features_tensor = torch.tensor(new_data, dtype=torch.float32)
    targets_tensor = torch.tensor(actual_targets, dtype=torch.float32).unsqueeze(1)
    dataset = TensorDataset(features_tensor, targets_tensor)
    dataloader = DataLoader(dataset, batch_size=16)
    
    ensemble_predictions = []
    with torch.no_grad():
        for features, _ in dataloader:
            model_predictions = [model(features).numpy().flatten() for model in models]
            averaged_predictions = np.mean(model_predictions, axis=0)
            ensemble_predictions.extend(averaged_predictions)
    
    mse = mean_squared_error(actual_targets, ensemble_predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(actual_targets, ensemble_predictions)
    results_df = pd.DataFrame({'Actual': actual_targets, 'Predicted': ensemble_predictions})
    results_df.to_csv('ensemble_predictions.csv', index=False)
    print(f"Results saved to ensemble_predictions.csv")
    print(f"MSE: {mse}, RMSE: {rmse}, R^2: {r2}")
    return ensemble_predictions, actual_targets, mse, rmse, r2

if __name__ == "__main__":
    model_paths = [
        'best_model_fold_1_val_loss_0.9349.pth',
        'best_model_fold_2_val_loss_0.9301.pth',
        'best_model_fold_3_val_loss_0.9320.pth'
    ]
    preprocessed_data_file = 'output/regionsolution.csv'
    
    # Load models with configurations
    models = load_models(model_paths, model_configs)
    predictions, actual_targets, mse, rmse, r2 = run_ensemble_on_new_data(models, preprocessed_data_file)
    print(predictions)