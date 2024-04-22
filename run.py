import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from train import TransformerModel, RegionSolutionDataset
from sklearn.metrics import mean_squared_error, r2_score

# Single model configuration as per train.py
model_config = {
    'input_dim': 43,  # Assuming the input dimension based on the dataset used
    'model_dim': 512,
    'num_heads': 4,
    'num_encoder_layers': 6,
    'num_decoder_layers': 4,
    'dropout_rate': 0.4366813665078332,
    'l1_lambda': 0.00235596007610553,
    'l2_lambda': 0.009789422077568735,
    'lr': 0.00048003182324549685,
    'batch_size': 64,
    'output_dim': 1,
    'gamma': 0.9765306136576826
    }

def load_model(model_path, model_config):
    model = TransformerModel(
        input_dim=model_config['input_dim'],
        model_dim=model_config['model_dim'],
        num_heads=model_config['num_heads'],
        num_encoder_layers=model_config['num_encoder_layers'],
        num_decoder_layers=model_config['num_decoder_layers'],
        output_dim=model_config['output_dim'],
        dropout_rate=model_config['dropout_rate']
    )
    model_state = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(model_state)
    model.eval()  # Set the model to evaluation mode
    return model

def load_preprocessed_data(data_file):
    # Use the RegionSolutionDataset class for loading and preprocessing data
    dataset = RegionSolutionDataset(data_file)
    features = []
    targets = []
    for i in range(len(dataset)):
        feature, target = dataset[i]
        features.append(feature.numpy())
        targets.append(target.numpy())
    features = np.array(features)
    targets = np.array(targets).flatten()  # Flatten targets to match prediction shape
    return features, targets

def run_model_on_new_data(model_path, preprocessed_data_file):
    model = load_model(model_path, model_config)
    new_data, actual_targets = load_preprocessed_data(preprocessed_data_file)
    features_tensor = torch.tensor(new_data, dtype=torch.float32)
    targets_tensor = torch.tensor(actual_targets, dtype=torch.float32).unsqueeze(1)
    dataset = TensorDataset(features_tensor, targets_tensor)
    dataloader = DataLoader(dataset, batch_size=16)
    
    predictions = []
    with torch.no_grad():
        for features, _ in dataloader:
            model_predictions = model(features).numpy().flatten()
            predictions.extend(model_predictions)
    
    mse = mean_squared_error(actual_targets, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(actual_targets, predictions)
    results_df = pd.DataFrame({'Actual': actual_targets, 'Predicted': predictions})
    results_df.to_csv('model_predictions.csv', index=False)
    print(f"Results saved to model_predictions.csv")
    print(f"MSE: {mse}, RMSE: {rmse}, R^2: {r2}")
    return predictions, actual_targets, mse, rmse, r2

if __name__ == "__main__":
    model_path = 'models/best_model_fold_2_val_loss_0.5011.pth'  # Update this path as necessary
    preprocessed_data_file = 'output/regionsolution.csv'
    
    # Run model on new data
    predictions, actual_targets, mse, rmse, r2 = run_model_on_new_data(model_path, preprocessed_data_file)
    print(predictions)