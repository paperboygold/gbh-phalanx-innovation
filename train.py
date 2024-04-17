import torch
from torch.utils.data import Dataset, DataLoader
from preprocess import read_and_preprocess
import pandas as pd
from sklearn.model_selection import train_test_split

from joblib import load
import torch.nn as nn

class UnitSolutionDataset(Dataset):
    def __init__(self, csv_file):
        self.data_frame = pd.read_csv(csv_file)
        self.data_frame['INTERVAL_DATETIME'] = pd.to_datetime(self.data_frame['INTERVAL_DATETIME']).astype(int) / 10**9  # Convert to timestamp

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        row = self.data_frame.iloc[idx]
        features = torch.tensor(row[['INTERVAL_DATETIME', 'INITIALMW', 'TOTALCLEARED', 'RAMPDOWNRATE', 'RAMPUPRATE', 'AVAILABILITY']].values.astype(float), dtype=torch.float32)
        target = torch.tensor(row[['LOWER5MIN', 'LOWER60SEC', 'LOWER6SEC', 'RAISE5MIN', 'RAISE60SEC', 'RAISE6SEC']].values.astype(float), dtype=torch.float32)
        return features, target
    
class TransformerModel(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_encoder_layers, num_decoder_layers, output_dim):
        super(TransformerModel, self).__init__()
        self.transformer = nn.Transformer(d_model=model_dim, nhead=num_heads, 
                                          num_encoder_layers=num_encoder_layers, 
                                          num_decoder_layers=num_decoder_layers,
                                          batch_first=True)
        self.input_linear = nn.Linear(input_dim, model_dim)
        self.output_linear = nn.Linear(model_dim, output_dim)

    def forward(self, src):
        src = self.input_linear(src)
        src = src.unsqueeze(1)  # Add batch dimension
        output = self.transformer(src, src)
        output = self.output_linear(output.squeeze(1))
        return output
    
def train(model, train_dataloader, val_dataloader, epochs):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        for features, targets in train_dataloader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        
        # Validation phase
        validation_loss = validate(model, val_dataloader)
        print(f"Epoch {epoch+1}, Training Loss: {loss.item()}, Validation Loss: {validation_loss}")

def transform_new_data(new_data_file):
    df = pd.read_csv(new_data_file)
    scaler = load('output/unit_solution_scaler.joblib')
    df_transformed = scaler.transform(df[['INITIALMW', 'TOTALCLEARED', 'RAMPDOWNRATE', 'RAMPUPRATE', 'AVAILABILITY']])
    return df_transformed

def validate(model, dataloader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for features, targets in dataloader:
            outputs = model(features)
            loss = nn.MSELoss()(outputs, targets)
            total_loss += loss.item()
    average_loss = total_loss / len(dataloader)
    return average_loss

# Assuming you have a validation dataloader set up
# validation_loss = validate(model, validation_dataloader)

# Preprocess and read data
processed_file_path = read_and_preprocess('data', 'unit_solution', limit_rows=10000)

full_dataset = UnitSolutionDataset(processed_file_path)

# Load dataset
full_dataset = UnitSolutionDataset(processed_file_path)

# Splitting the dataset into train and validation sets
train_size = int(0.8 * len(full_dataset))  # 80% for training
val_size = len(full_dataset) - train_size  # 20% for validation
train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

# Create DataLoaders for train and validation datasets
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Model initialization
model = TransformerModel(input_dim=6, model_dim=512, num_heads=8, num_encoder_layers=3, num_decoder_layers=3, output_dim=6)

# Train the model
train(model, train_dataloader, val_dataloader, epochs=10)