# Project Documentation

## Overview

This project involves the development of machine learning models for energy market data analysis. The primary tasks include data preprocessing, model training, hyperparameter tuning, and ensemble predictions. The project is structured into several Python scripts, each handling specific parts of the workflow.

## Constraints

At the time of writing we are still waiting for AWS to approve our request to access higher-level GPUs. As a result, we are currently unable to provide a well-performing model.

## Scripts and Their Functions

### 1. `download_p5_data.py`

- **Purpose**: Downloads market data reports from NEM (National Electrical Market) via specified URLs and handles data extraction from zip files.
- **Key Functions**:
  - `fetch_file_links(url)`: Fetches downloadable file links from a given URL.
  - `download_file(link, base_directory)`: Downloads a file from a link and handles retries on failure.
  - `generate_urls(start_year, start_month, end_year, end_month)`: Generates URLs for a given date range for data download.

### 2. `preprocess.py`

- **Purpose**: Processes raw CSV data files into a format suitable for machine learning models.
- **Key Functions**:
  - `process_region_solution(file_path, nrows)`: Processes region solution data.
  - `read_and_preprocess(directory, key, nrows)`: Orchestrates the reading and preprocessing of data based on the specified key.
- **Comments**:
  - Only the region solution data is used in this script due to time constraints.

### 3. `train.py`

- **Purpose**: Trains machine learning models using preprocessed data.
- **Key Components**:
  - `RegionSolutionDataset`: A dataset class for handling specific data related to region solutions.
  - `TransformerModel`: A neural network model based on the transformer architecture.
  - Training loop: Manages the training process including optioncross-validation and logging.

### 4. `tuning.py`

- **Purpose**: Optimizes model hyperparameters using Optuna.
- **Key Functions**:
  - `objective(trial)`: Defines the objective function for the Optuna study, which involves training a model and returning the validation loss.

### 5. `run.py`

- **Purpose**: Executes model inference on new data.
- **Key Functions**:
  - `load_model(model_path, model_config)`: Loads a single model from disk using the specified configuration.
  - `load_preprocessed_data(data_file)`: Loads and prepares preprocessed data from a file using the `RegionSolutionDataset`.
  - `run_model_on_new_data(model_path, preprocessed_data_file)`: Runs model predictions on new data, calculates performance metrics (MSE, RMSE, R^2), and saves the results to a CSV file.

## Requirements

- Python 3.8+
- Libraries: pandas, numpy, torch, scikit-learn, joblib, optuna

## Setup and Execution

1. Ensure all dependencies are installed using `pip install -r requirements.txt`.
2. Run the scripts in the following order:
   - Download data: `python download_p5_data.py`
   - Preprocess data: `python preprocess.py`
   - Train models: `python train.py`
   - Tune hyperparameters: `python tuning.py`
   - Run predictions: `python run.py`

## Note on Running

Currently it's best to run `preprocess.py` first, as it will save the scaler to disk. Then, run `train.py` to train the models. Finally, run `run.py` to make predictions. This is to make sure that nrows is set correctly as we didn't have time to fix this in the `train.py`.

## Data Flow

1. **Data Download**: Data is downloaded and extracted using `download_p5_data.py`.
2. **Preprocessing**: Raw data is transformed into a machine learning-friendly format with `preprocess.py`.
3. **Training**: Models are trained using the preprocessed data in `train.py`.
4. **Tuning**: Model parameters are optimized with `tuning.py`.
5. **Inference**: Trained models are used to make predictions on new data in `run.py`.