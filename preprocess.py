import pandas as pd
import os
import logging
from sklearn.preprocessing import StandardScaler
from joblib import dump
import multiprocessing
from sklearn.preprocessing import LabelEncoder
import re


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_and_save(file_path, process_function, output_file, nrows=None):
    logging.info(f"Processing file: {file_path}")
    df = process_function(file_path, nrows)
    logging.info(f"Processed data shape: {df.shape}")
    file_exists = os.path.exists(output_file)
    df.to_csv(output_file, mode='a', header=not file_exists, index=False)
    if file_exists:
        logging.info(f"{os.path.basename(file_path)} appended to {output_file}")
    else:
        logging.info(f"{os.path.basename(file_path)} saved to {output_file}")

def process_constraint_solution(file_path, nrows=None):
    df = pd.read_csv(file_path, header=1, nrows=nrows)
    df['INTERVAL_DATETIME'] = pd.to_datetime(df['INTERVAL_DATETIME'], format='%Y/%m/%d %H:%M:%S')
    
    # Extracting temporal features
    df['HOUR'] = df['INTERVAL_DATETIME'].dt.hour
    df['DAY_OF_WEEK'] = df['INTERVAL_DATETIME'].dt.dayofweek
    df['MONTH'] = df['INTERVAL_DATETIME'].dt.month

    # Label Encoding for categorical columns if necessary
    label_encoder = LabelEncoder()
    categorical_columns = ['CONSTRAINTID', 'DUID']  # Updated to include all categorical columns
    for col in categorical_columns:
        df[col] = label_encoder.fit_transform(df[col].astype(str))

    # Define all relevant columns, including newly extracted and encoded ones
    selected_columns = [
        'INTERVAL_DATETIME', 'CONSTRAINTID', 'RHS', 'MARGINALVALUE', 'VIOLATIONDEGREE', 
        'HOUR', 'DAY_OF_WEEK', 'MONTH', 'DUID', 'LHS'
    ]
    
    df = df[selected_columns]
    
    return df

def process_region_solution(file_path, nrows=None):
    df = pd.read_csv(file_path, header=1, nrows=nrows)
    
    # Convert INTERVAL_DATETIME to datetime and extract temporal features
    df['INTERVAL_DATETIME'] = pd.to_datetime(df['INTERVAL_DATETIME'], format='%Y/%m/%d %H:%M:%S')
    df['HOUR'] = df['INTERVAL_DATETIME'].dt.hour
    df['DAY_OF_WEEK'] = df['INTERVAL_DATETIME'].dt.dayofweek
    df['MONTH'] = df['INTERVAL_DATETIME'].dt.month

    # Label Encoding for 'REGIONID'
    label_encoder = LabelEncoder()
    df['REGIONID'] = label_encoder.fit_transform(df['REGIONID'].astype(str))

    # Fill NaN values in numerical columns with 0 or a suitable value
    numerical_cols = ['RRP', 'ROP', 'EXCESSGENERATION', 'RAISE6SECRRP', 'RAISE6SECROP', 'RAISE60SECRRP', 'RAISE60SECROP', 
                      'RAISE5MINRRP', 'RAISE5MINROP', 'RAISEREGRRP', 'RAISEREGROP', 'LOWER6SECRRP', 'LOWER6SECROP', 
                      'LOWER60SECRRP', 'LOWER60SECROP', 'LOWER5MINRRP', 'LOWER5MINROP', 'LOWERREGRRP', 'LOWERREGROP', 
                      'TOTALDEMAND', 'AVAILABLEGENERATION', 'AVAILABLELOAD', 'DEMANDFORECAST', 'DISPATCHABLEGENERATION', 
                      'DISPATCHABLELOAD', 'NETINTERCHANGE', 'TOTALINTERMITTENTGENERATION', 'DEMAND_AND_NONSCHEDGEN', 
                      'UIGF', 'SEMISCHEDULE_CLEAREDMW', 'SEMISCHEDULE_COMPLIANCEMW', 'SS_SOLAR_UIGF', 'SS_WIND_UIGF', 
                      'SS_SOLAR_CLEAREDMW', 'SS_WIND_CLEAREDMW', 'SS_SOLAR_COMPLIANCEMW', 'SS_WIND_COMPLIANCEMW', 
                      'WDR_INITIALMW', 'WDR_AVAILABLE', 'WDR_DISPATCHED']
    for col in numerical_cols:
        df[col] = df[col].fillna(0)

    # Include all columns from the CSV file, now with filled NaNs and encoded REGIONID
    selected_columns = [
        'INTERVAL_DATETIME', 'REGIONID', 'RRP', 'ROP', 'EXCESSGENERATION', 'RAISE6SECRRP', 'RAISE6SECROP', 
        'RAISE60SECRRP', 'RAISE60SECROP', 'RAISE5MINRRP', 'RAISE5MINROP', 'RAISEREGRRP', 'RAISEREGROP', 
        'LOWER6SECRRP', 'LOWER6SECROP', 'LOWER60SECRRP', 'LOWER60SECROP', 'LOWER5MINRRP', 'LOWER5MINROP', 
        'LOWERREGRRP', 'LOWERREGROP', 'TOTALDEMAND', 'AVAILABLEGENERATION', 'AVAILABLELOAD', 'DEMANDFORECAST', 
        'DISPATCHABLEGENERATION', 'DISPATCHABLELOAD', 'NETINTERCHANGE', 'HOUR', 'DAY_OF_WEEK', 'MONTH', 
        'TOTALINTERMITTENTGENERATION', 'DEMAND_AND_NONSCHEDGEN', 'UIGF', 'SEMISCHEDULE_CLEAREDMW', 
        'SEMISCHEDULE_COMPLIANCEMW', 'SS_SOLAR_UIGF', 'SS_WIND_UIGF', 'SS_SOLAR_CLEAREDMW', 'SS_WIND_CLEAREDMW', 
        'SS_SOLAR_COMPLIANCEMW', 'SS_WIND_COMPLIANCEMW', 'WDR_INITIALMW', 'WDR_AVAILABLE', 'WDR_DISPATCHED'
    ]
    
    return df[selected_columns]

def process_interconnector_solution(file_path, nrows=None):
    df = pd.read_csv(file_path, header=1, nrows=nrows)
    df['INTERVAL_DATETIME'] = pd.to_datetime(df['INTERVAL_DATETIME'], format='%Y/%m/%d %H:%M:%S')
    
    # Extracting temporal features
    df['HOUR'] = df['INTERVAL_DATETIME'].dt.hour
    df['DAY_OF_WEEK'] = df['INTERVAL_DATETIME'].dt.dayofweek
    df['MONTH'] = df['INTERVAL_DATETIME'].dt.month

    # Label Encoding for categorical columns if necessary
    label_encoder = LabelEncoder()
    categorical_columns = ['INTERCONNECTORID', 'EXPORTGENCONID', 'IMPORTGENCONID']  # Add other categorical columns as needed
    for col in categorical_columns:
        df[col] = label_encoder.fit_transform(df[col].astype(str))

    # Define all relevant columns, including newly extracted and encoded ones
    selected_columns = [
        'INTERVAL_DATETIME', 'INTERVENTION', 'INTERCONNECTORID', 'METEREDMWFLOW', 'MWFLOW', 
        'MWLOSSES', 'MARGINALVALUE', 'VIOLATIONDEGREE', 'MNSP', 'EXPORTLIMIT', 
        'IMPORTLIMIT', 'MARGINALLOSS', 'EXPORTGENCONID', 'IMPORTGENCONID', 
        'FCASEXPORTLIMIT', 'FCASIMPORTLIMIT', 'HOUR', 'DAY_OF_WEEK', 'MONTH'
    ]
    
    df = df[selected_columns]
    
    return df

def process_unit_solution(file_path, nrows=None):
    logging.info(f"Reading CSV file: {file_path}")
    df = pd.read_csv(file_path, header=1, nrows=nrows)
    df['INTERVAL_DATETIME'] = pd.to_datetime(df['INTERVAL_DATETIME'], format='%Y/%m/%d %H:%M:%S')
    logging.debug("Datetime conversion completed.")

    # Extracting temporal features
    df['HOUR'] = df['INTERVAL_DATETIME'].dt.hour
    df['DAY_OF_WEEK'] = df['INTERVAL_DATETIME'].dt.dayofweek
    df['MONTH'] = df['INTERVAL_DATETIME'].dt.month
    logging.debug("Extracted temporal features.")

    # Label Encoding for 'DUID' and 'CONNECTIONPOINTID'
    label_encoder_duid = LabelEncoder()
    df['DUID'] = label_encoder_duid.fit_transform(df['DUID'].astype(str))

    label_encoder_connectionpointid = LabelEncoder()
    df['CONNECTIONPOINTID'] = label_encoder_connectionpointid.fit_transform(df['CONNECTIONPOINTID'].astype(str))

    # Define all relevant columns, excluding 'TRADETYPE'
    selected_columns = [
        'INTERVAL_DATETIME', 'DUID', 'CONNECTIONPOINTID', 'AGCSTATUS', 'INITIALMW', 
        'TOTALCLEARED', 'RAMPDOWNRATE', 'RAMPUPRATE', 'LOWER5MIN', 'LOWER60SEC', 'LOWER6SEC', 
        'RAISE5MIN', 'RAISE60SEC', 'RAISE6SEC', 'LOWERREG', 'RAISEREG', 'AVAILABILITY', 
        'RAISE6SECFLAGS', 'RAISE60SECFLAGS', 'RAISE5MINFLAGS', 'RAISEREGFLAGS', 'LOWER6SECFLAGS', 
        'LOWER60SECFLAGS', 'LOWER5MINFLAGS', 'LOWERREGFLAGS', 'SEMIDISPATCHCAP', 
        'DISPATCHMODETIME', 'HOUR', 'DAY_OF_WEEK', 'MONTH'
    ]
    
    df = df[selected_columns]
    
    return df

def scale_and_save_data(combined_df, key):
    # Define numerical columns based on the key
    if key == 'UNITSOLUTION':
        numerical_cols = [
            'INITIALMW', 'RAMPDOWNRATE', 'RAMPUPRATE', 'LOWER5MIN', 'LOWER60SEC', 
            'LOWER6SEC', 'RAISE5MIN', 'RAISE60SEC', 'RAISE6SEC', 'LOWERREG', 'RAISEREG', 
            'SEMIDISPATCHCAP'
        ]
    elif key == 'INTERCONNECTORSOLN':
        numerical_cols = [
            'METEREDMWFLOW', 'MWLOSSES', 'VIOLATIONDEGREE', 
            'EXPORTLIMIT', 'IMPORTLIMIT', 'MARGINALLOSS', 'FCASEXPORTLIMIT', 'FCASIMPORTLIMIT'
        ]
    elif key == 'CONSTRAINTSOLUTION':
        numerical_cols = [
            'LHS'
        ]
    elif key == 'REGIONSOLUTION':
        numerical_cols = [
            'ROP', 'EXCESSGENERATION', 'RAISE6SECRRP', 'RAISE6SECROP', 'RAISE60SECRRP', 'RAISE60SECROP', 
            'RAISE5MINRRP', 'RAISE5MINROP', 'RAISEREGRRP', 'RAISEREGROP', 'LOWER6SECRRP', 'LOWER6SECROP', 
            'LOWER60SECRRP', 'LOWER60SECROP', 'LOWER5MINRRP', 'LOWER5MINROP', 'LOWERREGRRP', 'LOWERREGROP', 
            'TOTALDEMAND', 'AVAILABLEGENERATION', 'AVAILABLELOAD', 'DEMANDFORECAST', 'DISPATCHABLEGENERATION', 
            'DISPATCHABLELOAD', 'NETINTERCHANGE', 'TOTALINTERMITTENTGENERATION', 'DEMAND_AND_NONSCHEDGEN', 
            'UIGF', 'SEMISCHEDULE_CLEAREDMW', 'SEMISCHEDULE_COMPLIANCEMW', 'SS_SOLAR_UIGF', 'SS_WIND_UIGF', 
            'SS_SOLAR_CLEAREDMW', 'SS_WIND_CLEAREDMW', 'SS_SOLAR_COMPLIANCEMW', 'SS_WIND_COMPLIANCEMW', 
            'WDR_INITIALMW', 'WDR_AVAILABLE', 'WDR_DISPATCHED'
        ] 

    scaler = StandardScaler()
    # Ensure only numerical columns are scaled
    combined_df[numerical_cols] = scaler.fit_transform(combined_df[numerical_cols])
    scaler_filename = f'output/{key.lower()}_scaler.joblib'
    dump(scaler, scaler_filename)
    output_filename = f'output/{key.lower()}.csv'
    combined_df.to_csv(output_filename, index=False)
    logging.info(f"Data scaled and saved to {output_filename}. Scaler saved to {scaler_filename}.")

def read_and_preprocess(directory, key=None, nrows=None):
    output_path = f'output/{key.lower()}.csv'
    if os.path.exists(output_path):
        logging.info(f"Loading processed data from {output_path}")
        return [output_path]
    
    # Define the process map directly with keys that match file identifiers
    process_map = {
        'CONSTRAINTSOLUTION': process_constraint_solution,
        'INTERCONNECTORSOLN': process_interconnector_solution,
        'REGIONSOLUTION': process_region_solution,
        'UNITSOLUTION': process_unit_solution
    }

    # Validate the key if provided
    if key and key not in process_map:
        raise ValueError(f"Invalid key provided: {key}. Valid keys are: {list(process_map.keys())}")

    # List all CSV files in the directory
    all_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.CSV')]

    # Create a list to hold files and their types
    files_to_process = []

    # Identify file type and corresponding function
    for file in all_files:
        for process_key in process_map:
            if process_key in file.upper():
                if key and process_key != key:
                    continue  # Skip files not matching the specified key
                date_part = re.search(r'\d{12}', file)  # Regex to extract the datetime part
                date = pd.to_datetime(date_part.group(), format='%Y%m%d%H%M') if date_part else None
                files_to_process.append((date, file, process_map[process_key]))

    # Sort files by date
    files_to_process.sort()

    # Process each file and combine data
    combined_df = pd.DataFrame()  # Initialize an empty DataFrame
    processed_files = set()
    for date, file_path, process_function in files_to_process:
        if file_path in processed_files:
            continue
        processed_files.add(file_path)
        df = process_function(file_path, nrows=nrows)
        combined_df = pd.concat([combined_df, df], ignore_index=True)
        logging.info(f"Data from {file_path} processed and added to combined DataFrame.")

    # Scale and save data if key is 'UNITSOLUTION'
    if key == 'UNITSOLUTION':
        scale_and_save_data(combined_df, key)
    elif key == 'INTERCONNECTORSOLN':
        scale_and_save_data(combined_df, key)
    elif key == 'CONSTRAINTSOLUTION':
        scale_and_save_data(combined_df, key)
    elif key == 'REGIONSOLUTION':
        scale_and_save_data(combined_df, key)

    # Return paths of processed files, filtered by the key if provided
    if key:
        return [f'output/{key.lower()}.csv']
    else:
        return [f'output/{k.lower()}.csv' for k in process_map]

read_and_preprocess(directory='data', key='REGIONSOLUTION', nrows=100)
