import pandas as pd
import os
import logging
from sklearn.preprocessing import MinMaxScaler
from joblib import dump
import multiprocessing


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_and_save(file_path, process_function, output_file, nrows=None):
    df = process_function(file_path, nrows)
    file_exists = os.path.exists(output_file)
    df.to_csv(output_file, mode='a', header=not file_exists, index=False)
    if file_exists:
        logging.debug(f"{os.path.basename(file_path)} appended to {output_file}")
    else:
        logging.debug(f"{os.path.basename(file_path)} saved to {output_file}")

def process_constraint_solution(file_path, nrows=None):
    df = pd.read_csv(file_path, header=1, nrows=nrows)
    df['INTERVAL_DATETIME'] = pd.to_datetime(df['INTERVAL_DATETIME'], format='%Y/%m/%d %H:%M:%S')
    selected_columns = ['INTERVAL_DATETIME', 'CONSTRAINTID', 'RHS', 'MARGINALVALUE', 'VIOLATIONDEGREE']
    return df[selected_columns]

def process_interconnector_solution(file_path, nrows=None):
    df = pd.read_csv(file_path, header=1, nrows=nrows)
    df['INTERVAL_DATETIME'] = pd.to_datetime(df['INTERVAL_DATETIME'], format='%Y/%m/%d %H:%M:%S')
    # Selecting the most relevant columns based on the correlation analysis
    selected_columns = [
        'INTERVAL_DATETIME', 'INTERCONNECTORID', 'METEREDMWFLOW', 'MWFLOW', 
        'EXPORTLIMIT', 'IMPORTLIMIT', 'MWLOSSES', 'MARGINALLOSS', 'FCASEXPORTLIMIT', 'FCASIMPORTLIMIT'
    ]
    return df[selected_columns]

def process_region_solution(file_path, nrows=None):
    df = pd.read_csv(file_path, header=1, nrows=nrows)
    df['INTERVAL_DATETIME'] = pd.to_datetime(df['INTERVAL_DATETIME'], format='%Y/%m/%d %H:%M:%S')
    # Updated selected columns based on top correlations from both analyses
    selected_columns = [
        'INTERVAL_DATETIME', 'REGIONID', 'RRP', 'TOTALDEMAND', 'AVAILABLEGENERATION', 
        'DEMANDFORECAST', 'NETINTERCHANGE', 'ROP', 'RAISE5MINRRP', 'RAISE5MINROP', 
        'LOWER60SECRRP', 'LOWER60SECROP', 'RAISEREGRRP', 'RAISEREGROP', 
        'DEMAND_AND_NONSCHEDGEN'
    ]
    return df[selected_columns]

def process_unit_solution(file_path, nrows=None):
    df = pd.read_csv(file_path, header=1, nrows=nrows)
    df['INTERVAL_DATETIME'] = pd.to_datetime(df['INTERVAL_DATETIME'], format='%Y/%m/%d %H:%M:%S')
    
    # Extracting temporal features
    df['HOUR'] = df['INTERVAL_DATETIME'].dt.hour
    df['DAY_OF_WEEK'] = df['INTERVAL_DATETIME'].dt.dayofweek
    df['MONTH'] = df['INTERVAL_DATETIME'].dt.month

    selected_columns = [
        'INTERVAL_DATETIME', 'HOUR', 'DAY_OF_WEEK', 'MONTH', 'DUID', 'INITIALMW', 'TOTALCLEARED', 'RAMPDOWNRATE', 
        'RAMPUPRATE', 'AVAILABILITY', 'LOWER5MIN', 'LOWER60SEC', 'LOWER6SEC', 
        'RAISE5MIN', 'RAISE60SEC', 'RAISE6SEC', 'AGCSTATUS', 'UIGF', 'CONFORMANCE_MODE'
    ]
    df_selected = df[selected_columns]

    # Normalize numerical columns
    numerical_cols = ['INITIALMW', 'TOTALCLEARED', 'RAMPDOWNRATE', 'RAMPUPRATE', 'AVAILABILITY', 
                      'LOWER5MIN', 'LOWER60SEC', 'LOWER6SEC', 'RAISE5MIN', 'RAISE60SEC', 'RAISE6SEC']
    scaler = MinMaxScaler()
    df_selected[numerical_cols] = scaler.fit_transform(df_selected[numerical_cols].to_numpy())

    dump(scaler, 'output/unit_solution_scaler.joblib')

    return df_selected

def read_and_preprocess(directory, data_type, limit_rows=None):
    data_type = data_type.lower()  # Convert data_type to lowercase
    process_map = {
        'constraint_solution': process_constraint_solution,
        'interconnector_solution': process_interconnector_solution,
        'region_solution': process_region_solution,
        'unit_solution': process_unit_solution
    }
    
    if data_type not in process_map:
        raise ValueError(f"Unsupported data type: {data_type}")

    files_to_process = [os.path.join(directory, f) for f in os.listdir(directory) if data_type.upper() in f]
    output_file = f'output/{data_type}.csv'
    
    def worker(file_path):
        process_and_save(file_path, process_map[data_type], output_file, nrows=limit_rows)
        logging.debug(f"Processed {file_path} for {data_type}")

    # Set up a pool of processes
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    pool.map(worker, files_to_process)
    pool.close()
    pool.join()

    return f'output/{data_type}.csv'