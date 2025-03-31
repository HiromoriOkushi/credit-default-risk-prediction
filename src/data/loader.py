import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_application_data(file_path, is_train=True):
    """
    Load application train/test data from the given file path.
    
    Parameters:
    -----------
    file_path : str
        Base directory path where the data is stored
    is_train : bool, default=True
        If True, load the training data, else load the test data
        
    Returns:
    --------
    pandas.DataFrame
        Loaded DataFrame containing application data
    """
    file_name = "application_train.csv" if is_train else "application_test.csv"
    file_path = os.path.join(file_path, file_name)
    
    print(f"Loading {file_name} from {file_path}")
    return pd.read_csv(file_path)

def load_raw_data(file_path, **kwargs):
    """
    Load raw loan data from various sources (CSV, database, API).
    
    Parameters:
    -----------
    file_path : str
        Path to the data file
    **kwargs : dict
        Additional arguments to pass to the pandas read_csv function
        
    Returns:
    --------
    pandas.DataFrame
        Loaded DataFrame containing raw data
        
    Raises:
    -------
    ValueError
        If file format is not supported
    """
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path, **kwargs)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")

def split_data(data, test_size=0.2, val_size=0.25, random_state=42, time_column=None):
    """
    Split data into train/validation/test sets, with option for time-based splitting.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame to split
    test_size : float, default=0.2
        Proportion of the dataset to include in the test split
    val_size : float, default=0.25
        Proportion of the training data to include in the validation split
    random_state : int, default=42
        Random seed for reproducibility
    time_column : str, default=None
        If provided, perform time-based splitting using this column
        
    Returns:
    --------
    tuple
        (X_train, X_val, X_test, y_train, y_val, y_test) if 'TARGET' is in data
        (X_train, X_val, X_test) if 'TARGET' is not in data
    """
    if 'TARGET' in data.columns:
        X = data.drop('TARGET', axis=1)
        y = data['TARGET']
    else:
        X = data
        y = None
    
    if time_column:
        # Sort data by time
        X = X.sort_values(by=time_column)
        if y is not None:
            y = y.loc[X.index]
        
        # Split into train+val and test
        train_val_size = int(len(X) * (1 - test_size))
        X_train_val, X_test = X.iloc[:train_val_size], X.iloc[train_val_size:]
        
        if y is not None:
            y_train_val, y_test = y.iloc[:train_val_size], y.iloc[train_val_size:]
            
            # Split train+val into train and val
            val_size_adjusted = val_size / (1 - test_size)  # Adjust val_size relative to train+val
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_val, y_train_val, test_size=val_size_adjusted, random_state=random_state
            )
            
            return X_train, X_val, X_test, y_train, y_val, y_test
        else:
            # Split train+val into train and val
            val_size_adjusted = val_size / (1 - test_size)  # Adjust val_size relative to train+val
            X_train, X_val = train_test_split(
                X_train_val, test_size=val_size_adjusted, random_state=random_state
            )
            
            return X_train, X_val, X_test
    else:
        # Random splitting
        if y is not None:
            # First split into train+val and test
            X_train_val, X_test, y_train_val, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            
            # Then split train+val into train and val
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_val, y_train_val, test_size=val_size, random_state=random_state, stratify=y_train_val
            )
            
            return X_train, X_val, X_test, y_train, y_val, y_test
        else:
            # First split into train+val and test
            X_train_val, X_test = train_test_split(
                X, test_size=test_size, random_state=random_state
            )
            
            # Then split train+val into train and val
            X_train, X_val = train_test_split(
                X_train_val, test_size=val_size, random_state=random_state
            )
            
            return X_train, X_val, X_test

def save_processed_data(data, output_path, file_format='csv'):
    """
    Save processed data to disk.
    
    Parameters:
    -----------
    data : pandas.DataFrame or tuple of DataFrames
        Data to save. If tuple, each DataFrame will be saved with a suffix
    output_path : str
        Path where to save the data
    file_format : str, default='csv'
        Format to save the data in. Currently supports 'csv' and 'pickle'
        
    Returns:
    --------
    list
        List of paths where the data was saved
    
    Raises:
    -------
    ValueError
        If file format is not supported
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    saved_paths = []
    
    # If data is a tuple, save each DataFrame with a suffix
    if isinstance(data, tuple):
        suffixes = ['_train', '_val', '_test', '_train_target', '_val_target', '_test_target']
        for i, df in enumerate(data):
            if df is not None:
                if i < len(suffixes):
                    suffix = suffixes[i]
                else:
                    suffix = f"_part{i}"
                
                file_path = f"{output_path}{suffix}"
                
                if file_format == 'csv':
                    df.to_csv(f"{file_path}.csv", index=False)
                    saved_paths.append(f"{file_path}.csv")
                elif file_format == 'pickle':
                    df.to_pickle(f"{file_path}.pkl")
                    saved_paths.append(f"{file_path}.pkl")
                else:
                    raise ValueError(f"Unsupported file format: {file_format}")
    else:
        # Save single DataFrame
        if file_format == 'csv':
            data.to_csv(f"{output_path}.csv", index=False)
            saved_paths.append(f"{output_path}.csv")
        elif file_format == 'pickle':
            data.to_pickle(f"{output_path}.pkl")
            saved_paths.append(f"{output_path}.pkl")
        else:
            raise ValueError(f"Unsupported file format: {file_format}")
    
    return saved_paths

def load_all_data(base_path):
    """
    Load all dataset files into a dictionary of DataFrames.
    
    Parameters:
    -----------
    base_path : str
        Base directory path where all data files are stored
        
    Returns:
    --------
    dict
        Dictionary with file names as keys and DataFrames as values
    """
    files = {
        'application_train': 'application_train.csv',
        'application_test': 'application_test.csv',
        'bureau': 'bureau.csv',
        'bureau_balance': 'bureau_balance.csv',
        'pos_cash_balance': 'POS_CASH_balance.csv',
        'credit_card_balance': 'credit_card_balance.csv',
        'previous_application': 'previous_application.csv',
        'installments_payments': 'installments_payments.csv',
        'columns_description': 'HomeCredit_columns_description.csv'
    }
    
    data = {}
    for key, filename in files.items():
        file_path = os.path.join(base_path, filename)
        try:
            data[key] = pd.read_csv(file_path)
            print(f"Loaded {filename}: {data[key].shape[0]} rows, {data[key].shape[1]} columns")
        except Exception as e:
            print(f"Error loading {filename}: {e}")
    
    return data

def merge_datasets(main_df, secondary_dfs, join_keys, suffixes=None):
    """
    Merge multiple DataFrames with the main DataFrame.
    
    Parameters:
    -----------
    main_df : pandas.DataFrame
        Main DataFrame to merge other DataFrames into
    secondary_dfs : list of pandas.DataFrame
        List of secondary DataFrames to merge
    join_keys : list of str or list of tuples
        Keys to join on. Each element corresponds to a DataFrame in secondary_dfs.
        If tuple, first element is the key in main_df, second is the key in secondary_df.
    suffixes : list of str, default=None
        Suffixes to use for each merge. If None, defaults to _main and _secondary
        
    Returns:
    --------
    pandas.DataFrame
        Merged DataFrame
    """
    result = main_df.copy()
    
    if suffixes is None:
        suffixes = ['_main', '_secondary']
    
    for i, secondary_df in enumerate(secondary_dfs):
        join_key = join_keys[i]
        
        # Handle case where join_key is a tuple (different column names in main and secondary)
        if isinstance(join_key, tuple):
            main_key, secondary_key = join_key
            # Rename secondary key temporarily
            temp_df = secondary_df.copy()
            temp_df.rename(columns={secondary_key: main_key}, inplace=True)
            result = pd.merge(result, temp_df, on=main_key, how='left', suffixes=suffixes)
        else:
            result = pd.merge(result, secondary_df, on=join_key, how='left', suffixes=suffixes)
    
    return result