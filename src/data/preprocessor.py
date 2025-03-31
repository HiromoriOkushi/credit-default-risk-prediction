import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from scipy import stats


def identify_column_types(df):
    """
    Identify numerical and categorical columns in a DataFrame.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame
        
    Returns:
    --------
    tuple
        (numerical_columns, categorical_columns)
    """
    # Identify numerical columns (int or float)
    numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Identify categorical columns (object or category)
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    return numerical_columns, categorical_columns


def get_missing_values_summary(df):
    """
    Get summary of missing values in a DataFrame.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with columns, missing count, and percentage
    """
    # Calculate missing values
    missing = df.isnull().sum()
    missing_percent = 100 * missing / len(df)
    
    # Create summary DataFrame
    missing_summary = pd.DataFrame({
        'Column': missing.index,
        'Missing Values': missing.values,
        'Missing Percentage': missing_percent.values
    })
    
    # Sort by missing percentage (descending)
    missing_summary = missing_summary.sort_values('Missing Percentage', ascending=False)
    missing_summary = missing_summary[missing_summary['Missing Values'] > 0]
    
    return missing_summary


def handle_missing_values(df, strategy='median', columns=None):
    """
    Impute missing values based on specified strategy.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with missing values
    strategy : str, default='median'
        Strategy for imputation: 'mean', 'median', 'most_frequent', 'constant'
    columns : list, default=None
        List of columns to impute. If None, impute all columns
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with imputed values
    """
    result_df = df.copy()
    
    # If no columns specified, identify numerical and categorical columns
    if columns is None:
        numerical_columns, categorical_columns = identify_column_types(df)
    else:
        # Split specified columns into numerical and categorical
        numerical_columns = [col for col in columns if col in df.select_dtypes(include=['int64', 'float64']).columns]
        categorical_columns = [col for col in columns if col in df.select_dtypes(include=['object', 'category']).columns]
    
    # Impute numerical columns
    if numerical_columns:
        if strategy in ['mean', 'median']:
            imputer = SimpleImputer(strategy=strategy)
            result_df[numerical_columns] = imputer.fit_transform(result_df[numerical_columns])
        elif strategy == 'constant':
            # Fill with 0 for numerical columns
            result_df[numerical_columns] = result_df[numerical_columns].fillna(0)
    
    # Impute categorical columns
    if categorical_columns:
        if strategy == 'most_frequent':
            cat_imputer = SimpleImputer(strategy='most_frequent')
            result_df[categorical_columns] = cat_imputer.fit_transform(result_df[categorical_columns])
        elif strategy == 'constant':
            # Fill with 'Unknown' for categorical columns
            result_df[categorical_columns] = result_df[categorical_columns].fillna('Unknown')
    
    return result_df


def handle_outliers(df, method='winsorize', columns=None, limits=(0.05, 0.95)):
    """
    Detect and handle outliers in numerical columns.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame
    method : str, default='winsorize'
        Method to handle outliers: 'winsorize', 'trim', 'zscore'
    columns : list, default=None
        List of numerical columns to process. If None, process all numerical columns
    limits : tuple, default=(0.05, 0.95)
        Tuple of (lower percentile, upper percentile) for winsorization
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with outliers handled
    """
    result_df = df.copy()
    
    # If no columns specified, use all numerical columns
    if columns is None:
        numerical_columns, _ = identify_column_types(df)
        columns = numerical_columns
    else:
        # Ensure columns are numerical
        columns = [col for col in columns if col in df.select_dtypes(include=['int64', 'float64']).columns]
    
    if method == 'winsorize':
        # Winsorize: cap extreme values at specified percentiles
        for col in columns:
            if result_df[col].isnull().all():
                continue
                
            lower_limit = np.nanpercentile(result_df[col], limits[0] * 100)
            upper_limit = np.nanpercentile(result_df[col], limits[1] * 100)
            result_df[col] = result_df[col].clip(lower=lower_limit, upper=upper_limit)
    
    elif method == 'trim':
        # Trim: remove rows with outliers
        for col in columns:
            if result_df[col].isnull().all():
                continue
                
            lower_limit = np.nanpercentile(result_df[col], limits[0] * 100)
            upper_limit = np.nanpercentile(result_df[col], limits[1] * 100)
            result_df = result_df[(result_df[col] >= lower_limit) | 
                                  (result_df[col].isnull()) | 
                                  (result_df[col] <= upper_limit)]
    
    elif method == 'zscore':
        # Z-score: remove or cap values beyond specified standard deviations
        z_threshold = 3  # Common threshold for outlier detection
        for col in columns:
            if result_df[col].isnull().all():
                continue
                
            # Calculate z-scores with handling for NaN values
            z_scores = np.abs(stats.zscore(result_df[col], nan_policy='omit'))
            outliers = z_scores > z_threshold
            # Replace outliers with NaN (to be imputed later)
            result_df.loc[outliers, col] = np.nan
    
    return result_df


def encode_categorical_features(df, method='one-hot', columns=None, drop_first=True):
    """
    Encode categorical variables using specified method.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame
    method : str, default='one-hot'
        Encoding method: 'one-hot', 'label', 'binary'
    columns : list, default=None
        List of categorical columns to encode. If None, encode all categorical columns
    drop_first : bool, default=True
        Whether to drop the first category in one-hot encoding
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with encoded categorical features
    """
    result_df = df.copy()
    
    # If no columns specified, use all categorical columns
    if columns is None:
        _, categorical_columns = identify_column_types(df)
        columns = categorical_columns
    else:
        # Ensure columns are categorical
        columns = [col for col in columns if col in df.select_dtypes(include=['object', 'category']).columns]
    
    if method == 'one-hot':
        # One-hot encoding
        for col in columns:
            # Get dummies and drop the first category if requested
            dummies = pd.get_dummies(result_df[col], prefix=col, drop_first=drop_first)
            result_df = pd.concat([result_df, dummies], axis=1)
            result_df.drop(col, axis=1, inplace=True)
    
    elif method == 'label':
        # Label encoding
        for col in columns:
            le = LabelEncoder()
            # Handle NaN values by filling them first
            has_null = result_df[col].isnull().any()
            if has_null:
                result_df[col] = result_df[col].fillna('Unknown')
            result_df[col] = le.fit_transform(result_df[col].astype(str))
    
    elif method == 'binary':
        # Binary encoding for high cardinality categoricals
        for col in columns:
            # Handle NaN values
            has_null = result_df[col].isnull().any()
            if has_null:
                result_df[col] = result_df[col].fillna('Unknown')
                
            # Convert to category codes
            result_df[col] = result_df[col].astype('category').cat.codes
            # Convert to binary representation
            max_val = result_df[col].max()
            if max_val > 0:  # Skip if all values are the same
                num_bits = int(np.log2(max_val)) + 1
                
                # Create binary columns
                for bit in range(num_bits):
                    bit_name = f"{col}_bit{bit}"
                    result_df[bit_name] = ((result_df[col] >> bit) & 1).astype(int)
                
                # Drop original column
                result_df.drop(col, axis=1, inplace=True)
    
    return result_df


def handle_rare_categories(df, columns=None, threshold=0.01):
    """
    Handle rare categories in categorical columns by grouping them.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame
    columns : list, default=None
        List of categorical columns to process. If None, process all categorical columns
    threshold : float, default=0.01
        Threshold below which categories are considered rare (as a fraction of total)
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with rare categories grouped
    """
    result_df = df.copy()
    
    # If no columns specified, use all categorical columns
    if columns is None:
        _, categorical_columns = identify_column_types(df)
        columns = categorical_columns
    else:
        # Ensure columns are categorical
        columns = [col for col in columns if col in df.select_dtypes(include=['object', 'category']).columns]
    
    for col in columns:
        # Calculate value counts and proportions
        value_counts = result_df[col].value_counts(dropna=False)
        proportions = value_counts / len(result_df)
        
        # Identify rare categories
        rare_categories = proportions[proportions < threshold].index.tolist()
        
        # Skip if no rare categories or all categories are rare
        if not rare_categories or len(rare_categories) == len(proportions):
            continue
        
        # Replace rare categories with 'Other'
        result_df[col] = result_df[col].apply(lambda x: 'Other' if x in rare_categories else x)
    
    return result_df


def normalize_features(df, method='standard', columns=None):
    """
    Normalize/standardize numerical features.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame
    method : str, default='standard'
        Normalization method: 'standard', 'minmax', 'robust'
    columns : list, default=None
        List of numerical columns to normalize. If None, normalize all numerical columns
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with normalized features
    """
    result_df = df.copy()
    
    # If no columns specified, use all numerical columns
    if columns is None:
        numerical_columns, _ = identify_column_types(df)
        columns = numerical_columns
    else:
        # Ensure columns are numerical
        columns = [col for col in columns if col in df.select_dtypes(include=['int64', 'float64']).columns]
    
    # Filter out columns with all missing values
    columns = [col for col in columns if not result_df[col].isnull().all()]
    
    if method == 'standard':
        # Standardization (z-score normalization)
        scaler = StandardScaler()
        result_df[columns] = scaler.fit_transform(result_df[columns])
    
    elif method == 'minmax':
        # Min-Max normalization
        scaler = MinMaxScaler()
        result_df[columns] = scaler.fit_transform(result_df[columns])
    
    elif method == 'robust':
        # Robust scaling - less sensitive to outliers
        scaler = RobustScaler()
        result_df[columns] = scaler.fit_transform(result_df[columns])
    
    return result_df


def detect_anomalies(df, method='iqr', columns=None, k=1.5):
    """
    Detect anomalous values that might be errors rather than outliers.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame
    method : str, default='iqr'
        Method to detect anomalies: 'iqr', 'zscore'
    columns : list, default=None
        List of numerical columns to check. If None, check all numerical columns
    k : float, default=1.5
        Coefficient for IQR method
        
    Returns:
    --------
    dict
        Dictionary with columns as keys and lists of anomalous indices as values
    """
    # If no columns specified, use all numerical columns
    if columns is None:
        numerical_columns, _ = identify_column_types(df)
        columns = numerical_columns
    else:
        # Ensure columns are numerical
        columns = [col for col in columns if col in df.select_dtypes(include=['int64', 'float64']).columns]
    
    anomalies = {}
    
    for col in columns:
        # Skip columns with all missing values
        if df[col].isnull().all():
            continue
        
        if method == 'iqr':
            # IQR method
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            
            lower_bound = q1 - k * iqr
            upper_bound = q3 + k * iqr
            
            # Find indices of anomalies
            anomaly_indices = df.index[(df[col] < lower_bound) | (df[col] > upper_bound)].tolist()
            
        elif method == 'zscore':
            # Z-score method
            z_scores = np.abs(stats.zscore(df[col], nan_policy='omit'))
            anomaly_indices = df.index[z_scores > 3].tolist()  # Common threshold is 3 std deviations
        
        if anomaly_indices:
            anomalies[col] = anomaly_indices
    
    return anomalies


def create_binary_features(df, columns=None, thresholds=None):
    """
    Create binary features from numerical variables.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame
    columns : list, default=None
        List of numerical columns to process. If None, process all numerical columns
    thresholds : dict, default=None
        Dictionary mapping column names to thresholds. If None, use median as threshold
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with additional binary features
    """
    result_df = df.copy()
    
    # If no columns specified, use all numerical columns
    if columns is None:
        numerical_columns, _ = identify_column_types(df)
        columns = numerical_columns
    else:
        # Ensure columns are numerical
        columns = [col for col in columns if col in df.select_dtypes(include=['int64', 'float64']).columns]
    
    # Initialize thresholds if not provided
    if thresholds is None:
        thresholds = {}
    
    for col in columns:
        # Skip columns with all missing values
        if result_df[col].isnull().all():
            continue
        
        # Use provided threshold or calculate median
        threshold = thresholds.get(col, result_df[col].median())
        
        # Create binary feature
        binary_col_name = f"{col}_binary"
        result_df[binary_col_name] = (result_df[col] > threshold).astype(int)
    
    return result_df


def preprocess_pipeline(df, config):
    """
    Run complete preprocessing pipeline based on configuration.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame
    config : dict
        Configuration dictionary with preprocessing steps and parameters
        Example:
        {
            'missing_values': {'strategy': 'median', 'columns': None},
            'outliers': {'method': 'winsorize', 'columns': None, 'limits': (0.05, 0.95)},
            'rare_categories': {'threshold': 0.01, 'columns': None},
            'categorical_encoding': {'method': 'one-hot', 'columns': None, 'drop_first': True},
            'normalization': {'method': 'standard', 'columns': None},
            'binary_features': {'columns': None, 'thresholds': None}
        }
        
    Returns:
    --------
    pandas.DataFrame
        Preprocessed DataFrame
    """
    result_df = df.copy()
    
    # Apply preprocessing steps based on configuration
    
    # 1. Handle missing values
    if 'missing_values' in config:
        missing_config = config['missing_values']
        result_df = handle_missing_values(
            result_df, 
            strategy=missing_config.get('strategy', 'median'),
            columns=missing_config.get('columns', None)
        )
    
    # 2. Handle outliers
    if 'outliers' in config:
        outliers_config = config['outliers']
        result_df = handle_outliers(
            result_df,
            method=outliers_config.get('method', 'winsorize'),
            columns=outliers_config.get('columns', None),
            limits=outliers_config.get('limits', (0.05, 0.95))
        )
    
    # 3. Handle rare categories
    if 'rare_categories' in config:
        rare_config = config['rare_categories']
        result_df = handle_rare_categories(
            result_df,
            columns=rare_config.get('columns', None),
            threshold=rare_config.get('threshold', 0.01)
        )
    
    # 4. Encode categorical features
    if 'categorical_encoding' in config:
        cat_config = config['categorical_encoding']
        result_df = encode_categorical_features(
            result_df,
            method=cat_config.get('method', 'one-hot'),
            columns=cat_config.get('columns', None),
            drop_first=cat_config.get('drop_first', True)
        )
    
    # 5. Normalize numerical features
    if 'normalization' in config:
        norm_config = config['normalization']
        result_df = normalize_features(
            result_df,
            method=norm_config.get('method', 'standard'),
            columns=norm_config.get('columns', None)
        )
    
    # 6. Create binary features
    if 'binary_features' in config:
        binary_config = config['binary_features']
        result_df = create_binary_features(
            result_df,
            columns=binary_config.get('columns', None),
            thresholds=binary_config.get('thresholds', None)
        )
    
    return result_df