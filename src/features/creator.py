import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import itertools
from sklearn.preprocessing import PolynomialFeatures


def create_financial_ratios(df):
    """
    Calculate financial ratios from existing numerical features.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing financial data
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with additional financial ratio features
    """
    result_df = df.copy()
    
    # Common financial ratios for credit risk assessment
    
    # 1. Debt-to-Income ratio (DTI)
    if 'AMT_INCOME_TOTAL' in result_df.columns and 'AMT_CREDIT' in result_df.columns:
        # Avoid division by zero
        result_df['RATIO_DEBT_INCOME'] = result_df['AMT_CREDIT'] / result_df['AMT_INCOME_TOTAL'].replace(0, np.nan)
    
    # 2. Annuity to Income ratio
    if 'AMT_INCOME_TOTAL' in result_df.columns and 'AMT_ANNUITY' in result_df.columns:
        result_df['RATIO_ANNUITY_INCOME'] = result_df['AMT_ANNUITY'] / result_df['AMT_INCOME_TOTAL'].replace(0, np.nan)
    
    # 3. Credit to Goods price ratio
    if 'AMT_CREDIT' in result_df.columns and 'AMT_GOODS_PRICE' in result_df.columns:
        result_df['RATIO_CREDIT_GOODS'] = result_df['AMT_CREDIT'] / result_df['AMT_GOODS_PRICE'].replace(0, np.nan)
    
    # 4. Payment to Credit ratio
    if 'AMT_CREDIT' in result_df.columns and 'AMT_ANNUITY' in result_df.columns:
        result_df['RATIO_PAYMENT_CREDIT'] = result_df['AMT_ANNUITY'] / result_df['AMT_CREDIT'].replace(0, np.nan)
    
    # 5. Income per family member
    if 'AMT_INCOME_TOTAL' in result_df.columns and 'CNT_FAM_MEMBERS' in result_df.columns:
        result_df['INCOME_PER_FAMILY_MEMBER'] = result_df['AMT_INCOME_TOTAL'] / result_df['CNT_FAM_MEMBERS'].replace(0, np.nan)
    
    # 6. Credit per family member
    if 'AMT_CREDIT' in result_df.columns and 'CNT_FAM_MEMBERS' in result_df.columns:
        result_df['CREDIT_PER_FAMILY_MEMBER'] = result_df['AMT_CREDIT'] / result_df['CNT_FAM_MEMBERS'].replace(0, np.nan)
    
    # 7. Credit to income month ratio (how many months of income needed to pay off credit)
    if 'AMT_INCOME_TOTAL' in result_df.columns and 'AMT_CREDIT' in result_df.columns:
        result_df['MONTHS_INCOME_FOR_CREDIT'] = result_df['AMT_CREDIT'] / (result_df['AMT_INCOME_TOTAL'] / 12).replace(0, np.nan)
    
    # 8. Loan duration in months (if available)
    if 'AMT_CREDIT' in result_df.columns and 'AMT_ANNUITY' in result_df.columns:
        # Simple approximation of loan duration: Credit / Annuity
        result_df['LOAN_DURATION_MONTHS'] = result_df['AMT_CREDIT'] / result_df['AMT_ANNUITY'].replace(0, np.nan)
    
    # Handle infinite values
    for col in result_df.columns:
        if col.startswith('RATIO_') or col.endswith('_MEMBER') or col.endswith('_MONTHS'):
            result_df[col] = result_df[col].replace([np.inf, -np.inf], np.nan)
    
    return result_df


def create_time_features(df, date_column):
    """
    Generate time-based features from a date column.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the date column
    date_column : str
        Name of the column containing date information
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with additional time-based features
    """
    result_df = df.copy()
    
    # Skip if the date column doesn't exist
    if date_column not in result_df.columns:
        print(f"Warning: {date_column} not found in the DataFrame")
        return result_df
    
    # Convert to datetime if not already
    if not pd.api.types.is_datetime64_dtype(result_df[date_column]):
        try:
            result_df[date_column] = pd.to_datetime(result_df[date_column])
        except Exception as e:
            print(f"Error converting {date_column} to datetime: {e}")
            return result_df
    
    # Extract basic time components
    date_prefix = f"{date_column}_"
    result_df[f"{date_prefix}YEAR"] = result_df[date_column].dt.year
    result_df[f"{date_prefix}MONTH"] = result_df[date_column].dt.month
    result_df[f"{date_prefix}DAY"] = result_df[date_column].dt.day
    result_df[f"{date_prefix}DAYOFWEEK"] = result_df[date_column].dt.dayofweek
    result_df[f"{date_prefix}DAYOFYEAR"] = result_df[date_column].dt.dayofyear
    result_df[f"{date_prefix}QUARTER"] = result_df[date_column].dt.quarter
    result_df[f"{date_prefix}IS_WEEKEND"] = result_df[date_column].dt.dayofweek.isin([5, 6]).astype(int)
    
    # Calculate days relative to current date (if needed)
    current_date = datetime.now()
    result_df[f"{date_prefix}DAYS_SINCE"] = (current_date - result_df[date_column]).dt.days
    
    # Calculate days between application date and other dates (if they exist)
    if 'DAYS_BIRTH' in result_df.columns:
        # Often days are stored as negative values (days before application)
        # Calculate age in years more precisely
        result_df['APPLICANT_AGE_YEARS'] = -result_df['DAYS_BIRTH'] / 365.25
    
    if 'DAYS_EMPLOYED' in result_df.columns:
        # Calculate employment duration in years
        result_df['EMPLOYMENT_YEARS'] = -result_df['DAYS_EMPLOYED'] / 365.25
        
        # Ratio of employment length to age
        if 'DAYS_BIRTH' in result_df.columns:
            # Calculate employment to age ratio (% of life employed)
            result_df['EMPLOYMENT_TO_AGE_RATIO'] = (
                result_df['DAYS_EMPLOYED'] / result_df['DAYS_BIRTH']
            ).replace([np.inf, -np.inf], np.nan)
    
    # Check if there are other DAYS_* columns that could be used for relative time calculations
    days_columns = [col for col in result_df.columns if col.startswith('DAYS_') and col not in ['DAYS_BIRTH', 'DAYS_EMPLOYED']]
    
    for days_col in days_columns:
        feature_name = days_col.replace('DAYS_', '')
        # Calculate years
        result_df[f"{feature_name}_YEARS"] = -result_df[days_col] / 365.25
    
    # Seasonality features
    result_df[f"{date_prefix}SIN_MONTH"] = np.sin(2 * np.pi * result_df[date_column].dt.month / 12)
    result_df[f"{date_prefix}COS_MONTH"] = np.cos(2 * np.pi * result_df[date_column].dt.month / 12)
    result_df[f"{date_prefix}SIN_DAY"] = np.sin(2 * np.pi * result_df[date_column].dt.day / 31)
    result_df[f"{date_prefix}COS_DAY"] = np.cos(2 * np.pi * result_df[date_column].dt.day / 31)
    
    return result_df


def create_interaction_features(df, features_list=None, max_degree=2, interaction_only=True):
    """
    Create interaction terms between features.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing features
    features_list : list, default=None
        List of features to create interactions for. If None, uses all numerical features
    max_degree : int, default=2
        Maximum degree of polynomial features
    interaction_only : bool, default=True
        Whether to include only interaction terms or also powers
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with additional interaction features
    """
    result_df = df.copy()
    
    # If features_list is None, use all numerical features
    if features_list is None:
        features_list = result_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        # Exclude target variable and ID columns
        exclude_cols = ['SK_ID_CURR', 'SK_ID_PREV', 'SK_ID_BUREAU', 'TARGET']
        features_list = [col for col in features_list if col not in exclude_cols]
    
    # Handle case with empty or invalid features_list
    if not features_list or len(features_list) < 2:
        print("Warning: Not enough features for interactions. Need at least 2.")
        return result_df
    
    # Ensure all features in features_list exist in the DataFrame
    valid_features = [f for f in features_list if f in result_df.columns]
    if len(valid_features) < 2:
        print("Warning: Not enough valid features for interactions. Need at least 2.")
        return result_df
    
    # For manually creating specific interactions (pairwise)
    if max_degree == 2 and interaction_only:
        # Create all pairwise interactions
        for feat1, feat2 in itertools.combinations(valid_features, 2):
            interaction_name = f"{feat1}_x_{feat2}"
            result_df[interaction_name] = result_df[feat1] * result_df[feat2]
    else:
        # Use PolynomialFeatures for more complex interactions
        try:
            # Extract the selected features
            features_df = result_df[valid_features].copy()
            
            # Replace NaN values with 0 for polynomial feature generation
            features_df = features_df.fillna(0)
            
            # Generate polynomial features
            poly = PolynomialFeatures(degree=max_degree, interaction_only=interaction_only, include_bias=False)
            poly_features = poly.fit_transform(features_df)
            
            # Get the feature names
            feature_names = poly.get_feature_names_out(valid_features)
            
            # Create a DataFrame with the new features
            poly_df = pd.DataFrame(poly_features, columns=feature_names, index=result_df.index)
            
            # Remove the original features from the polynomial DataFrame
            original_feature_cols = [col for col in feature_names if col in valid_features]
            poly_df = poly_df.drop(columns=original_feature_cols)
            
            # Combine with the original DataFrame
            result_df = pd.concat([result_df, poly_df], axis=1)
        except Exception as e:
            print(f"Error generating polynomial features: {e}")
    
    return result_df


def create_aggregated_features(df, group_columns, agg_columns, agg_functions=None):
    """
    Generate aggregated metrics for specified columns grouped by certain keys.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame to aggregate
    group_columns : str or list
        Column(s) to group by
    agg_columns : list
        Columns to aggregate
    agg_functions : dict or list, default=None
        Functions to use for aggregation. If None, uses ['mean', 'min', 'max', 'sum', 'count']
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with aggregated features
    """
    # Ensure group_columns is a list
    if isinstance(group_columns, str):
        group_columns = [group_columns]
    
    # Skip if group_columns or agg_columns are not in df
    missing_group_cols = [col for col in group_columns if col not in df.columns]
    missing_agg_cols = [col for col in agg_columns if col not in df.columns]
    
    if missing_group_cols:
        print(f"Warning: Group columns not found in DataFrame: {missing_group_cols}")
        return None
    
    if missing_agg_cols:
        print(f"Warning: Aggregation columns not found in DataFrame: {missing_agg_cols}")
        valid_agg_cols = [col for col in agg_columns if col in df.columns]
        if not valid_agg_cols:
            return None
        agg_columns = valid_agg_cols
    
    # Default aggregation functions if none provided
    if agg_functions is None:
        agg_functions = ['mean', 'min', 'max', 'sum', 'count']
    
    # Prepare aggregation dictionary
    if isinstance(agg_functions, list):
        agg_dict = {col: agg_functions for col in agg_columns}
    else:
        agg_dict = agg_functions
    
    # Perform groupby and aggregation
    agg_df = df.groupby(group_columns)[agg_columns].agg(agg_dict)
    
    # Flatten multi-index columns if necessary
    if isinstance(agg_df.columns, pd.MultiIndex):
        agg_df.columns = [f"{col[0]}_{col[1]}".upper() for col in agg_df.columns]
    
    # Reset index to make group columns regular columns again
    agg_df = agg_df.reset_index()
    
    return agg_df


def create_bureau_aggregations(bureau_df, bureau_balance_df=None, client_id_col='SK_ID_CURR'):
    """
    Create aggregated features from bureau data for each client.
    
    Parameters:
    -----------
    bureau_df : pandas.DataFrame
        DataFrame containing bureau data
    bureau_balance_df : pandas.DataFrame, default=None
        DataFrame containing bureau balance data (monthly history)
    client_id_col : str, default='SK_ID_CURR'
        Column name for client ID
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with aggregated bureau features for each client
    """
    if client_id_col not in bureau_df.columns:
        print(f"Error: Client ID column '{client_id_col}' not found in bureau DataFrame")
        return None
    
    # Numerical columns to aggregate
    num_cols = bureau_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    num_cols = [col for col in num_cols if col != client_id_col]
    
    # Create aggregations for numerical columns
    agg_functions = {
        'DAYS_CREDIT': ['min', 'max', 'mean', 'var'],
        'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
        'DAYS_CREDIT_UPDATE': ['mean'],
        'CREDIT_DAY_OVERDUE': ['max', 'mean'],
        'AMT_CREDIT_MAX_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
        'AMT_ANNUITY': ['max', 'mean'],
        'CNT_CREDIT_PROLONG': ['sum'],
    }
    
    # Filter to include only columns that exist in the DataFrame
    agg_functions = {k: v for k, v in agg_functions.items() if k in bureau_df.columns}
    
    # Add default aggregations for other numerical columns
    for col in num_cols:
        if col not in agg_functions:
            agg_functions[col] = ['mean', 'min', 'max']
    
    # Aggregate by client ID
    bureau_agg = bureau_df.groupby(client_id_col).agg(agg_functions)
    
    # Flatten column names
    bureau_agg.columns = [f'BUREAU_{col[0]}_{col[1]}'.upper() for col in bureau_agg.columns]
    
    # Reset index
    bureau_agg = bureau_agg.reset_index()
    
    # Create categorical aggregations
    cat_cols = [col for col in bureau_df.columns if bureau_df[col].dtype == 'object']
    
    for col in cat_cols:
        # Count number of each category per client
        cat_count = bureau_df.groupby([client_id_col, col]).size().unstack().fillna(0)
        cat_count.columns = [f'BUREAU_{col}_{category}_COUNT' for category in cat_count.columns]
        
        # Merge with main aggregations
        bureau_agg = bureau_agg.merge(cat_count, on=client_id_col, how='left')
    
    # Add count of bureau records per client
    bureau_counts = bureau_df.groupby(client_id_col).size().reset_index()
    bureau_counts.columns = [client_id_col, 'BUREAU_COUNT']
    bureau_agg = bureau_agg.merge(bureau_counts, on=client_id_col, how='left')
    
    # Process bureau balance data if available
    if bureau_balance_df is not None and 'SK_ID_BUREAU' in bureau_balance_df.columns:
        # First aggregate bureau_balance by SK_ID_BUREAU
        bb_agg = bureau_balance_df.groupby('SK_ID_BUREAU').agg({
            'MONTHS_BALANCE': ['min', 'max', 'mean', 'size'],
            'STATUS': lambda x: x.value_counts().get('C', 0)  # Count of 'C' statuses (closed)
        })
        
        # Flatten column names
        bb_agg.columns = [f'BB_{col[0]}_{col[1]}'.upper() for col in bb_agg.columns]
        bb_agg = bb_agg.reset_index()
        
        # Count different statuses
        status_counts = pd.get_dummies(bureau_balance_df['STATUS'], prefix='BB_STATUS')
        status_agg = bureau_balance_df.join(status_counts).groupby('SK_ID_BUREAU').sum()
        
        # Merge bureau_balance aggregations with bureau data
        bb_agg = bb_agg.merge(status_agg, on='SK_ID_BUREAU', how='left')
        
        # Now merge with bureau data to get client IDs
        bureau_w_bb = bureau_df[['SK_ID_CURR', 'SK_ID_BUREAU']].merge(bb_agg, on='SK_ID_BUREAU', how='left')
        
        # Finally aggregate to client level
        client_bb_agg = bureau_w_bb.groupby('SK_ID_CURR').agg({col: ['mean', 'sum', 'max'] 
                                                              for col in bb_agg.columns 
                                                              if col != 'SK_ID_BUREAU'})
        
        # Flatten column names
        client_bb_agg.columns = [f'{col[0]}_{col[1]}'.upper() for col in client_bb_agg.columns]
        client_bb_agg = client_bb_agg.reset_index()
        
        # Merge with bureau aggregations
        bureau_agg = bureau_agg.merge(client_bb_agg, on=client_id_col, how='left')
    
    return bureau_agg


def create_previous_app_aggregations(prev_app_df, installments_df=None, client_id_col='SK_ID_CURR'):
    """
    Create aggregated features from previous applications data for each client.
    
    Parameters:
    -----------
    prev_app_df : pandas.DataFrame
        DataFrame containing previous applications data
    installments_df : pandas.DataFrame, default=None
        DataFrame containing installments payment data
    client_id_col : str, default='SK_ID_CURR'
        Column name for client ID
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with aggregated previous applications features for each client
    """
    if client_id_col not in prev_app_df.columns:
        print(f"Error: Client ID column '{client_id_col}' not found in previous applications DataFrame")
        return None
    
    # Numerical columns to aggregate
    num_cols = prev_app_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    num_cols = [col for col in num_cols if col != client_id_col and col != 'SK_ID_PREV']
    
    # Define aggregations for important numerical columns
    agg_functions = {
        'AMT_ANNUITY': ['min', 'max', 'mean'],
        'AMT_APPLICATION': ['min', 'max', 'mean'],
        'AMT_CREDIT': ['min', 'max', 'mean', 'sum'],
        'AMT_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'AMT_GOODS_PRICE': ['min', 'max', 'mean'],
        'HOUR_APPR_PROCESS_START': ['min', 'max', 'mean'],
        'DAYS_DECISION': ['min', 'max', 'mean'],
        'CNT_PAYMENT': ['min', 'max', 'mean', 'sum'],
        'DAYS_FIRST_DRAWING': ['min', 'max', 'mean'],
        'DAYS_FIRST_DUE': ['min', 'max', 'mean'],
        'DAYS_LAST_DUE_1ST_VERSION': ['min', 'max', 'mean'],
        'DAYS_LAST_DUE': ['min', 'max', 'mean'],
        'DAYS_TERMINATION': ['min', 'max', 'mean'],
    }
    
    # Filter to include only columns that exist in the DataFrame
    agg_functions = {k: v for k, v in agg_functions.items() if k in prev_app_df.columns}
    
    # Add default aggregations for other numerical columns
    for col in num_cols:
        if col not in agg_functions:
            agg_functions[col] = ['mean', 'min', 'max']
    
    # Aggregate by client ID
    prev_agg = prev_app_df.groupby(client_id_col).agg(agg_functions)
    
    # Flatten column names
    prev_agg.columns = [f'PREV_{col[0]}_{col[1]}'.upper() for col in prev_agg.columns]
    
    # Reset index
    prev_agg = prev_agg.reset_index()
    
    # Create categorical aggregations
    cat_cols = [col for col in prev_app_df.columns if prev_app_df[col].dtype == 'object']
    
    for col in cat_cols:
        # Count number of each category per client
        cat_count = prev_app_df.groupby([client_id_col, col]).size().unstack().fillna(0)
        cat_count.columns = [f'PREV_{col}_{category}_COUNT' for category in cat_count.columns]
        
        # Merge with main aggregations
        prev_agg = prev_agg.merge(cat_count, on=client_id_col, how='left')
    
    # Add count of previous applications per client
    prev_counts = prev_app_df.groupby(client_id_col).size().reset_index()
    prev_counts.columns = [client_id_col, 'PREV_APP_COUNT']
    prev_agg = prev_agg.merge(prev_counts, on=client_id_col, how='left')
    
    # Calculate approval rate
    if 'NAME_CONTRACT_STATUS' in prev_app_df.columns:
        approved = prev_app_df[prev_app_df['NAME_CONTRACT_STATUS'] == 'Approved'].groupby(client_id_col).size()
        total = prev_app_df.groupby(client_id_col).size()
        approval_rate = (approved / total).fillna(0).reset_index()
        approval_rate.columns = [client_id_col, 'PREV_APPROVAL_RATE']
        prev_agg = prev_agg.merge(approval_rate, on=client_id_col, how='left')
    
    # Process installments data if available
    if installments_df is not None and 'SK_ID_PREV' in installments_df.columns:
        # First aggregate installments by SK_ID_PREV
        inst_agg = installments_df.groupby('SK_ID_PREV').agg({
            'NUM_INSTALMENT_VERSION': ['nunique'],
            'NUM_INSTALMENT_NUMBER': ['max'],
            'DAYS_INSTALMENT': ['min', 'max', 'mean'],
            'DAYS_ENTRY_PAYMENT': ['min', 'max', 'mean'],
            'AMT_INSTALMENT': ['min', 'max', 'mean', 'sum'],
            'AMT_PAYMENT': ['min', 'max', 'mean', 'sum']
        })
        
        # Calculate payment ratio and late payment features
        payment_ratio = installments_df['AMT_PAYMENT'] / installments_df['AMT_INSTALMENT']
        days_late = installments_df['DAYS_ENTRY_PAYMENT'] - installments_df['DAYS_INSTALMENT']
        
        # Handle infinite and NaN values
        payment_ratio = payment_ratio.replace([np.inf, -np.inf], np.nan)
        
        # Add to installments DataFrame
        installments_df['PAYMENT_RATIO'] = payment_ratio
        installments_df['DAYS_LATE'] = days_late
        
        # Aggregate these new features
        payment_agg = installments_df.groupby('SK_ID_PREV').agg({
            'PAYMENT_RATIO': ['min', 'max', 'mean'],
            'DAYS_LATE': ['min', 'max', 'mean', 'sum']
        })
        
        # Merge with installment aggregations
        inst_agg = pd.concat([inst_agg, payment_agg], axis=1)
        
        # Flatten column names
        inst_agg.columns = [f'INST_{col[0]}_{col[1]}'.upper() for col in inst_agg.columns]
        inst_agg = inst_agg.reset_index()
        
        # Now merge with previous applications to get client IDs
        prev_w_inst = prev_app_df[['SK_ID_CURR', 'SK_ID_PREV']].merge(inst_agg, on='SK_ID_PREV', how='left')
        
        # Finally aggregate to client level
        client_inst_agg = prev_w_inst.groupby('SK_ID_CURR').agg({col: ['mean', 'sum', 'max'] 
                                                              for col in inst_agg.columns 
                                                              if col != 'SK_ID_PREV'})
        
        # Flatten column names
        client_inst_agg.columns = [f'{col[0]}_{col[1]}'.upper() for col in client_inst_agg.columns]
        client_inst_agg = client_inst_agg.reset_index()
        
        # Merge with previous application aggregations
        prev_agg = prev_agg.merge(client_inst_agg, on=client_id_col, how='left')
    
    return prev_agg


def create_pos_cash_aggregations(pos_cash_df, client_id_col='SK_ID_CURR'):
    """
    Create aggregated features from POS cash balance data for each client.
    
    Parameters:
    -----------
    pos_cash_df : pandas.DataFrame
        DataFrame containing POS cash balance data
    client_id_col : str, default='SK_ID_CURR'
        Column name for client ID
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with aggregated POS cash features for each client
    """
    if client_id_col not in pos_cash_df.columns:
        print(f"Error: Client ID column '{client_id_col}' not found in POS cash DataFrame")
        return None
    
    # Numerical columns to aggregate
    num_cols = pos_cash_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    num_cols = [col for col in num_cols if col != client_id_col and col != 'SK_ID_PREV']
    
    # Define aggregations
    agg_functions = {
        'MONTHS_BALANCE': ['min', 'max', 'mean', 'size'],
        'CNT_INSTALMENT': ['min', 'max', 'mean'],
        'CNT_INSTALMENT_FUTURE': ['min', 'max', 'mean'],
        'SK_DPD': ['max', 'mean', 'sum'],
        'SK_DPD_DEF': ['max', 'mean', 'sum']
    }
    
    # Filter to include only columns that exist in the DataFrame
    agg_functions = {k: v for k, v in agg_functions.items() if k in pos_cash_df.columns}
    
    # Add default aggregations for other numerical columns
    for col in num_cols:
        if col not in agg_functions:
            agg_functions[col] = ['mean', 'min', 'max']
    
    # First aggregate by SK_ID_PREV to get loan-level aggregations
    pos_prev_agg = pos_cash_df.groupby(['SK_ID_CURR', 'SK_ID_PREV']).agg(agg_functions)
    
    # Flatten column names
    pos_prev_agg.columns = [f'POS_{col[0]}_{col[1]}'.upper() for col in pos_prev_agg.columns]
    pos_prev_agg = pos_prev_agg.reset_index()
    
    # Then aggregate to client level
    pos_agg = pos_prev_agg.groupby(client_id_col).agg({col: ['mean', 'max', 'sum'] 
                                                      for col in pos_prev_agg.columns 
                                                      if col not in [client_id_col, 'SK_ID_PREV']})
    
    # Flatten column names
    pos_agg.columns = [f'{col[0]}_{col[1]}'.upper() for col in pos_agg.columns]
    pos_agg = pos_agg.reset_index()
    
    # Calculate completed loans ratio
    if 'NAME_CONTRACT_STATUS' in pos_cash_df.columns:
        # Count completed contracts per client
        completed = pos_cash_df[pos_cash_df['NAME_CONTRACT_STATUS'] == 'Completed'].groupby(['SK_ID_CURR', 'SK_ID_PREV']).size().reset_index()
        completed_count = completed.groupby(client_id_col).size().reset_index()
        completed_count.columns = [client_id_col, 'POS_COMPLETED_COUNT']
        
        # Count total unique contracts per client
        total_count = pos_cash_df.groupby(['SK_ID_CURR', 'SK_ID_PREV']).size().reset_index().groupby(client_id_col).size().reset_index()
        total_count.columns = [client_id_col, 'POS_TOTAL_COUNT']
        
        # Calculate completion ratio
        completion_ratio = completed_count.merge(total_count, on=client_id_col, how='right')
        completion_ratio['POS_COMPLETION_RATIO'] = completion_ratio['POS_COMPLETED_COUNT'] / completion_ratio['POS_TOTAL_COUNT']
        completion_ratio = completion_ratio[[client_id_col, 'POS_COMPLETION_RATIO']]
        
        # Merge with main aggregations
        pos_agg = pos_agg.merge(completion_ratio, on=client_id_col, how='left')
    
    return pos_agg


def create_credit_card_aggregations(cc_df, client_id_col='SK_ID_CURR'):
    """
    Create aggregated features from credit card balance data for each client.
    
    Parameters:
    -----------
    cc_df : pandas.DataFrame
        DataFrame containing credit card balance data
    client_id_col : str, default='SK_ID_CURR'
        Column name for client ID
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with aggregated credit card features for each client
    """
    if client_id_col not in cc_df.columns:
        print(f"Error: Client ID column '{client_id_col}' not found in credit card DataFrame")
        return None
    
    # Numerical columns to aggregate
    num_cols = cc_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    num_cols = [col for col in num_cols if col != client_id_col and col != 'SK_ID_PREV']
    
    # Define aggregations
    agg_functions = {
        'MONTHS_BALANCE': ['min', 'max', 'mean', 'size'],
        'AMT_BALANCE': ['min', 'max', 'mean', 'sum'],
        'AMT_CREDIT_LIMIT_ACTUAL': ['min', 'max', 'mean'],
        'AMT_DRAWINGS_ATM_CURRENT': ['min', 'max', 'mean', 'sum'],
        'AMT_DRAWINGS_CURRENT': ['min', 'max', 'mean', 'sum'],
        'AMT_DRAWINGS_OTHER_CURRENT': ['min', 'max', 'mean', 'sum'],
        'AMT_DRAWINGS_POS_CURRENT': ['min', 'max', 'mean', 'sum'],
        'AMT_INST_MIN_REGULARITY': ['min', 'max', 'mean'],
        'AMT_PAYMENT_CURRENT': ['min', 'max', 'mean', 'sum'],
        'AMT_PAYMENT_TOTAL_CURRENT': ['min', 'max', 'mean', 'sum'],
        'AMT_RECEIVABLE_PRINCIPAL': ['min', 'max', 'mean', 'sum'],
        'AMT_RECIVABLE': ['min', 'max', 'mean', 'sum'],
        'AMT_TOTAL_RECEIVABLE': ['min', 'max', 'mean', 'sum'],
        'CNT_DRAWINGS_ATM_CURRENT': ['min', 'max', 'mean', 'sum'],
        'CNT_DRAWINGS_CURRENT': ['min', 'max', 'mean', 'sum'],
        'CNT_DRAWINGS_OTHER_CURRENT': ['min', 'max', 'mean', 'sum'],
        'CNT_DRAWINGS_POS_CURRENT': ['min', 'max', 'mean', 'sum'],
        'CNT_INSTALMENT_MATURE_CUM': ['min', 'max', 'mean', 'sum'],
        'SK_DPD': ['max', 'mean', 'sum'],
        'SK_DPD_DEF': ['max', 'mean', 'sum']
    }
    
    # Filter to include only columns that exist in the DataFrame
    agg_functions = {k: v for k, v in agg_functions.items() if k in cc_df.columns}
    
    # Add default aggregations for other numerical columns
    for col in num_cols:
        if col not in agg_functions:
            agg_functions[col] = ['mean', 'min', 'max']
    
    # First aggregate by SK_ID_PREV to get card-level aggregations
    cc_prev_agg = cc_df.groupby(['SK_ID_CURR', 'SK_ID_PREV']).agg(agg_functions)
    
    # Flatten column names
    cc_prev_agg.columns = [f'CC_{col[0]}_{col[1]}'.upper() for col in cc_prev_agg.columns]
    cc_prev_agg = cc_prev_agg.reset_index()
    
    # Create credit utilization ratio if relevant columns exist
    if 'AMT_BALANCE' in cc_df.columns and 'AMT_CREDIT_LIMIT_ACTUAL' in cc_df.columns:
        # Calculate at the monthly level
        cc_df['CREDIT_UTILIZATION'] = cc_df['AMT_BALANCE'] / cc_df['AMT_CREDIT_LIMIT_ACTUAL'].replace(0, np.nan)
        cc_df['CREDIT_UTILIZATION'] = cc_df['CREDIT_UTILIZATION'].replace([np.inf, -np.inf], np.nan)
        
        # Aggregate utilization ratio to SK_ID_PREV level
        util_agg = cc_df.groupby(['SK_ID_CURR', 'SK_ID_PREV'])['CREDIT_UTILIZATION'].agg(['mean', 'max'])
        util_agg.columns = ['CC_CREDIT_UTILIZATION_MEAN', 'CC_CREDIT_UTILIZATION_MAX']
        util_agg = util_agg.reset_index()
        
        # Merge with card-level aggregations
        cc_prev_agg = cc_prev_agg.merge(util_agg, on=['SK_ID_CURR', 'SK_ID_PREV'], how='left')
    
    # Calculate payment ratio if relevant columns exist
    if 'AMT_PAYMENT_CURRENT' in cc_df.columns and 'AMT_TOTAL_RECEIVABLE' in cc_df.columns:
        # Calculate at the monthly level
        cc_df['PAYMENT_RATIO'] = cc_df['AMT_PAYMENT_CURRENT'] / cc_df['AMT_TOTAL_RECEIVABLE'].replace(0, np.nan)
        cc_df['PAYMENT_RATIO'] = cc_df['PAYMENT_RATIO'].replace([np.inf, -np.inf], np.nan)
        
        # Aggregate payment ratio to SK_ID_PREV level
        payment_agg = cc_df.groupby(['SK_ID_CURR', 'SK_ID_PREV'])['PAYMENT_RATIO'].agg(['mean', 'min'])
        payment_agg.columns = ['CC_PAYMENT_RATIO_MEAN', 'CC_PAYMENT_RATIO_MIN']
        payment_agg = payment_agg.reset_index()
        
        # Merge with card-level aggregations
        cc_prev_agg = cc_prev_agg.merge(payment_agg, on=['SK_ID_CURR', 'SK_ID_PREV'], how='left')
    
    # Then aggregate to client level
    cc_agg = cc_prev_agg.groupby(client_id_col).agg({col: ['mean', 'max', 'sum'] 
                                                    for col in cc_prev_agg.columns 
                                                    if col not in [client_id_col, 'SK_ID_PREV']})
    
    # Flatten column names
    cc_agg.columns = [f'{col[0]}_{col[1]}'.upper() for col in cc_agg.columns]
    cc_agg = cc_agg.reset_index()
    
    # Count number of credit cards per client
    card_count = cc_df[['SK_ID_CURR', 'SK_ID_PREV']].drop_duplicates().groupby(client_id_col).size().reset_index()
    card_count.columns = [client_id_col, 'CC_COUNT']
    cc_agg = cc_agg.merge(card_count, on=client_id_col, how='left')
    
    return cc_agg


def create_lag_features(df, columns, lag_periods, id_column=None, time_column=None):
    """
    Create lagged features for time series data.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing time series data
    columns : list
        Columns to create lags for
    lag_periods : list
        List of lag periods to create
    id_column : str, default=None
        Column name for entity ID (for panel data). If None, treats data as a single time series
    time_column : str, default=None
        Column name for time index. If None, uses DataFrame index
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with additional lag features
    """
    result_df = df.copy()
    
    # Validate columns
    missing_cols = [col for col in columns if col not in result_df.columns]
    if missing_cols:
        print(f"Warning: Columns not found in DataFrame: {missing_cols}")
        columns = [col for col in columns if col in result_df.columns]
    
    if not columns:
        print("Error: No valid columns to create lags for")
        return result_df
    
    # Set up time index if needed
    if time_column and time_column in result_df.columns:
        # Ensure time column is datetime
        if not pd.api.types.is_datetime64_dtype(result_df[time_column]):
            try:
                result_df[time_column] = pd.to_datetime(result_df[time_column])
            except:
                print(f"Warning: Could not convert {time_column} to datetime")
        
        # Sort by time
        result_df = result_df.sort_values(by=time_column)
    
    # For panel data (multiple time series identified by id_column)
    if id_column and id_column in result_df.columns:
        # Group by ID
        for col in columns:
            for lag in lag_periods:
                # Create lag feature
                result_df[f"{col}_LAG_{lag}"] = result_df.groupby(id_column)[col].shift(lag)
                
                # Create difference feature
                result_df[f"{col}_DIFF_{lag}"] = result_df[col] - result_df[f"{col}_LAG_{lag}"]
                
                # Create percentage change feature
                pct_col = f"{col}_PCT_CHANGE_{lag}"
                result_df[pct_col] = result_df.groupby(id_column)[col].pct_change(periods=lag)
                result_df[pct_col] = result_df[pct_col].replace([np.inf, -np.inf], np.nan)
    
    # For single time series
    else:
        for col in columns:
            for lag in lag_periods:
                # Create lag feature
                result_df[f"{col}_LAG_{lag}"] = result_df[col].shift(lag)
                
                # Create difference feature
                result_df[f"{col}_DIFF_{lag}"] = result_df[col] - result_df[f"{col}_LAG_{lag}"]
                
                # Create percentage change feature
                pct_col = f"{col}_PCT_CHANGE_{lag}"
                result_df[pct_col] = result_df[col].pct_change(periods=lag)
                result_df[pct_col] = result_df[pct_col].replace([np.inf, -np.inf], np.nan)
    
    return result_df