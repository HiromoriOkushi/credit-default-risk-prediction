import numpy as np
import pandas as pd
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler, NearMiss, TomekLinks
from imblearn.combine import SMOTETomek, SMOTEENN
from collections import Counter


def undersample_majority(X, y, sampling_strategy=0.5, random_state=42, method='random'):
    """
    Undersample majority class to reduce class imbalance.
    
    Parameters:
    -----------
    X : pandas.DataFrame or numpy.ndarray
        Feature data
    y : pandas.Series or numpy.ndarray
        Target variable
    sampling_strategy : float or dict, default=0.5
        If float, it represents the ratio of the minority class to the majority class after resampling.
        If dict, it contains the target number of samples for each class.
    random_state : int, default=42
        Random seed for reproducibility
    method : str, default='random'
        Undersampling method: 'random', 'nearmiss', 'tomek'
        
    Returns:
    --------
    tuple
        (X_resampled, y_resampled)
    """
    # Convert to numpy arrays if pandas objects
    X_array = X.values if isinstance(X, pd.DataFrame) else X
    y_array = y.values if isinstance(y, pd.Series) else y
    
    # Store original feature names and index if DataFrame
    feature_names = X.columns if isinstance(X, pd.DataFrame) else None
    index = X.index if isinstance(X, pd.DataFrame) else None
    
    # Initialize the undersampler based on the method
    if method == 'random':
        undersampler = RandomUnderSampler(
            sampling_strategy=sampling_strategy,
            random_state=random_state
        )
    elif method == 'nearmiss':
        undersampler = NearMiss(
            sampling_strategy=sampling_strategy,
            version=1  # NearMiss-1 algorithm
        )
    elif method == 'tomek':
        undersampler = TomekLinks(
            sampling_strategy=sampling_strategy
        )
    else:
        raise ValueError(f"Unsupported undersampling method: {method}. Choose from 'random', 'nearmiss', 'tomek'.")
    
    # Apply undersampling
    X_resampled, y_resampled = undersampler.fit_resample(X_array, y_array)
    
    # Print resampling results
    print(f"Original class distribution: {Counter(y_array)}")
    print(f"Resampled class distribution: {Counter(y_resampled)}")
    
    # Convert back to pandas if input was pandas
    if isinstance(X, pd.DataFrame):
        X_resampled = pd.DataFrame(X_resampled, columns=feature_names)
    if isinstance(y, pd.Series):
        y_resampled = pd.Series(y_resampled, name=y.name)
    
    return X_resampled, y_resampled


def oversample_minority(X, y, method='smote', sampling_strategy=0.5, random_state=42, **kwargs):
    """
    Oversample minority class to reduce class imbalance.
    
    Parameters:
    -----------
    X : pandas.DataFrame or numpy.ndarray
        Feature data
    y : pandas.Series or numpy.ndarray
        Target variable
    method : str, default='smote'
        Oversampling method: 'smote', 'adasyn', 'borderline_smote', 'random'
    sampling_strategy : float or dict, default=0.5
        If float, it represents the ratio of the minority class to the majority class after resampling.
        If dict, it contains the target number of samples for each class.
    random_state : int, default=42
        Random seed for reproducibility
    **kwargs : dict
        Additional keyword arguments for the oversampling method
        
    Returns:
    --------
    tuple
        (X_resampled, y_resampled)
    """
    # Convert to numpy arrays if pandas objects
    X_array = X.values if isinstance(X, pd.DataFrame) else X
    y_array = y.values if isinstance(y, pd.Series) else y
    
    # Store original feature names and index if DataFrame
    feature_names = X.columns if isinstance(X, pd.DataFrame) else None
    index = X.index if isinstance(X, pd.DataFrame) else None
    
    # Initialize the oversampler based on the method
    if method == 'smote':
        oversampler = SMOTE(
            sampling_strategy=sampling_strategy,
            random_state=random_state,
            **kwargs
        )
    elif method == 'adasyn':
        oversampler = ADASYN(
            sampling_strategy=sampling_strategy,
            random_state=random_state,
            **kwargs
        )
    elif method == 'borderline_smote':
        oversampler = BorderlineSMOTE(
            sampling_strategy=sampling_strategy,
            random_state=random_state,
            **kwargs
        )
    elif method == 'random':
        # Random oversampling with replacement
        # Find minority and majority classes
        classes, counts = np.unique(y_array, return_counts=True)
        minority_class = classes[np.argmin(counts)]
        majority_class = classes[np.argmax(counts)]
        
        # Separate minority and majority samples
        X_minority = X_array[y_array == minority_class]
        y_minority = y_array[y_array == minority_class]
        X_majority = X_array[y_array == majority_class]
        y_majority = y_array[y_array == majority_class]
        
        # Calculate number of samples to generate
        if isinstance(sampling_strategy, float):
            n_samples = int(sampling_strategy * len(X_majority))
        elif isinstance(sampling_strategy, dict):
            n_samples = sampling_strategy.get(minority_class, len(X_minority))
        else:
            n_samples = len(X_minority)
        
        # Resample minority class
        X_minority_resampled, y_minority_resampled = resample(
            X_minority, y_minority,
            replace=True,
            n_samples=n_samples,
            random_state=random_state
        )
        
        # Combine with majority class
        X_resampled = np.vstack([X_majority, X_minority_resampled])
        y_resampled = np.hstack([y_majority, y_minority_resampled])
        
        # Print resampling results
        print(f"Original class distribution: {Counter(y_array)}")
        print(f"Resampled class distribution: {Counter(y_resampled)}")
        
        # Convert back to pandas if input was pandas
        if isinstance(X, pd.DataFrame):
            X_resampled = pd.DataFrame(X_resampled, columns=feature_names)
        if isinstance(y, pd.Series):
            y_resampled = pd.Series(y_resampled, name=y.name)
        
        return X_resampled, y_resampled
    
    else:
        raise ValueError(
            f"Unsupported oversampling method: {method}. "
            f"Choose from 'smote', 'adasyn', 'borderline_smote', 'random'."
        )
    
    # Apply oversampling
    X_resampled, y_resampled = oversampler.fit_resample(X_array, y_array)
    
    # Print resampling results
    print(f"Original class distribution: {Counter(y_array)}")
    print(f"Resampled class distribution: {Counter(y_resampled)}")
    
    # Convert back to pandas if input was pandas
    if isinstance(X, pd.DataFrame):
        X_resampled = pd.DataFrame(X_resampled, columns=feature_names)
    if isinstance(y, pd.Series):
        y_resampled = pd.Series(y_resampled, name=y.name)
    
    return X_resampled, y_resampled


def create_balanced_weights(y):
    """
    Create class weights for imbalanced data.
    
    Parameters:
    -----------
    y : pandas.Series or numpy.ndarray
        Target variable
        
    Returns:
    --------
    dict
        Dictionary with class weights
    """
    # Count samples in each class
    if isinstance(y, pd.Series):
        class_counts = y.value_counts().to_dict()
    else:
        classes, counts = np.unique(y, return_counts=True)
        class_counts = dict(zip(classes, counts))
    
    # Total number of samples
    n_samples = sum(class_counts.values())
    
    # Number of classes
    n_classes = len(class_counts)
    
    # Create class weights
    class_weights = {
        cls: n_samples / (n_classes * count)
        for cls, count in class_counts.items()
    }
    
    return class_weights


def combine_sampling(X, y, under_strategy=0.5, over_strategy=0.5, method='smote_tomek', random_state=42, **kwargs):
    """
    Apply combination of oversampling and undersampling.
    
    Parameters:
    -----------
    X : pandas.DataFrame or numpy.ndarray
        Feature data
    y : pandas.Series or numpy.ndarray
        Target variable
    under_strategy : float or dict, default=0.5
        Undersampling strategy
    over_strategy : float or dict, default=0.5
        Oversampling strategy
    method : str, default='smote_tomek'
        Combined sampling method: 'smote_tomek', 'smote_enn', 'manual'
    random_state : int, default=42
        Random seed for reproducibility
    **kwargs : dict
        Additional keyword arguments for the sampling methods
        
    Returns:
    --------
    tuple
        (X_resampled, y_resampled)
    """
    # Convert to numpy arrays if pandas objects
    X_array = X.values if isinstance(X, pd.DataFrame) else X
    y_array = y.values if isinstance(y, pd.Series) else y
    
    # Store original feature names and index if DataFrame
    feature_names = X.columns if isinstance(X, pd.DataFrame) else None
    index = X.index if isinstance(X, pd.DataFrame) else None
    
    # Initialize the combined sampler based on the method
    if method == 'smote_tomek':
        combined_sampler = SMOTETomek(
            sampling_strategy=over_strategy,
            random_state=random_state,
            **kwargs
        )
    elif method == 'smote_enn':
        combined_sampler = SMOTEENN(
            sampling_strategy=over_strategy,
            random_state=random_state,
            **kwargs
        )
    elif method == 'manual':
        # First oversample, then undersample
        X_over, y_over = oversample_minority(
            X, y, 
            method='smote',
            sampling_strategy=over_strategy,
            random_state=random_state
        )
        
        X_combined, y_combined = undersample_majority(
            X_over, y_over,
            sampling_strategy=under_strategy,
            random_state=random_state
        )
        
        return X_combined, y_combined
    else:
        raise ValueError(
            f"Unsupported combined sampling method: {method}. "
            f"Choose from 'smote_tomek', 'smote_enn', 'manual'."
        )
    
    # Apply combined sampling
    X_resampled, y_resampled = combined_sampler.fit_resample(X_array, y_array)
    
    # Print resampling results
    print(f"Original class distribution: {Counter(y_array)}")
    print(f"Resampled class distribution: {Counter(y_resampled)}")
    
    # Convert back to pandas if input was pandas
    if isinstance(X, pd.DataFrame):
        X_resampled = pd.DataFrame(X_resampled, columns=feature_names)
    if isinstance(y, pd.Series):
        y_resampled = pd.Series(y_resampled, name=y.name)
    
    return X_resampled, y_resampled


def handle_imbalance(X, y, method, sampling_params=None, random_state=42):
    """
    Apply appropriate imbalance handling method.
    
    Parameters:
    -----------
    X : pandas.DataFrame or numpy.ndarray
        Feature data
    y : pandas.Series or numpy.ndarray
        Target variable
    method : str
        Method to handle imbalance: 'undersample', 'oversample', 'combined', 'weights'
    sampling_params : dict, default=None
        Parameters for the sampling method
    random_state : int, default=42
        Random seed for reproducibility
        
    Returns:
    --------
    tuple or dict
        For 'undersample', 'oversample', 'combined': (X_resampled, y_resampled)
        For 'weights': Dictionary with class weights
    """
    # Default parameters if none provided
    if sampling_params is None:
        sampling_params = {}
    
    # Apply the appropriate method
    if method == 'undersample':
        undersample_method = sampling_params.get('undersample_method', 'random')
        sampling_strategy = sampling_params.get('sampling_strategy', 0.5)
        
        return undersample_majority(
            X, y, 
            sampling_strategy=sampling_strategy,
            random_state=random_state,
            method=undersample_method
        )
    
    elif method == 'oversample':
        oversample_method = sampling_params.get('oversample_method', 'smote')
        sampling_strategy = sampling_params.get('sampling_strategy', 0.5)
        
        return oversample_minority(
            X, y, 
            method=oversample_method,
            sampling_strategy=sampling_strategy,
            random_state=random_state,
            **{k: v for k, v in sampling_params.items() if k not in ['oversample_method', 'sampling_strategy']}
        )
    
    elif method == 'combined':
        combined_method = sampling_params.get('combined_method', 'smote_tomek')
        under_strategy = sampling_params.get('under_strategy', 0.5)
        over_strategy = sampling_params.get('over_strategy', 0.5)
        
        return combine_sampling(
            X, y, 
            under_strategy=under_strategy,
            over_strategy=over_strategy,
            method=combined_method,
            random_state=random_state,
            **{k: v for k, v in sampling_params.items() 
               if k not in ['combined_method', 'under_strategy', 'over_strategy']}
        )
    
    elif method == 'weights':
        return create_balanced_weights(y)
    
    else:
        raise ValueError(
            f"Unsupported imbalance handling method: {method}. "
            f"Choose from 'undersample', 'oversample', 'combined', 'weights'."
        )


def get_sample_weight_array(y, class_weight='balanced'):
    """
    Create an array of sample weights based on class weights.
    
    Parameters:
    -----------
    y : pandas.Series or numpy.ndarray
        Target variable
    class_weight : str or dict, default='balanced'
        Class weights. If 'balanced', automatically calculate balanced weights.
        If dict, contains {class_label: weight} mapping.
        
    Returns:
    --------
    numpy.ndarray
        Array of sample weights corresponding to each instance in y
    """
    # Calculate class weights if set to 'balanced'
    if class_weight == 'balanced':
        class_weight = create_balanced_weights(y)
    
    # Convert y to numpy array for indexing
    y_array = np.array(y)
    
    # Create sample weights array
    sample_weight = np.ones(len(y_array))
    
    # Assign weights based on class
    for cls, weight in class_weight.items():
        sample_weight[y_array == cls] = weight
    
    return sample_weight


def evaluate_sampling_impact(X, y, model, sampling_methods, cv=5, scoring='roc_auc', random_state=42):
    """
    Evaluate the impact of different sampling methods on model performance.
    
    Parameters:
    -----------
    X : pandas.DataFrame
        Feature data
    y : pandas.Series
        Target variable
    model : object
        Model to evaluate (must implement fit and predict_proba)
    sampling_methods : list of dict
        List of dictionaries with sampling configurations to evaluate
        Each dict must contain 'name' and 'method' keys, plus optional parameters
    cv : int, default=5
        Number of cross-validation folds
    scoring : str, default='roc_auc'
        Scoring metric for evaluation
    random_state : int, default=42
        Random seed for reproducibility
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with evaluation results for each sampling method
    """
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    
    # Initialize results
    results = []
    
    # Baseline without sampling
    baseline_scores = cross_val_score(
        model, X, y, 
        cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state),
        scoring=scoring
    )
    
    results.append({
        'name': 'No Sampling (Baseline)',
        'method': 'none',
        'mean_score': np.mean(baseline_scores),
        'std_score': np.std(baseline_scores),
        'detailed_scores': baseline_scores
    })
    
    # Evaluate each sampling method
    for config in sampling_methods:
        method_name = config['name']
        method_type = config['method']
        
        # Create a copy of the configuration without 'name' and 'method' keys
        params = {k: v for k, v in config.items() if k not in ['name', 'method']}
        
        print(f"Evaluating {method_name}...")
        
        # Apply sampling method for each fold separately
        fold_scores = []
        cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
        
        for train_idx, test_idx in cv_splitter.split(X, y):
            X_fold_train, X_fold_test = X.iloc[train_idx], X.iloc[test_idx]
            y_fold_train, y_fold_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Apply sampling method to training data only
            if method_type != 'weights':
                X_resampled, y_resampled = handle_imbalance(
                    X_fold_train, y_fold_train, 
                    method=method_type, 
                    sampling_params=params,
                    random_state=random_state
                )
                
                # Train model on resampled data
                model.fit(X_resampled, y_resampled)
                
            else:
                # For class weights approach
                class_weights = handle_imbalance(
                    X_fold_train, y_fold_train, 
                    method='weights',
                    random_state=random_state
                )
                
                # Create sample weights
                sample_weights = get_sample_weight_array(y_fold_train, class_weights)
                
                # Train with sample weights
                model.fit(X_fold_train, y_fold_train, sample_weight=sample_weights)
            
            # Evaluate on test fold
            from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score
            
            y_pred_proba = model.predict_proba(X_fold_test)[:, 1]
            y_pred = (y_pred_proba >= 0.5).astype(int)
            
            if scoring == 'roc_auc':
                score = roc_auc_score(y_fold_test, y_pred_proba)
            elif scoring == 'f1':
                score = f1_score(y_fold_test, y_pred)
            elif scoring == 'accuracy':
                score = accuracy_score(y_fold_test, y_pred)
            elif scoring == 'precision':
                score = precision_score(y_fold_test, y_pred)
            elif scoring == 'recall':
                score = recall_score(y_fold_test, y_pred)
            else:
                score = roc_auc_score(y_fold_test, y_pred_proba)
            
            fold_scores.append(score)
        
        # Calculate aggregate statistics
        results.append({
            'name': method_name,
            'method': method_type,
            'mean_score': np.mean(fold_scores),
            'std_score': np.std(fold_scores),
            'detailed_scores': fold_scores
        })
    
    # Convert results to DataFrame
    results_df = pd.DataFrame([
        {
            'Sampling Method': r['name'],
            'Method Type': r['method'],
            f'Mean {scoring.upper()}': r['mean_score'],
            f'Std {scoring.upper()}': r['std_score']
        }
        for r in results
    ])
    
    # Sort by performance
    results_df = results_df.sort_values(f'Mean {scoring.upper()}', ascending=False)
    
    return results_df