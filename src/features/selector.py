import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import (
    SelectKBest, f_classif, mutual_info_classif, RFE, 
    SelectFromModel, VarianceThreshold
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


def remove_high_correlation(df, threshold=0.9, method='pearson'):
    """
    Remove highly correlated features based on correlation threshold.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing features
    threshold : float, default=0.9
        Correlation threshold above which features are considered highly correlated
    method : str, default='pearson'
        Correlation method: 'pearson', 'kendall', 'spearman'
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with highly correlated features removed
    list
        List of columns that were removed
    """
    # Calculate the correlation matrix
    correlation_matrix = df.corr(method=method)
    
    # Create a mask for the upper triangle
    upper = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
    
    # Find features with correlation greater than threshold
    to_drop = [column for column in upper.columns if any(upper[column].abs() > threshold)]
    
    # Drop highly correlated features
    df_reduced = df.drop(columns=to_drop)
    
    print(f"Removed {len(to_drop)} highly correlated features.")
    
    return df_reduced, to_drop


def plot_correlation_heatmap(df, columns=None, figsize=(12, 10), method='pearson'):
    """
    Plot correlation heatmap for selected features.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing features
    columns : list, default=None
        List of columns to include in the heatmap. If None, use all columns
    figsize : tuple, default=(12, 10)
        Figure size (width, height)
    method : str, default='pearson'
        Correlation method: 'pearson', 'kendall', 'spearman'
        
    Returns:
    --------
    matplotlib.figure.Figure
        Correlation heatmap figure
    """
    # Filter columns if specified
    if columns is not None:
        data = df[columns].copy()
    else:
        data = df.copy()
    
    # Calculate correlation matrix
    corr = data.corr(method=method)
    
    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, cmap="coolwarm", center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.5}, annot=False)
    
    plt.tight_layout()
    
    return fig


def select_variance_threshold(df, threshold=0.01):
    """
    Select features based on variance threshold.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing features
    threshold : float, default=0.01
        Features with a variance lower than this threshold will be removed
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with low variance features removed
    list
        List of columns that were removed
    """
    # Initialize the selector
    selector = VarianceThreshold(threshold=threshold)
    
    # Fit the selector
    selector.fit(df)
    
    # Get the selected feature indices
    selected_idx = selector.get_support(indices=True)
    
    # Get the selected feature names
    selected_columns = df.columns[selected_idx].tolist()
    
    # Get the removed feature names
    removed_columns = [col for col in df.columns if col not in selected_columns]
    
    # Create a new DataFrame with only the selected features
    df_reduced = df[selected_columns].copy()
    
    print(f"Removed {len(removed_columns)} low variance features.")
    
    return df_reduced, removed_columns


def rank_features_correlation(df, target, method='pearson'):
    """
    Rank features based on correlation with target variable.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing features
    target : pandas.Series
        Target variable
    method : str, default='pearson'
        Correlation method: 'pearson', 'kendall', 'spearman'
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with features ranked by correlation
    """
    # Initialize list to store correlation values
    correlations = []
    
    # Calculate correlation for each feature
    for column in df.columns:
        if df[column].dtype in ['int64', 'float64']:
            if method == 'pearson':
                corr, p_value = pearsonr(df[column].fillna(0), target)
            else:
                # For non-pearson methods, use pandas corr
                corr = df[column].fillna(0).corr(target, method=method)
                p_value = np.nan
            
            correlations.append({
                'Feature': column,
                'Correlation': corr,
                'Abs_Correlation': abs(corr),
                'P_Value': p_value
            })
    
    # Convert to DataFrame and sort by absolute correlation
    corr_df = pd.DataFrame(correlations)
    corr_df = corr_df.sort_values('Abs_Correlation', ascending=False)
    
    return corr_df


def select_importance_features(df, target, n_features=None, method='mutual_info', random_state=42):
    """
    Select top features using statistical methods.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing features
    target : pandas.Series
        Target variable
    n_features : int, default=None
        Number of features to select. If None, half of features are selected
    method : str, default='mutual_info'
        Feature selection method: 'mutual_info', 'f_classif', 'tree_based'
    random_state : int, default=42
        Random seed for reproducibility
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with selected features
    list
        List of features in order of importance
    """
    # Set default number of features if not specified
    if n_features is None:
        n_features = df.shape[1] // 2
    
    # Initialize feature selector based on method
    if method == 'mutual_info':
        selector = SelectKBest(mutual_info_classif, k=n_features)
    elif method == 'f_classif':
        selector = SelectKBest(f_classif, k=n_features)
    elif method == 'tree_based':
        # Use Random Forest to select features
        model = RandomForestClassifier(n_estimators=100, random_state=random_state)
        selector = SelectFromModel(model, max_features=n_features)
    else:
        raise ValueError(f"Unsupported method: {method}")
    
    # Fit the selector
    selector.fit(df, target)
    
    # Get the selected feature indices
    if method in ['mutual_info', 'f_classif']:
        selected_idx = selector.get_support(indices=True)
        
        # Get feature scores
        scores = selector.scores_
        
        # Create a mapping of feature indices to scores
        feature_scores = {idx: scores[idx] for idx in range(len(scores))}
        
        # Sort the selected indices by score in descending order
        sorted_idx = sorted(selected_idx, key=lambda idx: feature_scores[idx], reverse=True)
        
    else:  # tree_based
        # Get feature importances
        importances = selector.estimator_.feature_importances_
        
        # Create a mapping of feature indices to importance
        feature_importances = {idx: importances[idx] for idx in range(len(importances))}
        
        # Get the selected feature indices
        selected_idx = selector.get_support(indices=True)
        
        # Sort the selected indices by importance in descending order
        sorted_idx = sorted(selected_idx, key=lambda idx: feature_importances[idx], reverse=True)
    
    # Get the selected feature names in order of importance
    selected_columns = df.columns[sorted_idx].tolist()
    
    # Create a new DataFrame with only the selected features
    df_reduced = df[selected_columns].copy()
    
    print(f"Selected {len(selected_columns)} features using {method} method.")
    
    return df_reduced, selected_columns


def select_features_rfe(X, y, estimator=None, n_features=None, step=1, cv=None, verbose=0):
    """
    Recursive feature elimination for feature selection.
    
    Parameters:
    -----------
    X : pandas.DataFrame
        Feature DataFrame
    y : pandas.Series
        Target variable
    estimator : object, default=None
        A supervised learning estimator with a fit method.
        If None, uses LogisticRegression
    n_features : int, default=None
        Number of features to select. If None, half of features are selected
    step : int or float, default=1
        Number or percentage of features to remove at each iteration
    cv : int, default=None
        Cross-validation folds. If None, no cross-validation is used
    verbose : int, default=0
        Controls verbosity of output
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with selected features
    list
        List of features in order of importance
    """
    # Set default estimator if not specified
    if estimator is None:
        estimator = LogisticRegression(random_state=42, max_iter=1000)
    
    # Set default number of features if not specified
    if n_features is None:
        n_features = X.shape[1] // 2
    
    # Initialize RFE
    if cv is None:
        selector = RFE(estimator=estimator, n_features_to_select=n_features, step=step, verbose=verbose)
    else:
        from sklearn.feature_selection import RFECV
        selector = RFECV(estimator=estimator, min_features_to_select=n_features, step=step, cv=cv, verbose=verbose)
    
    # Fit the selector
    selector.fit(X, y)
    
    # Get the selected feature indices
    selected_idx = selector.get_support(indices=True)
    
    # Get the selected feature names
    selected_columns = X.columns[selected_idx].tolist()
    
    # Try to get feature ranking if available
    if hasattr(selector, 'ranking_'):
        # Create a DataFrame with feature names and rankings
        ranking_df = pd.DataFrame({
            'Feature': X.columns,
            'Ranking': selector.ranking_
        })
        
        # Sort by ranking (1 is best)
        ranking_df = ranking_df.sort_values('Ranking')
        
        # Use the ordered list of features
        ordered_features = ranking_df['Feature'].tolist()
        
        # Filter to only include selected features
        selected_columns = [feat for feat in ordered_features if feat in selected_columns]
    
    # Create a new DataFrame with only the selected features
    df_reduced = X[selected_columns].copy()
    
    print(f"Selected {len(selected_columns)} features using RFE.")
    
    return df_reduced, selected_columns


def get_tree_based_feature_importance(X, y, n_estimators=100, random_state=42, model_type='rf'):
    """
    Get feature importance using tree-based models.
    
    Parameters:
    -----------
    X : pandas.DataFrame
        Feature DataFrame
    y : pandas.Series
        Target variable
    n_estimators : int, default=100
        Number of trees in the forest/boosting
    random_state : int, default=42
        Random seed for reproducibility
    model_type : str, default='rf'
        Type of tree-based model: 'rf' (Random Forest) or 'gb' (Gradient Boosting)
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with features and their importance scores
    """
    # Initialize the model
    if model_type == 'rf':
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    elif model_type == 'gb':
        model = GradientBoostingClassifier(n_estimators=n_estimators, random_state=random_state)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Train the model
    model.fit(X, y)
    
    # Get feature importances
    importances = model.feature_importances_
    
    # Create DataFrame with feature names and importances
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': importances
    })
    
    # Sort by importance
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    return importance_df


def plot_feature_importance(importance_df, top_n=20, figsize=(12, 8)):
    """
    Plot feature importance.
    
    Parameters:
    -----------
    importance_df : pandas.DataFrame
        DataFrame with features and their importance scores.
        Must have 'Feature' and 'Importance' columns
    top_n : int, default=20
        Number of top features to display
    figsize : tuple, default=(12, 8)
        Figure size (width, height)
        
    Returns:
    --------
    matplotlib.figure.Figure
        Feature importance figure
    """
    # Take top N features
    top_features = importance_df.head(top_n)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot horizontal bar chart
    sns.barplot(x='Importance', y='Feature', data=top_features, ax=ax)
    
    # Set title and labels
    ax.set_title(f'Top {top_n} Most Important Features')
    ax.set_xlabel('Importance')
    ax.set_ylabel('Feature')
    
    plt.tight_layout()
    
    return fig


def run_feature_selection(df, target, config):
    """
    Apply feature selection pipeline based on configuration.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing features
    target : pandas.Series
        Target variable
    config : dict
        Configuration dictionary with feature selection steps and parameters.
        Example:
        {
            'variance_threshold': {'threshold': 0.01},
            'correlation_removal': {'threshold': 0.9, 'method': 'pearson'},
            'feature_importance': {
                'method': 'tree_based',
                'n_features': 50,
                'random_state': 42
            }
        }
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with selected features
    dict
        Dictionary with information about the feature selection process
    """
    result_df = df.copy()
    selection_info = {
        'initial_shape': result_df.shape,
        'removed_features': {},
        'importance_ranking': None
    }
    
    # 1. Remove low variance features
    if 'variance_threshold' in config:
        variance_config = config['variance_threshold']
        result_df, removed_var = select_variance_threshold(
            result_df,
            threshold=variance_config.get('threshold', 0.01)
        )
        selection_info['removed_features']['low_variance'] = removed_var
    
    # 2. Remove highly correlated features
    if 'correlation_removal' in config:
        corr_config = config['correlation_removal']
        result_df, removed_corr = remove_high_correlation(
            result_df,
            threshold=corr_config.get('threshold', 0.9),
            method=corr_config.get('method', 'pearson')
        )
        selection_info['removed_features']['high_correlation'] = removed_corr
    
    # 3. Select features based on importance or statistical tests
    if 'feature_importance' in config:
        imp_config = config['feature_importance']
        method = imp_config.get('method', 'tree_based')
        n_features = imp_config.get('n_features', None)
        random_state = imp_config.get('random_state', 42)
        
        if method == 'rfe':
            # RFE requires additional parameters
            estimator_type = imp_config.get('estimator', 'logistic')
            cv = imp_config.get('cv', None)
            step = imp_config.get('step', 1)
            
            if estimator_type == 'logistic':
                estimator = LogisticRegression(random_state=random_state, max_iter=1000)
            elif estimator_type == 'rf':
                estimator = RandomForestClassifier(n_estimators=100, random_state=random_state)
            else:
                estimator = None
            
            result_df, selected_features = select_features_rfe(
                result_df, target, estimator=estimator, n_features=n_features,
                step=step, cv=cv, verbose=0
            )
            
        elif method == 'tree_importance':
            # Get feature importance directly without feature reduction
            model_type = imp_config.get('model_type', 'rf')
            n_estimators = imp_config.get('n_estimators', 100)
            
            importance_df = get_tree_based_feature_importance(
                result_df, target, n_estimators=n_estimators,
                random_state=random_state, model_type=model_type
            )
            
            selection_info['importance_ranking'] = importance_df
            
            # Optionally filter to top N features
            if n_features is not None:
                top_features = importance_df.head(n_features)['Feature'].tolist()
                result_df = result_df[top_features].copy()
                selection_info['selected_features'] = top_features
        else:
            # Use statistical methods
            result_df, selected_features = select_importance_features(
                result_df, target, n_features=n_features,
                method=method, random_state=random_state
            )
            selection_info['selected_features'] = selected_features
    
    # Record final results
    selection_info['final_shape'] = result_df.shape
    selection_info['reduction_percentage'] = round(
        (1 - (result_df.shape[1] / selection_info['initial_shape'][1])) * 100, 2
    )
    
    return result_df, selection_info