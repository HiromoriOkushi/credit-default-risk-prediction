import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, roc_auc_score, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report, f1_score, precision_score, recall_score,
    accuracy_score, log_loss, brier_score_loss
)
from scipy import stats
import warnings


def calculate_classification_metrics(y_true, y_pred, y_scores=None):
    """
    Calculate basic classification metrics for binary classification.
    
    Parameters:
    -----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted target values (binary)
    y_scores : array-like, default=None
        Predicted probabilities or scores
        
    Returns:
    --------
    dict
        Dictionary with classification metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'tn': confusion_matrix(y_true, y_pred)[0, 0],
        'fp': confusion_matrix(y_true, y_pred)[0, 1],
        'fn': confusion_matrix(y_true, y_pred)[1, 0],
        'tp': confusion_matrix(y_true, y_pred)[1, 1]
    }
    
    # Calculate specificity (true negative rate)
    metrics['specificity'] = metrics['tn'] / (metrics['tn'] + metrics['fp'])
    
    # Calculate probability-based metrics if scores are provided
    if y_scores is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_scores)
        metrics['avg_precision'] = average_precision_score(y_true, y_scores)
        metrics['log_loss'] = log_loss(y_true, y_scores)
        metrics['brier_score'] = brier_score_loss(y_true, y_scores)
    
    return metrics


def calculate_roc_auc(y_true, y_scores):
    """
    Calculate ROC AUC score and return ROC curve data.
    
    Parameters:
    -----------
    y_true : array-like
        True target values
    y_scores : array-like
        Predicted probabilities or scores
        
    Returns:
    --------
    dict
        Dictionary with ROC AUC score and curve data
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    auc_score = roc_auc_score(y_true, y_scores)
    
    # Create curve data for plotting
    curve_data = pd.DataFrame({
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thresholds
    })
    
    return {
        'auc': auc_score,
        'curve_data': curve_data
    }


def plot_roc_curve(y_true, y_scores, figsize=(10, 6)):
    """
    Plot ROC curve.
    
    Parameters:
    -----------
    y_true : array-like
        True target values
    y_scores : array-like
        Predicted probabilities or scores
    figsize : tuple, default=(10, 6)
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure
        ROC curve figure
    """
    roc_data = calculate_roc_auc(y_true, y_scores)
    curve_data = roc_data['curve_data']
    auc_score = roc_data['auc']
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot ROC curve
    ax.plot(curve_data['fpr'], curve_data['tpr'], lw=2, label=f'ROC curve (AUC = {auc_score:.3f})')
    
    # Plot diagonal reference line
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    
    # Set plot properties
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return fig


def calculate_precision_recall_curve(y_true, y_scores):
    """
    Generate precision-recall curve data.
    
    Parameters:
    -----------
    y_true : array-like
        True target values
    y_scores : array-like
        Predicted probabilities or scores
        
    Returns:
    --------
    dict
        Dictionary with precision-recall curve data and average precision
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    avg_precision = average_precision_score(y_true, y_scores)
    
    # Create curve data for plotting
    # Note: precision and recall have one more element than thresholds
    curve_data = pd.DataFrame({
        'precision': precision[:-1],
        'recall': recall[:-1],
        'thresholds': thresholds
    })
    
    # Add the last point
    last_point = pd.DataFrame({
        'precision': [precision[-1]],
        'recall': [recall[-1]],
        'thresholds': [1.0]  # Assume 1.0 as the highest threshold
    })
    
    curve_data = pd.concat([curve_data, last_point])
    
    return {
        'avg_precision': avg_precision,
        'curve_data': curve_data
    }


def plot_precision_recall_curve(y_true, y_scores, figsize=(10, 6)):
    """
    Plot precision-recall curve.
    
    Parameters:
    -----------
    y_true : array-like
        True target values
    y_scores : array-like
        Predicted probabilities or scores
    figsize : tuple, default=(10, 6)
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure
        Precision-recall curve figure
    """
    pr_data = calculate_precision_recall_curve(y_true, y_scores)
    curve_data = pr_data['curve_data']
    avg_precision = pr_data['avg_precision']
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot precision-recall curve
    ax.plot(curve_data['recall'], curve_data['precision'], lw=2, 
            label=f'Precision-Recall curve (AP = {avg_precision:.3f})')
    
    # Set plot properties
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)
    
    # Add baseline for no-skill classifier
    no_skill = sum(y_true) / len(y_true)
    ax.plot([0, 1], [no_skill, no_skill], 'k--', lw=1, label=f'No Skill Classifier (baseline = {no_skill:.3f})')
    ax.legend(loc='best')
    
    plt.tight_layout()
    
    return fig


def calculate_confusion_matrix(y_true, y_pred, normalize=None):
    """
    Calculate confusion matrix.
    
    Parameters:
    -----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted target values (binary)
    normalize : str, default=None
        Normalization option: 'true' (normalize by row), 'pred' (normalize by column),
        'all' (normalize by all), None (no normalization)
        
    Returns:
    --------
    numpy.ndarray
        Confusion matrix as a NumPy array
    """
    cm = confusion_matrix(y_true, y_pred, normalize=normalize)
    return cm


def plot_confusion_matrix(y_true, y_pred, normalize=None, figsize=(8, 6), cmap='Blues'):
    """
    Plot confusion matrix.
    
    Parameters:
    -----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted target values (binary)
    normalize : str, default=None
        Normalization option: 'true', 'pred', 'all', None
    figsize : tuple, default=(8, 6)
        Figure size
    cmap : str, default='Blues'
        Colormap for the plot
        
    Returns:
    --------
    matplotlib.figure.Figure
        Confusion matrix figure
    """
    cm = calculate_confusion_matrix(y_true, y_pred, normalize=normalize)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot confusion matrix
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', cmap=cmap, 
                square=True, cbar=True, ax=ax)
    
    # Set plot properties
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    
    # Set title based on normalization
    if normalize:
        ax.set_title(f'Normalized Confusion Matrix ({normalize})')
    else:
        ax.set_title('Confusion Matrix')
    
    ax.set_xticklabels(['Negative (0)', 'Positive (1)'])
    ax.set_yticklabels(['Negative (0)', 'Positive (1)'])
    
    plt.tight_layout()
    
    return fig


def calculate_lift_gain(y_true, y_scores, bins=10):
    """
    Calculate lift and gain charts data.
    
    Parameters:
    -----------
    y_true : array-like
        True target values
    y_scores : array-like
        Predicted probabilities or scores
    bins : int, default=10
        Number of bins for the charts
        
    Returns:
    --------
    dict
        Dictionary with lift and gain chart data
    """
    # Create a DataFrame with true values and scores
    df = pd.DataFrame({
        'y_true': y_true,
        'y_scores': y_scores
    })
    
    # Sort by scores in descending order
    df = df.sort_values('y_scores', ascending=False)
    
    # Calculate the total number of positives
    total_positives = df['y_true'].sum()
    total_records = len(df)
    
    # Create bins (deciles or specified number of bins)
    df['bin'] = pd.qcut(df.index, bins, labels=False)
    
    # Calculate metrics for each bin
    bin_stats = []
    cumulative_positives = 0
    cumulative_records = 0
    
    for bin_idx in range(bins):
        bin_df = df[df['bin'] == bin_idx]
        bin_records = len(bin_df)
        bin_positives = bin_df['y_true'].sum()
        bin_rate = bin_positives / bin_records if bin_records > 0 else 0
        
        cumulative_records += bin_records
        cumulative_positives += bin_positives
        cumulative_pct = cumulative_records / total_records
        cumulative_capture_rate = cumulative_positives / total_positives if total_positives > 0 else 0
        
        # Calculate lift
        bin_lift = bin_rate / (total_positives / total_records) if total_positives > 0 else 0
        cumulative_lift = (cumulative_positives / cumulative_records) / (total_positives / total_records) if total_positives > 0 and cumulative_records > 0 else 0
        
        bin_stats.append({
            'bin': bin_idx + 1,
            'count': bin_records,
            'positives': bin_positives,
            'rate': bin_rate,
            'lift': bin_lift,
            'cumulative_records': cumulative_records,
            'cumulative_positives': cumulative_positives,
            'cumulative_pct': cumulative_pct,
            'cumulative_capture_rate': cumulative_capture_rate,
            'cumulative_lift': cumulative_lift
        })
    
    return {
        'bin_stats': pd.DataFrame(bin_stats),
        'total_positives': total_positives,
        'total_records': total_records,
        'baseline_rate': total_positives / total_records
    }


def plot_lift_chart(y_true, y_scores, bins=10, figsize=(12, 6)):
    """
    Plot lift chart.
    
    Parameters:
    -----------
    y_true : array-like
        True target values
    y_scores : array-like
        Predicted probabilities or scores
    bins : int, default=10
        Number of bins for the chart
    figsize : tuple, default=(12, 6)
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure
        Lift chart figure
    """
    lift_data = calculate_lift_gain(y_true, y_scores, bins=bins)
    bin_stats = lift_data['bin_stats']
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot bin lift
    ax.bar(bin_stats['bin'], bin_stats['lift'], alpha=0.6, color='skyblue', 
           label='Bin Lift')
    
    # Plot cumulative lift
    ax.plot(bin_stats['bin'], bin_stats['cumulative_lift'], 'o-', color='red', 
            label='Cumulative Lift')
    
    # Add reference line for no lift
    ax.axhline(y=1.0, color='gray', linestyle='--', label='No Lift (Random)')
    
    # Set plot properties
    ax.set_xlabel('Bin (Sorted by Predicted Probability)')
    ax.set_ylabel('Lift')
    ax.set_title('Lift Chart')
    ax.set_xticks(bin_stats['bin'])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return fig


def plot_gain_chart(y_true, y_scores, bins=10, figsize=(12, 6)):
    """
    Plot cumulative gain chart.
    
    Parameters:
    -----------
    y_true : array-like
        True target values
    y_scores : array-like
        Predicted probabilities or scores
    bins : int, default=10
        Number of bins for the chart
    figsize : tuple, default=(12, 6)
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure
        Gain chart figure
    """
    gain_data = calculate_lift_gain(y_true, y_scores, bins=bins)
    bin_stats = gain_data['bin_stats']
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot cumulative gain curve
    ax.plot(bin_stats['cumulative_pct'], bin_stats['cumulative_capture_rate'], 'o-', 
            label='Model', linewidth=2)
    
    # Add reference line for random model
    ax.plot([0, 1], [0, 1], 'k--', label='Random Model')
    
    # Add perfect model curve (requires sorting by actual values)
    df_perfect = pd.DataFrame({
        'y_true': y_true
    }).sort_values('y_true', ascending=False)
    
    perfect_x = np.arange(len(df_perfect) + 1) / len(df_perfect)
    perfect_y = np.append(0, np.cumsum(df_perfect['y_true']) / sum(df_perfect['y_true']))
    
    ax.plot(perfect_x, perfect_y, 'g--', label='Perfect Model')
    
    # Set plot properties
    ax.set_xlabel('Percentage of Samples')
    ax.set_ylabel('Percentage of Positives Captured')
    ax.set_title('Cumulative Gain Chart')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    # Set axis limits
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    
    return fig


def calculate_ks_statistic(y_true, y_scores):
    """
    Calculate Kolmogorov-Smirnov statistic and related metrics.
    
    Parameters:
    -----------
    y_true : array-like
        True target values
    y_scores : array-like
        Predicted probabilities or scores
        
    Returns:
    --------
    dict
        Dictionary with KS statistic and related data
    """
    # Create a DataFrame with true values and scores
    df = pd.DataFrame({
        'y_true': y_true,
        'y_scores': y_scores
    })
    
    # Sort by scores in descending order
    df = df.sort_values('y_scores', ascending=False)
    
    # Calculate cumulative distributions
    total_positive = df['y_true'].sum()
    total_negative = len(df) - total_positive
    
    df['cumulative_positive'] = df['y_true'].cumsum() / total_positive
    df['cumulative_negative'] = (1 - df['y_true']).cumsum() / total_negative
    
    # Calculate KS for each threshold
    df['ks'] = df['cumulative_positive'] - df['cumulative_negative']
    
    # Find maximum KS and corresponding threshold
    max_ks_idx = df['ks'].abs().idxmax()
    max_ks = df.loc[max_ks_idx, 'ks']
    max_ks_threshold = df.loc[max_ks_idx, 'y_scores']
    
    # Calculate KS statistic using scipy
    ks_stat, p_value = stats.ks_2samp(
        df.loc[df['y_true'] == 1, 'y_scores'],
        df.loc[df['y_true'] == 0, 'y_scores']
    )
    
    return {
        'ks_statistic': max_ks,
        'ks_threshold': max_ks_threshold,
        'scipy_ks': ks_stat,
        'p_value': p_value,
        'ks_data': df[['y_scores', 'cumulative_positive', 'cumulative_negative', 'ks']]
    }


def plot_ks_curve(y_true, y_scores, figsize=(12, 6)):
    """
    Plot Kolmogorov-Smirnov curve.
    
    Parameters:
    -----------
    y_true : array-like
        True target values
    y_scores : array-like
        Predicted probabilities or scores
    figsize : tuple, default=(12, 6)
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure
        KS curve figure
    """
    ks_data = calculate_ks_statistic(y_true, y_scores)
    df = ks_data['ks_data']
    max_ks = ks_data['ks_statistic']
    ks_threshold = ks_data['ks_threshold']
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot cumulative distributions
    ax.plot(df['y_scores'], df['cumulative_positive'], label='Cumulative Positive', color='green')
    ax.plot(df['y_scores'], df['cumulative_negative'], label='Cumulative Negative', color='red')
    
    # Highlight KS
    ax.vlines(x=ks_threshold, ymin=min(df.loc[df['y_scores'] == ks_threshold, 'cumulative_negative']), 
              ymax=max(df.loc[df['y_scores'] == ks_threshold, 'cumulative_positive']), 
              colors='blue', linestyles='--', label=f'KS = {max_ks:.3f}')
    
    # Set plot properties
    ax.set_xlabel('Threshold (Predicted Probability)')
    ax.set_ylabel('Cumulative Distribution')
    ax.set_title('Kolmogorov-Smirnov Curve')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    # Reverse x-axis (higher probabilities to the left)
    ax.set_xlim(ax.get_xlim()[::-1])
    
    plt.tight_layout()
    
    return fig


def calculate_threshold_metrics(y_true, y_scores, thresholds=None):
    """
    Calculate classification metrics at different thresholds.
    
    Parameters:
    -----------
    y_true : array-like
        True target values
    y_scores : array-like
        Predicted probabilities or scores
    thresholds : array-like, default=None
        List of thresholds to evaluate. If None, uses np.arange(0.1, 1.0, 0.1)
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with metrics at each threshold
    """
    if thresholds is None:
        thresholds = np.arange(0.1, 1.0, 0.1)
    
    threshold_metrics = []
    
    for threshold in thresholds:
        y_pred = (y_scores >= threshold).astype(int)
        
        # Calculate confusion matrix values
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Calculate metrics
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        threshold_metrics.append({
            'threshold': threshold,
            'tp': tp,
            'fp': fp,
            'tn': tn,
            'fn': fn,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'f1': f1
        })
    
    return pd.DataFrame(threshold_metrics)


def plot_threshold_impact(y_true, y_scores, metric='f1', figsize=(12, 6)):
    """
    Plot the impact of threshold on various metrics.
    
    Parameters:
    -----------
    y_true : array-like
        True target values
    y_scores : array-like
        Predicted probabilities or scores
    metric : str or list, default='f1'
        Metric(s) to highlight. If str, highlights that metric.
        If list, plots all metrics in the list.
    figsize : tuple, default=(12, 6)
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure
        Threshold impact figure
    """
    # Use more thresholds for a smoother curve
    thresholds = np.arange(0.01, 1.0, 0.01)
    metrics_df = calculate_threshold_metrics(y_true, y_scores, thresholds)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Determine which metrics to plot
    if isinstance(metric, str):
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'specificity', 'f1']
        highlight_metric = metric
    else:
        metrics_to_plot = metric
        highlight_metric = None
    
    # Plot each metric
    for m in metrics_to_plot:
        if m == highlight_metric:
            ax.plot(metrics_df['threshold'], metrics_df[m], 'o-', linewidth=2, label=m.capitalize())
        else:
            ax.plot(metrics_df['threshold'], metrics_df[m], '--', alpha=0.7, label=m.capitalize())
    
    # If a specific metric is highlighted, find its optimal threshold
    if highlight_metric:
        best_idx = metrics_df[highlight_metric].idxmax()
        best_threshold = metrics_df.loc[best_idx, 'threshold']
        best_value = metrics_df.loc[best_idx, highlight_metric]
        
        ax.scatter([best_threshold], [best_value], s=100, c='red', zorder=5,
                   label=f'Best {highlight_metric.capitalize()} ({best_value:.3f} at {best_threshold:.2f})')
        
        ax.axvline(x=best_threshold, color='gray', linestyle='--', alpha=0.5)
    
    # Set plot properties
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Metric Value')
    ax.set_title('Impact of Classification Threshold on Metrics')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Set axis limits
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    
    return fig


def calculate_expected_calibration_error(y_true, y_scores, n_bins=10):
    """
    Calculate Expected Calibration Error (ECE) to assess probability calibration.
    
    Parameters:
    -----------
    y_true : array-like
        True target values
    y_scores : array-like
        Predicted probabilities or scores
    n_bins : int, default=10
        Number of bins for ECE calculation
        
    Returns:
    --------
    dict
        Dictionary with ECE and related calibration metrics
    """
    # Create bins and assign predictions to bins
    bin_indices = np.linspace(0, 1, n_bins + 1)
    bin_assignments = np.digitize(y_scores, bin_indices[1:-1])
    
    # Initialize data structures
    bin_data = []
    ece = 0
    
    # Calculate ECE for each bin
    for bin_idx in range(n_bins):
        # Get predictions and true values in this bin
        bin_mask = bin_assignments == bin_idx
        bin_y_true = y_true[bin_mask]
        bin_y_scores = y_scores[bin_mask]
        
        if len(bin_y_true) > 0:
            # Calculate bin statistics
            bin_confidence = np.mean(bin_y_scores)
            bin_accuracy = np.mean(bin_y_true)
            bin_count = len(bin_y_true)
            
            # Calculate calibration error for this bin
            bin_error = np.abs(bin_confidence - bin_accuracy)
            
            # Update ECE
            ece += (bin_count / len(y_true)) * bin_error
            
            bin_data.append({
                'bin_idx': bin_idx,
                'min_prob': bin_indices[bin_idx],
                'max_prob': bin_indices[bin_idx + 1],
                'avg_confidence': bin_confidence,
                'accuracy': bin_accuracy,
                'count': bin_count,
                'error': bin_error
            })
    
    return {
        'ece': ece,
        'bin_data': pd.DataFrame(bin_data)
    }


def plot_calibration_curve(y_true, y_scores, n_bins=10, figsize=(10, 8)):
    """
    Plot calibration curve (reliability diagram).
    
    Parameters:
    -----------
    y_true : array-like
        True target values
    y_scores : array-like
        Predicted probabilities or scores
    n_bins : int, default=10
        Number of bins for calibration curve
    figsize : tuple, default=(10, 8)
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure
        Calibration curve figure
    """
    calibration_data = calculate_expected_calibration_error(y_true, y_scores, n_bins=n_bins)
    bin_data = calibration_data['bin_data']
    ece = calibration_data['ece']
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
    
    # Plot calibration curve
    bin_centers = (bin_data['min_prob'] + bin_data['max_prob']) / 2
    ax.plot(bin_centers, bin_data['accuracy'], 'o-', label=f'Model (ECE: {ece:.3f})')
    
    # Plot histogram of predictions
    ax2 = ax.twinx()
    hist_data = np.histogram(y_scores, bins=n_bins, range=(0, 1))[0]
    hist_data = hist_data / np.sum(hist_data)
    ax2.bar(bin_centers, hist_data, width=1/n_bins, alpha=0.3, color='gray', label='Fraction of Predictions')
    ax2.set_ylabel('Fraction of Predictions')
    
    # Set plot properties
    ax.set_xlabel('Predicted Probability')
    ax.set_ylabel('Observed Frequency')
    ax.set_title('Calibration Curve (Reliability Diagram)')
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Set axis limits
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    
    return fig


def calculate_gini_coefficient(y_true, y_scores):
    """
    Calculate Gini coefficient.
    Gini coefficient is a common metric in credit scoring, related to ROC AUC.
    Gini = 2 * AUC - 1
    
    Parameters:
    -----------
    y_true : array-like
        True target values
    y_scores : array-like
        Predicted probabilities or scores
        
    Returns:
    --------
    float
        Gini coefficient
    """
    auc = roc_auc_score(y_true, y_scores)
    gini = 2 * auc - 1
    return gini


def calculate_psi(expected, actual, bins=10):
    """
    Calculate Population Stability Index (PSI) to detect population drift.
    PSI measures how much a distribution has shifted over time.
    
    Parameters:
    -----------
    expected : array-like
        Expected/reference distribution (e.g., training scores)
    actual : array-like
        Actual/current distribution (e.g., test scores)
    bins : int, default=10
        Number of bins for PSI calculation
        
    Returns:
    --------
    dict
        Dictionary with PSI value and bin details
    """
    # Create bins based on expected distribution
    bin_edges = np.percentile(expected, np.linspace(0, 100, bins + 1))
    
    # Ensure unique bin edges
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < 3:
        warnings.warn("Not enough unique bin edges. Using equal width bins instead.")
        bin_edges = np.linspace(min(expected), max(expected), bins + 1)
    
    # Count observations in each bin
    expected_counts, _ = np.histogram(expected, bins=bin_edges)
    actual_counts, _ = np.histogram(actual, bins=bin_edges)
    
    # Convert to percentages
    expected_pct = expected_counts / len(expected)
    actual_pct = actual_counts / len(actual)
    
    # Handle zero percentages (avoid division by zero)
    expected_pct = np.where(expected_pct == 0, 0.0001, expected_pct)
    actual_pct = np.where(actual_pct == 0, 0.0001, actual_pct)
    
    # Calculate PSI for each bin
    psi_values = (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
    
    # Create bin details
    bin_data = []
    for i in range(len(bin_edges) - 1):
        bin_data.append({
            'bin': i + 1,
            'min_value': bin_edges[i],
            'max_value': bin_edges[i + 1],
            'expected_count': expected_counts[i],
            'actual_count': actual_counts[i],
            'expected_pct': expected_pct[i],
            'actual_pct': actual_pct[i],
            'psi': psi_values[i]
        })
    
    total_psi = np.sum(psi_values)
    
    return {
        'psi': total_psi,
        'bin_data': pd.DataFrame(bin_data),
        'interpretation': interpret_psi(total_psi)
    }


def interpret_psi(psi):
    """
    Interpret PSI value.
    
    Parameters:
    -----------
    psi : float
        PSI value
        
    Returns:
    --------
    str
        Interpretation of PSI value
    """
    if psi < 0.1:
        return "No significant change"
    elif psi < 0.25:
        return "Moderate change - requires investigation"
    else:
        return "Significant change - immediate action required"


def plot_psi(expected, actual, bins=10, figsize=(12, 8)):
    """
    Plot Population Stability Index (PSI).
    
    Parameters:
    -----------
    expected : array-like
        Expected/reference distribution (e.g., training scores)
    actual : array-like
        Actual/current distribution (e.g., test scores)
    bins : int, default=10
        Number of bins for PSI calculation
    figsize : tuple, default=(12, 8)
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure
        PSI figure
    """
    psi_data = calculate_psi(expected, actual, bins=bins)
    bin_data = psi_data['bin_data']
    total_psi = psi_data['psi']
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot distributions
    bin_centers = (bin_data['min_value'] + bin_data['max_value']) / 2
    bar_width = (bin_data['max_value'] - bin_data['min_value']).mean() * 0.4
    
    ax1.bar(bin_centers - bar_width/2, bin_data['expected_pct'], width=bar_width, 
            alpha=0.7, label='Expected (Reference)')
    ax1.bar(bin_centers + bar_width/2, bin_data['actual_pct'], width=bar_width, 
            alpha=0.7, label='Actual (Current)')
    
    ax1.set_xlabel('Score Range')
    ax1.set_ylabel('Percentage')
    ax1.set_title(f'Population Stability Index: {total_psi:.4f} ({psi_data["interpretation"]})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot PSI by bin
    ax2.bar(bin_centers, bin_data['psi'], width=bar_width*1.5, alpha=0.7, color='red')
    ax2.set_xlabel('Score Range')
    ax2.set_ylabel('PSI')
    ax2.set_title('PSI Contribution by Bin')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return fig


def evaluate_model(model, X, y, threshold=0.5, return_plots=False, bins=10):
    """
    Evaluate model using multiple metrics.
    
    Parameters:
    -----------
    model : object
        Trained model with predict and predict_proba methods
    X : array-like
        Feature data
    y : array-like
        True target values
    threshold : float, default=0.5
        Classification threshold
    return_plots : bool, default=False
        Whether to return plots
    bins : int, default=10
        Number of bins for lift, gain, and calibration charts
        
    Returns:
    --------
    dict
        Dictionary with evaluation metrics and optionally plots
    """
    # Get predictions and probabilities
    if hasattr(model, 'predict_proba'):
        y_scores = model.predict_proba(X)[:, 1]
    else:
        y_scores = model.predict(X)
    
    y_pred = (y_scores >= threshold).astype(int)
    
    # Calculate basic metrics
    basic_metrics = calculate_classification_metrics(y, y_pred, y_scores)
    
    # Calculate ROC and PR curves
    roc_data = calculate_roc_auc(y, y_scores)
    pr_data = calculate_precision_recall_curve(y, y_scores)
    
    # Calculate lift and gain
    lift_gain_data = calculate_lift_gain(y, y_scores, bins=bins)
    
    # Calculate KS statistic
    ks_data = calculate_ks_statistic(y, y_scores)
    
    # Calculate threshold metrics
    threshold_metrics = calculate_threshold_metrics(y, y_scores)
    
    # Calculate calibration metrics
    calibration_data = calculate_expected_calibration_error(y, y_scores, n_bins=bins)
    
    # Calculate Gini coefficient
    gini = calculate_gini_coefficient(y, y_scores)
    
    # Combine all metrics
    evaluation = {
        'basic_metrics': basic_metrics,
        'roc_auc': roc_data['auc'],
        'gini': gini,
        'avg_precision': pr_data['avg_precision'],
        'ks_statistic': ks_data['ks_statistic'],
        'ks_threshold': ks_data['ks_threshold'],
        'ece': calibration_data['ece'],
        'classification_threshold': threshold
    }
    
    # Add curves data
    evaluation['curves'] = {
        'roc_curve': roc_data['curve_data'],
        'pr_curve': pr_data['curve_data'],
        'lift_gain': lift_gain_data['bin_stats'],
        'ks_curve': ks_data['ks_data'],
        'threshold_metrics': threshold_metrics,
        'calibration': calibration_data['bin_data']
    }
    
    # Generate plots if requested
    if return_plots:
        evaluation['plots'] = {
            'roc_curve': plot_roc_curve(y, y_scores),
            'pr_curve': plot_precision_recall_curve(y, y_scores),
            'confusion_matrix': plot_confusion_matrix(y, y_pred),
            'lift_chart': plot_lift_chart(y, y_scores, bins=bins),
            'gain_chart': plot_gain_chart(y, y_scores, bins=bins),
            'ks_curve': plot_ks_curve(y, y_scores),
            'threshold_impact': plot_threshold_impact(y, y_scores),
            'calibration_curve': plot_calibration_curve(y, y_scores, n_bins=bins)
        }
    
    return evaluation 