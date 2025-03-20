"""
Utility functions for visualization and plotting in adaptiveforecast.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, Optional, Tuple

def _convert_period_index(data):
    """Helper function to convert PeriodIndex to DatetimeIndex if needed."""
    if data is None:
        return None
    if isinstance(data.index, pd.PeriodIndex):
        data = data.copy()
        data.index = data.index.to_timestamp()
    return data

def plot_individual_forecast(
    train_y,
    test_y,
    predictions,
    future_predictions=None,
    title: str = None,
    figsize: Tuple[int, int] = (15, 7)
) -> plt.Figure:
    """
    Plot individual model forecasts with training, test, and future predictions.
    
    Parameters
    ----------
    train_y : pd.Series
        Training data
    test_y : pd.Series
        Test data
    predictions : pd.Series
        Model predictions on test set
    future_predictions : pd.Series, optional
        Future predictions beyond test set
    title : str, optional
        Plot title
    figsize : tuple, default=(15, 7)
        Figure size
        
    Returns
    -------
    matplotlib.figure.Figure
        The figure object
    """
    # Convert PeriodIndex to DatetimeIndex if needed
    train_y = _convert_period_index(train_y)
    test_y = _convert_period_index(test_y)
    predictions = _convert_period_index(predictions)
    future_predictions = _convert_period_index(future_predictions)
    
    plt.figure(figsize=figsize)
    
    if train_y is not None:
        plt.plot(train_y.index, train_y, 'k-', label='Training Data')
    
    if test_y is not None:
        plt.plot(test_y.index, test_y, 'b-', label='Test Data')
    
    if predictions is not None:
        plt.plot(predictions.index, predictions, 'r--', label='Test Predictions')
    
    if future_predictions is not None:
        plt.plot(future_predictions.index, future_predictions, 'g--', label='Future Predictions')
    
    if title:
        plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    
    return plt.gcf()

def plot_ensemble_forecast(
    ensemble_results: Dict,
    train_y=None,
    test_y=None,
    ensemble_type: str = 'auto_ensemble',
    figsize: Tuple[int, int] = (15, 7)
) -> plt.Figure:
    """
    Plot ensemble forecasts with training, test, and future predictions.
    
    Parameters
    ----------
    ensemble_results : dict
        Dictionary containing ensemble results
    train_y : pd.Series, optional
        Training data
    test_y : pd.Series, optional
        Test data
    ensemble_type : str, default='auto_ensemble'
        Type of ensemble to plot ('auto_ensemble' or 'online_ensemble')
    figsize : tuple, default=(15, 7)
        Figure size
        
    Returns
    -------
    matplotlib.figure.Figure
        The figure object
    """
    results = ensemble_results.get(ensemble_type)
    if results is None:
        raise ValueError(f"No results found for {ensemble_type}")
    
    title = f"{ensemble_type.replace('_', ' ').title()} Forecasts"
    
    return plot_individual_forecast(
        train_y=train_y,
        test_y=test_y,
        predictions=results.get('test_predictions'),
        future_predictions=results.get('future_predictions'),
        title=title,
        figsize=figsize
    )

def plot_cleaning_report(
    original_data,
    cleaned_data,
    figsize: Tuple[int, int] = (12, 6)
) -> Dict[str, plt.Figure]:
    """
    Create visualization plots for data cleaning report.
    
    Parameters
    ----------
    original_data : pd.DataFrame
        Original time series data
    cleaned_data : pd.DataFrame
        Cleaned time series data
    figsize : tuple, default=(12, 6)
        Base figure size
        
    Returns
    -------
    dict
        Dictionary containing figure objects for time series and distribution plots
    """
    # Convert PeriodIndex to DatetimeIndex if needed
    original_data = _convert_period_index(original_data)
    cleaned_data = _convert_period_index(cleaned_data)
    
    plots = {}
    
    # Time series plot
    fig_time_series = plt.figure(figsize=figsize)
    for col in cleaned_data.columns:
        if col in original_data.columns:
            plt.plot(original_data.index, original_data[col], 
                    alpha=0.5, label=f"{col} (original)")
            plt.plot(cleaned_data.index, cleaned_data[col], 
                    label=f"{col} (cleaned)")
    plt.title("Original vs Cleaned Time Series")
    plt.legend()
    plots['time_series'] = fig_time_series
    
    # Distribution plot
    fig_dist = plt.figure(figsize=(figsize[0], figsize[1] * len(cleaned_data.columns)))
    for i, col in enumerate(cleaned_data.columns):
        if col in original_data.columns:
            plt.subplot(len(cleaned_data.columns), 1, i+1)
            sns.kdeplot(original_data[col].dropna(), label="Original")
            sns.kdeplot(cleaned_data[col].dropna(), label="Cleaned")
            plt.title(f"Distribution of {col}")
            plt.legend()
    plots['distribution'] = fig_dist
    
    return plots

def plot_model_weights(
    model_weights: Dict[str, float],
    title: str = "Model Weights in Ensemble",
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Plot model weights in ensemble.
    
    Parameters
    ----------
    model_weights : dict
        Dictionary of model names and their weights
    title : str, default="Model Weights in Ensemble"
        Plot title
    figsize : tuple, default=(10, 6)
        Figure size
        
    Returns
    -------
    matplotlib.figure.Figure
        The figure object
    """
    plt.figure(figsize=figsize)
    models = list(model_weights.keys())
    weights = list(model_weights.values())
    
    plt.bar(models, weights)
    plt.title(title)
    plt.xlabel('Models')
    plt.ylabel('Weight')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return plt.gcf()

def compare_and_select_best_model(
    model_results: Dict,
    metric: str = 'mape',
    return_comparison: bool = True
) -> Tuple[str, pd.Series, Dict]:
    """
    Compare models based on specified metric and return the best model's future predictions.
    
    Parameters
    ----------
    model_results : dict
        Dictionary containing results from all models including individual and ensemble models
    metric : str, default='mape'
        Metric to use for comparison ('rmse', 'mape', 'mae', 'mse')
    return_comparison : bool, default=True
        Whether to return the comparison DataFrame
        
    Returns
    -------
    tuple
        - best_model: str, name of the best performing model
        - future_predictions: pd.Series, future predictions from best model
        - comparison_dict: dict containing:
            - 'metrics_comparison': pd.DataFrame with all metrics for all models
            - 'best_metrics': dict of best metrics and corresponding models
    """
    # Initialize storage for metrics
    metrics_data = []
    available_metrics = ['rmse', 'mape', 'mae', 'mse']
    
    # Collect metrics from all models
    for model_name, results in model_results.items():
        if 'metrics' in results:
            metric_values = results['metrics']
            metric_row = {
                'model': model_name,
                **{m: metric_values.get(m, None) for m in available_metrics}
            }
            metrics_data.append(metric_row)
    
    # Create comparison DataFrame
    metrics_df = pd.DataFrame(metrics_data)
    metrics_df = metrics_df.set_index('model')
    
    # Find best model for each metric
    best_metrics = {}
    for m in available_metrics:
        if m in metrics_df.columns:
            if m in ['rmse', 'mape', 'mae', 'mse']:  # Lower is better
                best_value = metrics_df[m].min()
                best_model = metrics_df[m].idxmin()
            else:  # Higher is better
                best_value = metrics_df[m].max()
                best_model = metrics_df[m].idxmax()
            
            best_metrics[m] = {
                'model': best_model,
                'value': best_value
            }
    
    # Get best model based on specified metric
    best_model = best_metrics[metric]['model']
    future_predictions = model_results[best_model]['future_predictions']
    
    # Format comparison dictionary
    comparison_dict = {
        'metrics_comparison': metrics_df,
        'best_metrics': best_metrics
    }
    
    # Print comparison if requested
    if return_comparison:
        print("\nMetrics Comparison:")
        print("-" * 50)
        print(metrics_df.round(4))
        print("\nBest Model for each metric:")
        print("-" * 50)
        for m, info in best_metrics.items():
            print(f"{m.upper()}: {info['model']} ({info['value']:.4f})")
        print(f"\nSelected best model ({metric}): {best_model}")
    
    return best_model, future_predictions, comparison_dict

def plot_model_comparison(
    comparison_dict: Dict,
    metric: str = None,
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """
    Create a visualization of model comparison metrics.
    
    Parameters
    ----------
    comparison_dict : dict
        Dictionary containing metrics comparison data
    metric : str, optional
        Specific metric to highlight
    figsize : tuple, default=(12, 6)
        Figure size
        
    Returns
    -------
    matplotlib.figure.Figure
        The figure object
    """
    metrics_df = comparison_dict['metrics_comparison']
    
    plt.figure(figsize=figsize)
    
    # Create bar plot for each metric
    metrics = metrics_df.columns
    n_metrics = len(metrics)
    
    fig, axes = plt.subplots(1, n_metrics, figsize=(figsize[0] * n_metrics/2, figsize[1]))
    if n_metrics == 1:
        axes = [axes]
    
    for ax, m in zip(axes, metrics):
        bars = ax.bar(metrics_df.index, metrics_df[m])
        ax.set_title(f'{m.upper()} by Model')
        ax.set_xticklabels(metrics_df.index, rotation=45, ha='right')
        
        # Highlight best model for this metric
        best_model = comparison_dict['best_metrics'][m]['model']
        best_idx = metrics_df.index.get_loc(best_model)
        bars[best_idx].set_color('green')
        
        # Highlight specified metric if provided
        if metric and m == metric:
            for bar in bars:
                bar.set_alpha(1.0)
            ax.set_title(f'{m.upper()} by Model (Selected Metric)', color='darkgreen')
        else:
            for bar in bars:
                bar.set_alpha(0.7)
    
    plt.tight_layout()
    return plt.gcf() 