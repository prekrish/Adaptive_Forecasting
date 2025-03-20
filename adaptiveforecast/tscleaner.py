"""
Time series data cleaning and preprocessing utilities.

This module focuses on cleaning and preparing time series data for 
forecasting models, providing utilities to handle common issues in time series.
"""

import logging
import warnings
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

import numpy as np
import pandas as pd
from sktime.transformations.series.outlier_detection import HampelFilter

logger = logging.getLogger(__name__)


class TSCleaner:
    """Time series data cleaner with common cleaning operations for time series data."""
    
    def __init__(self):
        """Initialize the time series cleaner."""
        self.cleaning_steps = []
        self._fitted = False
    
    def add_cleaning_step(self, cleaning_func: Callable, **kwargs) -> 'TSCleaner':
        """Add a cleaning step to the pipeline.
        
        Parameters
        ----------
        cleaning_func : Callable
            Function that takes a DataFrame and returns a cleaned DataFrame
        **kwargs : Dict
            Arguments to pass to the cleaning function
            
        Returns
        -------
        TSCleaner
            Self for method chaining
        """
        self.cleaning_steps.append((cleaning_func, kwargs))
        return self
    
    def fit(self, X: pd.DataFrame) -> 'TSCleaner':
        """Prepare the cleaner (no actual fitting needed for most cleaning operations).
        
        Parameters
        ----------
        X : pd.DataFrame
            Time series data
            
        Returns
        -------
        TSCleaner
            Self for method chaining
        """
        self._fitted = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply all cleaning steps to the data.
        
        Parameters
        ----------
        X : pd.DataFrame
            Time series data to clean
            
        Returns
        -------
        pd.DataFrame
            Cleaned time series data
        """
        X_cleaned = X.copy()
        
        for cleaning_func, kwargs in self.cleaning_steps:
            X_cleaned = cleaning_func(X_cleaned, **kwargs)
        
        return X_cleaned
    
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit and then clean the data.
        
        Parameters
        ----------
        X : pd.DataFrame
            Time series data to clean
            
        Returns
        -------
        pd.DataFrame
            Cleaned time series data
        """
        return self.fit(X).transform(X)


# Individual data cleaning functions

def remove_missing_values(
    X: pd.DataFrame, 
    method: str = 'linear', 
    limit: Optional[int] = None
) -> pd.DataFrame:
    """Handle missing values in time series data.
    
    Parameters
    ----------
    X : pd.DataFrame
        Time series data with potential missing values
    method : str, optional
        Interpolation method, by default 'linear'
        Options: 'linear', 'time', 'index', 'pad', 'nearest', 'zero', 'polynomial'
    limit : Optional[int], optional
        Maximum number of consecutive NaNs to fill, by default None
        
    Returns
    -------
    pd.DataFrame
        Time series data with missing values handled
    """
    # First interpolate with the specified method
    result = X.interpolate(method=method, limit=limit)
    
    # Then forward fill for any remaining NaNs at the beginning
    result = result.fillna(method='ffill')
    
    # Then backward fill for any remaining NaNs at the end
    result = result.fillna(method='bfill')
    
    return result


def detect_and_handle_outliers(
    X: pd.DataFrame,
    method: str = 'hampel',
    window_length: int = 10,
    n_sigma: float = 3.0,
    replace_method: str = 'interpolate'
) -> pd.DataFrame:
    """Detect and handle outliers in time series data.
    
    Parameters
    ----------
    X : pd.DataFrame
        Time series data
    method : str, optional
        Outlier detection method, by default 'hampel'
        Options: 'hampel', 'iqr', 'zscore'
    window_length : int, optional
        Window length for hampel filter, by default 10
    n_sigma : float, optional
        Number of standard deviations for outlier detection, by default 3.0
    replace_method : str, optional
        Method to replace outliers, by default 'interpolate'
        Options: 'interpolate', 'mean', 'median', 'mode', 'nan'
        
    Returns
    -------
    pd.DataFrame
        Time series data with outliers handled
    
    Raises
    ------
    ValueError
        If an unsupported method is specified
    """
    result = X.copy()
    
    if method.lower() == 'hampel':
        # Use Hampel filter from sktime
        hampel = HampelFilter(window_length=window_length, n_sigma=n_sigma)
        result = hampel.fit_transform(X)
    
    elif method.lower() == 'iqr':
        # IQR method
        for col in X.columns:
            series = X[col].copy()
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - n_sigma * iqr
            upper_bound = q3 + n_sigma * iqr
            
            # Create mask for outliers
            outlier_mask = (series < lower_bound) | (series > upper_bound)
            
            # Replace outliers
            if replace_method == 'interpolate':
                # Replace with NaN first, then interpolate
                series[outlier_mask] = np.nan
                series = series.interpolate(method='linear')
                series = series.fillna(method='ffill').fillna(method='bfill')
            elif replace_method == 'mean':
                series[outlier_mask] = series[~outlier_mask].mean()
            elif replace_method == 'median':
                series[outlier_mask] = series[~outlier_mask].median()
            elif replace_method == 'nan':
                series[outlier_mask] = np.nan
            else:
                raise ValueError(f"Unsupported replace method: {replace_method}")
            
            result[col] = series
    
    elif method.lower() == 'zscore':
        # Z-score method
        for col in X.columns:
            series = X[col].copy()
            mean = series.mean()
            std = series.std()
            z_scores = np.abs((series - mean) / std)
            
            # Create mask for outliers
            outlier_mask = z_scores > n_sigma
            
            # Replace outliers
            if replace_method == 'interpolate':
                # Replace with NaN first, then interpolate
                series[outlier_mask] = np.nan
                series = series.interpolate(method='linear')
                series = series.fillna(method='ffill').fillna(method='bfill')
            elif replace_method == 'mean':
                series[outlier_mask] = series[~outlier_mask].mean()
            elif replace_method == 'median':
                series[outlier_mask] = series[~outlier_mask].median()
            elif replace_method == 'nan':
                series[outlier_mask] = np.nan
            else:
                raise ValueError(f"Unsupported replace method: {replace_method}")
            
            result[col] = series
            
    else:
        raise ValueError(f"Unsupported outlier detection method: {method}")
    
    return result


def fix_index_and_frequency(
    X: pd.DataFrame, 
    freq: Optional[str] = None, 
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    fill_missing: bool = True
) -> pd.DataFrame:
    """Fix the index and ensure regular frequency for time series data.
    
    Parameters
    ----------
    X : pd.DataFrame
        Time series data with datetime index
    freq : Optional[str], optional
        Target frequency, by default None
        If None, tries to infer frequency from data
    start_date : Optional[str], optional
        Start date if you want to extend the series, by default None
    end_date : Optional[str], optional
        End date if you want to extend the series, by default None
    fill_missing : bool, optional
        Whether to fill missing values in the reindexed series, by default True
        
    Returns
    -------
    pd.DataFrame
        Time series data with fixed index and frequency
        
    Raises
    ------
    ValueError
        If input data does not have a datetime index
    """
    # Check if index is datetime
    if not isinstance(X.index, pd.DatetimeIndex):
        raise ValueError("Input data must have a datetime index")
    
    # Infer frequency if not provided
    if freq is None:
        freq = pd.infer_freq(X.index)
        if freq is None:
            warnings.warn(
                "Could not infer frequency from data. "
                "Using 'D' (daily) as default frequency."
            )
            freq = 'D'
    
    # Determine start and end dates
    if start_date is None:
        start_date = X.index.min()
    else:
        start_date = pd.to_datetime(start_date)
    
    if end_date is None:
        end_date = X.index.max()
    else:
        end_date = pd.to_datetime(end_date)
    
    # Create a complete datetime index with the specified frequency
    new_index = pd.date_range(start=start_date, end=end_date, freq=freq)
    
    # Reindex the data frame
    result = X.reindex(new_index)
    
    # Fill missing values if requested
    if fill_missing and result.isna().any().any():
        result = remove_missing_values(result)
    
    return result


def handle_duplicate_indices(
    X: pd.DataFrame, 
    method: str = 'mean'
) -> pd.DataFrame:
    """Handle duplicate indices in time series data.
    
    Parameters
    ----------
    X : pd.DataFrame
        Time series data
    method : str, optional
        Method to handle duplicates, by default 'mean'
        Options: 'mean', 'median', 'sum', 'min', 'max', 'first', 'last'
        
    Returns
    -------
    pd.DataFrame
        Time series data without duplicate indices
        
    Raises
    ------
    ValueError
        If an unsupported method is specified
    """
    # Check if there are any duplicate indices
    if not X.index.duplicated().any():
        return X
    
    # Group by index and aggregate
    if method == 'mean':
        return X.groupby(X.index).mean()
    elif method == 'median':
        return X.groupby(X.index).median()
    elif method == 'sum':
        return X.groupby(X.index).sum()
    elif method == 'min':
        return X.groupby(X.index).min()
    elif method == 'max':
        return X.groupby(X.index).max()
    elif method == 'first':
        return X.groupby(X.index).first()
    elif method == 'last':
        return X.groupby(X.index).last()
    else:
        raise ValueError(f"Unsupported method: {method}")


def remove_constant_columns(
    X: pd.DataFrame, 
    threshold: float = 0.0
) -> pd.DataFrame:
    """Remove columns with little or no variation.
    
    Parameters
    ----------
    X : pd.DataFrame
        Time series data
    threshold : float, optional
        Threshold for standard deviation, by default 0.0
        Columns with std <= threshold will be removed
        
    Returns
    -------
    pd.DataFrame
        Time series data with constant columns removed
    """
    # Calculate standard deviation for each column
    std = X.std()
    
    # Find columns to keep
    cols_to_keep = std[std > threshold].index
    
    if len(cols_to_keep) < len(X.columns):
        warnings.warn(
            f"Removing {len(X.columns) - len(cols_to_keep)} constant or near-constant columns"
        )
        return X[cols_to_keep]
    
    return X


def handle_extreme_values(
    X: pd.DataFrame,
    lower_quantile: float = 0.001,
    upper_quantile: float = 0.999,
    method: str = 'clip'
) -> pd.DataFrame:
    """Handle extreme values in time series data.
    
    Parameters
    ----------
    X : pd.DataFrame
        Time series data
    lower_quantile : float, optional
        Lower quantile for extreme values, by default 0.001
    upper_quantile : float, optional
        Upper quantile for extreme values, by default 0.999
    method : str, optional
        Method to handle extreme values, by default 'clip'
        Options: 'clip', 'remove', 'winsorize'
        
    Returns
    -------
    pd.DataFrame
        Time series data with extreme values handled
        
    Raises
    ------
    ValueError
        If an unsupported method is specified
    """
    result = X.copy()
    
    # Calculate lower and upper bounds
    lower_bounds = X.quantile(lower_quantile)
    upper_bounds = X.quantile(upper_quantile)
    
    if method == 'clip':
        # Clip values to the bounds
        for col in X.columns:
            result[col] = result[col].clip(
                lower=lower_bounds[col], 
                upper=upper_bounds[col]
            )
    
    elif method == 'remove':
        # Set extreme values to NaN and then interpolate
        for col in X.columns:
            mask = (result[col] < lower_bounds[col]) | (result[col] > upper_bounds[col])
            result.loc[mask, col] = np.nan
        
        # Interpolate missing values
        result = remove_missing_values(result)
    
    elif method == 'winsorize':
        # Replace extreme values with the bounds
        for col in X.columns:
            lower_mask = result[col] < lower_bounds[col]
            upper_mask = result[col] > upper_bounds[col]
            
            result.loc[lower_mask, col] = lower_bounds[col]
            result.loc[upper_mask, col] = upper_bounds[col]
    
    else:
        raise ValueError(f"Unsupported method: {method}")
    
    return result


def create_cleaning_pipeline(
    fix_index: bool = True,
    handle_duplicates: bool = True,
    remove_missing: bool = True,
    handle_outliers: bool = True,
    handle_extremes: bool = True,
    remove_constants: bool = True,
    index_params: Dict = None,
    duplicate_params: Dict = None,
    missing_params: Dict = None,
    outlier_params: Dict = None,
    extreme_params: Dict = None,
    constant_params: Dict = None
) -> TSCleaner:
    """Create a cleaning pipeline for time series data.
    
    Parameters
    ----------
    fix_index : bool, optional
        Whether to fix index and frequency, by default True
    handle_duplicates : bool, optional
        Whether to handle duplicate indices, by default True
    remove_missing : bool, optional
        Whether to remove missing values, by default True
    handle_outliers : bool, optional
        Whether to handle outliers, by default True
    handle_extremes : bool, optional
        Whether to handle extreme values, by default True
    remove_constants : bool, optional
        Whether to remove constant columns, by default True
    index_params : Dict, optional
        Parameters for fix_index_and_frequency, by default None
    duplicate_params : Dict, optional
        Parameters for handle_duplicate_indices, by default None
    missing_params : Dict, optional
        Parameters for remove_missing_values, by default None
    outlier_params : Dict, optional
        Parameters for detect_and_handle_outliers, by default None
    extreme_params : Dict, optional
        Parameters for handle_extreme_values, by default None
    constant_params : Dict, optional
        Parameters for remove_constant_columns, by default None
        
    Returns
    -------
    TSCleaner
        Cleaning pipeline for time series data
    """
    # Initialize parameter dictionaries
    index_params = index_params or {}
    duplicate_params = duplicate_params or {}
    missing_params = missing_params or {}
    outlier_params = outlier_params or {}
    extreme_params = extreme_params or {}
    constant_params = constant_params or {}
    
    # Create cleaner
    cleaner = TSCleaner()
    
    # The order of operations matters here:
    
    # 1. First, handle duplicate indices to ensure we have a unique index
    if handle_duplicates:
        cleaner.add_cleaning_step(handle_duplicate_indices, **duplicate_params)
    
    # 2. Fix index and frequency
    if fix_index:
        cleaner.add_cleaning_step(fix_index_and_frequency, **index_params)
    
    # 3. Remove constant columns that might interfere with further operations
    if remove_constants:
        cleaner.add_cleaning_step(remove_constant_columns, **constant_params)
    
    # 4. Handle extreme values
    if handle_extremes:
        cleaner.add_cleaning_step(handle_extreme_values, **extreme_params)
    
    # 5. Handle outliers
    if handle_outliers:
        cleaner.add_cleaning_step(detect_and_handle_outliers, **outlier_params)
    
    # 6. Finally handle any remaining missing values
    if remove_missing:
        cleaner.add_cleaning_step(remove_missing_values, **missing_params)
    
    return cleaner


def create_report(
    X: pd.DataFrame,
    clean_X: pd.DataFrame
) -> Dict:
    """Create a report on the data cleaning process.
    
    Parameters
    ----------
    X : pd.DataFrame
        Original time series data
    clean_X : pd.DataFrame
        Cleaned time series data
    
        
    Returns
    -------
    Dict
        Report containing statistics and visualizations
    """
    report = {
        'original_shape': X.shape,
        'cleaned_shape': clean_X.shape,
        'missing_values_before': X.isna().sum().to_dict(),
        'missing_values_after': clean_X.isna().sum().to_dict(),
        'duplicates_before': X.index.duplicated().sum(),
        'duplicates_after': clean_X.index.duplicated().sum(),
        'stats_before': {
            col: {
                'mean': X[col].mean(),
                'std': X[col].std(),
                'min': X[col].min(),
                'max': X[col].max(),
                'median': X[col].median(),
                'skew': X[col].skew(),
                'kurtosis': X[col].kurtosis()
            } for col in X.columns
        },
        'stats_after': {
            col: {
                'mean': clean_X[col].mean(),
                'std': clean_X[col].std(),
                'min': clean_X[col].min(),
                'max': clean_X[col].max(),
                'median': clean_X[col].median(),
                'skew': clean_X[col].skew(),
                'kurtosis': clean_X[col].kurtosis()
            } for col in clean_X.columns if col in X.columns
        },
        'columns_removed': [col for col in X.columns if col not in clean_X.columns],
        'rows_before': len(X),
        'rows_after': len(clean_X)
    }
    
    
    
    return report