"""
AdaptiveForecaster - A flexible class for time series forecasting with sktime
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Union, Dict, Any, Optional, Tuple

from sktime.forecasting.model_selection import (
    ForecastingGridSearchCV, 
    temporal_train_test_split
)
from sktime.split import (
    ExpandingWindowSplitter, 
    SlidingWindowSplitter,
    TemporalTrainTestSplitter
)
from sktime.performance_metrics.forecasting import (
    MeanAbsolutePercentageError, 
    MeanSquaredError,
    MeanAbsoluteError
)

# Import our models factory
from models import get_forecaster, ALGORITHM_MAP

class AdaptiveForecaster:
    """
    A flexible class for time series forecasting that allows users to:
    - Choose from various forecasting algorithms
    - Apply different cross-validation strategies
    - Select performance metrics
    - Fit, predict and evaluate models
    - Visualize results
    
    This class wraps around the forecaster factory in models.py
    """
    
    def __init__(
        self,
        algorithm: str = 'arima',
        seasonal_period: int = 12,
        fh: Union[int, List[int]] = 3,
        cv_strategy: str = 'expanding',
        cv_window_length: int = 12,
        cv_step_length: int = 1,
        cv_initial_window: int = 24,
        test_size: float = 0.2,
        metric: str = 'rmse',
        n_jobs: int = -1,
        verbose: int = 1
    ):
        """
        Initialize the AdaptiveForecaster.
        
        Parameters
        ----------
        algorithm : str, default='arima'
            Name of the forecasting algorithm to use. Must be a key in ALGORITHM_MAP.
        seasonal_period : int, default=12
            The seasonal period of the time series.
        fh : int or list of ints, default=3
            The forecast horizon.
        cv_strategy : str, default='expanding'
            The cross-validation strategy to use. 
            Options: 'expanding', 'sliding', 'temporal'.
        cv_window_length : int, default=12
            The length of the window for cross-validation.
        cv_step_length : int, default=1
            The step length for sliding and expanding window cross-validation.
        cv_initial_window : int, default=24
            The initial window size for expanding window cross-validation.
        test_size : float, default=0.2
            The proportion of the dataset to include in the test split when using
            temporal_train_test_split.
        metric : str, default='rmse'
            The performance metric to use. 
            Options: 'rmse', 'mse', 'mae', 'mape'.
        n_jobs : int, default=-1
            The number of jobs to run in parallel.
        verbose : int, default=1
            The verbosity level.
        """
        # Validate algorithm
        if algorithm not in ALGORITHM_MAP:
            raise ValueError(f"Unknown algorithm: {algorithm}. Available: {list(ALGORITHM_MAP.keys())}")
        
        self.algorithm = algorithm
        self.seasonal_period = seasonal_period
        self.fh = fh
        self.cv_strategy = cv_strategy
        self.cv_window_length = cv_window_length
        self.cv_step_length = cv_step_length
        self.cv_initial_window = cv_initial_window
        self.test_size = test_size
        self.metric_name = metric
        self.n_jobs = n_jobs
        self.verbose = verbose
        
        # Initialize attributes that will be set later
        self.forecaster = None
        self.cv_splitter = None
        self.grid_search = None
        self.best_params = None
        self.best_score = None
        self.predictions = None
        self.prediction_intervals = None
        self.train_y = None
        self.test_y = None
        self.metric_func = None
        self.test_score = None
        
        # Set up the forecaster and cross-validation
        self._setup_forecaster()
        self._setup_metric()
    
    def _setup_forecaster(self):
        """Set up the forecaster using models.py factory function."""
        self.forecaster = get_forecaster(
            algorithm=self.algorithm,
            seasonal_period=self.seasonal_period
        )
    
    def _setup_metric(self):
        """Set up the performance metric function."""
        self.metrics_map = {
            'rmse': MeanSquaredError(square_root=True),
            'mse': MeanSquaredError(square_root=False),
            'mae': MeanAbsoluteError(),
            'mape': MeanAbsolutePercentageError()
        }
        
        if self.metric_name not in self.metrics_map:
            raise ValueError(f"Unknown metric: {self.metric_name}. Available: {list(self.metrics_map.keys())}")
        
        self.metric_func = self.metrics_map[self.metric_name]
    
    def _setup_cv_splitter(self, y=None):
        """Set up the cross-validation splitter based on cv_strategy."""
        if self.cv_strategy == 'expanding':
            self.cv_splitter = ExpandingWindowSplitter(
                initial_window=self.cv_initial_window,
                step_length=self.cv_step_length,
                fh=self.fh
            )
        elif self.cv_strategy == 'sliding':
            self.cv_splitter = SlidingWindowSplitter(
                window_length=self.cv_window_length,
                step_length=self.cv_step_length,
                fh=self.fh
            )
        elif self.cv_strategy == 'temporal':
            if y is None:
                raise ValueError("For temporal CV, y must be provided at initialization")
            self.cv_splitter = TemporalTrainTestSplitter(
                test_size=self.test_size
            )
        else:
            raise ValueError(f"Unknown CV strategy: {self.cv_strategy}. "
                             f"Available: ['expanding', 'sliding', 'temporal']")
    
    def setup_grid_search(self, y=None, param_grid=None):
        """
        Set up the grid search cross-validation.
        
        Parameters
        ----------
        y : pd.Series, optional
            The time series data. Required for temporal CV.
        param_grid : dict, optional
            The parameter grid for grid search. If None, uses the default from the forecaster.
        """
        self._setup_cv_splitter(y)
        
        if param_grid is None:
            param_grid = self.forecaster.get_param_grid()
        
        self.grid_search = ForecastingGridSearchCV(
            forecaster=self.forecaster.permuted,
            param_grid=param_grid,
            cv=self.cv_splitter,
            scoring=self.metric_func,
            n_jobs=self.n_jobs,
            verbose=self.verbose
        )
        
        return self
    
    def split_data(self, y, test_size=None):
        """
        Split the data into training and test sets.
        
        Parameters
        ----------
        y : pd.Series
            The time series data.
        test_size : float, optional
            The proportion of the dataset to include in the test split.
            If None, uses the value from initialization.
        
        Returns
        -------
        train_y, test_y : pd.Series
            The training and test data.
        """
        if test_size is None:
            test_size = self.test_size
        
        self.train_y, self.test_y = temporal_train_test_split(y, test_size=test_size)
        return self.train_y, self.test_y
    
    def fit(self, y, X=None, fh=None, use_test_set=False):
        """
        Fit the forecaster to the training data.
        
        Parameters
        ----------
        y : pd.Series
            The time series data.
        X : pd.DataFrame, optional
            Exogenous variables.
        fh : int or list of ints, optional
            The forecast horizon. If None, uses the value from initialization.
        use_test_set : bool, default=False
            Whether to split the data into train/test sets for final evaluation.
            If True, y is split internally and only the training portion is used for fitting.
        
        Returns
        -------
        self : AdaptiveForecaster
            The fitted forecaster.
        """
        if fh is not None:
            self.fh = fh
        
        # Handle train/test split if requested
        if use_test_set and self.train_y is None:
            print("Splitting data into train/test sets...")
            self.train_y, self.test_y = self.split_data(y)
            # Use training data for fitting
            y_to_fit = self.train_y
        else:
            # Use all data for fitting
            y_to_fit = y
        
        if self.grid_search is None:
            self.setup_grid_search(y_to_fit)
        
        print(f"Fitting {self.algorithm} forecaster with {self.cv_strategy} cross-validation...")
        self.grid_search.fit(y_to_fit, X=X, fh=self.fh)
        
        self.best_params = self.grid_search.best_params_
        self.best_score = self.grid_search.best_score_
        
        return self
    
    def predict(self, fh=None, X=None, return_pred_int=False, coverage=[0.95]):
        """
        Make predictions with the fitted forecaster.
        
        Parameters
        ----------
        fh : int or list of ints, optional
            The forecast horizon. If None, uses the value from initialization.
        X : pd.DataFrame, optional
            Exogenous variables.
        return_pred_int : bool, default=False
            Whether to return prediction intervals.
        coverage : list of float, default=[0.95]
            The coverage of prediction intervals.
        
        Returns
        -------
        predictions : pd.Series
            The point forecasts.
        """
        if self.grid_search is None or not hasattr(self.grid_search, 'best_forecaster_'):
            raise ValueError("Forecaster has not been fitted yet. Call fit() first.")
        
        if fh is None:
            fh = self.fh
        
        print("Making predictions...")
        self.predictions = self.grid_search.best_forecaster_.predict(fh=fh, X=X)
        
        if return_pred_int:
            try:
                self.prediction_intervals = self.grid_search.best_forecaster_.predict_interval(
                    fh=fh, X=X, coverage=coverage
                )
            except Exception as e:
                print(f"Warning: Could not compute prediction intervals: {e}")
                self.prediction_intervals = None
        
        return self.predictions
    
    def evaluate(self, y_true=None, in_sample=False, metrics=None):
        """
        Evaluate the forecaster on test data or in-sample using multiple metrics.
        
        Parameters
        ----------
        y_true : pd.Series, optional
            The true values to compare against. If None, uses self.test_y or the training data.
        in_sample : bool, default=False
            Whether to evaluate on the training data (in-sample) instead of test data.
            If True and y_true is None, will use the training data.
        metrics : list of str, optional
            List of metric names to compute. If None, computes all available metrics.
            Available: 'rmse', 'mse', 'mae', 'mape'
        
        Returns
        -------
        scores : dict
            Dictionary with metric names as keys and scores as values.
        """
        if self.predictions is None:
            raise ValueError("No predictions available. Call predict() first.")
        
        if y_true is None:
            if in_sample:
                if self.train_y is not None:
                    y_true = self.train_y
                else:
                    raise ValueError("No training data available. Provide y_true or call split_data() first.")
            else:
                if self.test_y is not None:
                    y_true = self.test_y
                else:
                    raise ValueError("No test data available. Provide y_true, call split_data(), "
                                   "or set in_sample=True to evaluate on training data.")
        
        # Align indices for evaluation
        # Only use the indices that appear in both series
        common_indices = y_true.index.intersection(self.predictions.index)
        if len(common_indices) == 0:
            raise ValueError("No common indices between true values and predictions")
        
        y_true_aligned = y_true.loc[common_indices]
        predictions_aligned = self.predictions.loc[common_indices]
        
        # Determine which metrics to compute
        if metrics is None:
            metrics_to_compute = list(self.metrics_map.keys())
        else:
            invalid_metrics = [m for m in metrics if m not in self.metrics_map]
            if invalid_metrics:
                raise ValueError(f"Unknown metrics: {invalid_metrics}. Available: {list(self.metrics_map.keys())}")
            metrics_to_compute = metrics
        
        # Compute all requested metrics
        scores = {}
        for metric_name in metrics_to_compute:
            metric_func = self.metrics_map[metric_name]
            scores[metric_name] = metric_func(y_true_aligned, predictions_aligned)
        
        # Store the main metric score
        self.test_score = scores[self.metric_name]
        
        # Print results
        eval_type = "In-sample" if in_sample else "Test"
        print(f"\n{eval_type} Performance Metrics:")
        print("-" * 30)
        for metric_name, score in scores.items():
            print(f"{metric_name.upper()}: {score:.4f}")
        
        return scores
    
    def plot_forecasts(self, y=None, title=None, figsize=(15, 7), include_intervals=True):
        """
        Plot the time series, forecasts, and prediction intervals.
        
        Parameters
        ----------
        y : pd.Series, optional
            The full time series data. If None, uses the train and test data.
        title : str, optional
            The title of the plot.
        figsize : tuple, default=(15, 7)
            The figure size.
        include_intervals : bool, default=True
            Whether to include prediction intervals in the plot.
        
        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object.
        """
        if self.predictions is None:
            raise ValueError("No predictions available. Call predict() first.")
        
        plt.figure(figsize=figsize)
        
        # Plot historical data
        if y is not None:
            plt.plot(y.index, y, 'k-', label='Historical Data')
        else:
            if self.train_y is not None:
                plt.plot(self.train_y.index, self.train_y, 'k-', label='Training Data')
            if self.test_y is not None:
                plt.plot(self.test_y.index, self.test_y, 'b-', label='Test Data')
        
        # Plot predictions
        plt.plot(self.predictions.index, self.predictions, 'r-', 
                 label=f'{self.algorithm.capitalize()} Forecast')
        
        # Plot prediction intervals if available
        if include_intervals and self.prediction_intervals is not None:
            for coverage in self.prediction_intervals.index.get_level_values(0).unique():
                lower = self.prediction_intervals.loc[coverage]["lower"]
                upper = self.prediction_intervals.loc[coverage]["upper"]
                plt.fill_between(
                    lower.index, lower, upper, alpha=0.2, color='r',
                    label=f"{int(coverage*100)}% Prediction Interval"
                )
        
        # Set title and labels
        if title is None:
            title = f"{self.algorithm.capitalize()} Forecast with {self.cv_strategy.capitalize()} CV"
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        
        return plt.gcf()
    
    def summary(self, include_metrics=None):
        """
        Print a summary of the forecasting results.
        
        Parameters
        ----------
        include_metrics : list of str, optional
            List of metric names to include in summary. If None, includes the main metric.
            
        Returns
        -------
        summary_dict : dict
            A dictionary containing the summary information.
        """
        if self.grid_search is None or not hasattr(self.grid_search, 'best_forecaster_'):
            raise ValueError("Forecaster has not been fitted yet. Call fit() first.")
        
        # Compute additional metrics if requested
        all_metrics = {}
        if include_metrics and self.test_y is not None and self.predictions is not None:
            all_metrics = self.evaluate(metrics=include_metrics)
        
        summary_dict = {
            "Algorithm": self.algorithm,
            "CV Strategy": self.cv_strategy,
            "Primary Metric": self.metric_name.upper(),
            "CV Score": self.best_score,
            "Test Score": self.test_score if self.test_score is not None else "Not evaluated",
            "Best Parameters": self.best_params,
        }
        
        # Add other metrics if available
        for metric, score in all_metrics.items():
            if metric != self.metric_name:  # Skip primary metric as it's already included
                summary_dict[f"{metric.upper()} Score"] = score
        
        print("\n" + "="*50)
        print(f"ADAPTIVE FORECASTER SUMMARY")
        print("="*50)
        
        # Print algorithm and strategy first
        print(f"Algorithm: {summary_dict['Algorithm']}")
        print(f"CV Strategy: {summary_dict['CV Strategy']}")
        
        # Print metrics section
        print("\nMetrics:")
        print(f"  Primary ({summary_dict['Primary Metric']})")
        print(f"    CV: {summary_dict['CV Score']:.4f}")
        print(f"    Test: {summary_dict['Test Score'] if isinstance(summary_dict['Test Score'], str) else summary_dict['Test Score']:.4f}")
        
        # Print additional metrics if available
        for k, v in summary_dict.items():
            if k.endswith(' Score') and k not in ['CV Score', 'Test Score']:
                print(f"  {k}: {v:.4f}")
        
        # Print best parameters
        print("\nBest Parameters:")
        for param, value in summary_dict['Best Parameters'].items():
            print(f"  {param}: {value}")
        
        print("="*50 + "\n")
        
        return summary_dict

# Example usage
if __name__ == "__main__":
    from sktime.datasets import load_airline
    
    # Load data
    y = load_airline()
    
    # Create and configure forecaster
    forecaster = AdaptiveForecaster(
        algorithm='exp_smoothing',
        seasonal_period=12,
        fh=3,
        cv_strategy='expanding',
        cv_initial_window=36,
        metric='rmse'
    )
    
    # Split data
    train_y, test_y = forecaster.split_data(y)
    
    # Fit and predict
    forecaster.fit(train_y)
    predictions = forecaster.predict(return_pred_int=True)
    
    # Evaluate
    score = forecaster.evaluate()
    
    # Plot results
    fig = forecaster.plot_forecasts(y)
    plt.show()
    
    # Print summary
    forecaster.summary()