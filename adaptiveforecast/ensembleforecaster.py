from sktime.forecasting.compose import AutoEnsembleForecaster
from sktime.forecasting.online_learning import (
    NormalHedgeEnsemble,
    OnlineEnsembleForecaster,
)
from sktime.performance_metrics.forecasting import mean_absolute_percentage_error


class EnsembleManager:
    """
    Manages ensemble forecasting operations using multiple base forecasters.
    """
    def __init__(self, model_results: dict):
        """
        Initialize EnsembleManager with trained models.
        
        Parameters
        ----------
        model_results : dict
            Dictionary containing results from individual forecasters
        """
        self.model_results = model_results
        self.auto_ensemble = None
        self.online_ensemble = None
        self.train_y = None
        self.test_y = None
        self._setup_forecasters()
    
        
    def _setup_forecasters(self):
        """Create list of (name, forecaster) tuples from model results."""
        self.forecasters = [
            (algo, results['best_forecaster']) 
            for algo, results in self.model_results.items()
        ]
    
    def fit_auto_ensemble(self, y_train):
        """
        Fit AutoEnsembleForecaster with all base forecasters.
        
        Parameters
        ----------
        y_train : pd.Series
            Training data
        """
        self.auto_ensemble = AutoEnsembleForecaster(forecasters=self.forecasters)
        self.auto_ensemble.fit(y=y_train)
        return self.auto_ensemble
    
    def fit_online_ensemble(self, y_train, n_estimators=None):
        """
        Fit OnlineEnsembleForecaster with NormalHedgeEnsemble.
        
        Parameters
        ----------
        y_train : pd.Series
            Training data
        n_estimators : int, optional
            Number of estimators for NormalHedgeEnsemble
        """
        if n_estimators is None:
            n_estimators = len(self.forecasters)
            
        hedge_expert = NormalHedgeEnsemble(
            n_estimators=n_estimators,
            loss_func=mean_absolute_percentage_error
        )
        
        self.online_ensemble = OnlineEnsembleForecaster(
            forecasters=self.forecasters,
            ensemble_algorithm=hedge_expert
        )
        self.online_ensemble.fit(y=y_train)
        return self.online_ensemble
    
    def predict_ensembles(self, y_test, fh):
        """
        Generate predictions from both ensemble methods.
        
        Parameters
        ----------
        y_test : pd.Series
            Test data for online ensemble updating
        fh : list
            Forecast horizon
        
        Returns
        -------
        dict
            Dictionary containing predictions and metrics for both ensembles
        """
        results = {}
        
        # Auto ensemble predictions
        if self.auto_ensemble is not None:
            auto_pred = self.auto_ensemble.predict(fh=fh)
            auto_score = mean_absolute_percentage_error(y_test, auto_pred, symmetric=False)
            results['auto_ensemble'] = {
                'predictions': auto_pred,
                'mape': auto_score,
                'forecaster': self.auto_ensemble
            }
        
        # Online ensemble predictions
        if self.online_ensemble is not None:
            online_pred = self.online_ensemble.update_predict_single(y_test, fh=fh)
            online_score = mean_absolute_percentage_error(y_test, online_pred, symmetric=False)
            results['online_ensemble'] = {
                'predictions': online_pred,
                'mape': online_score,
                'forecaster': self.online_ensemble
            }
        
        return results
    
    def split_data(self, y, test_size=0.2):
        """
        Split data into training and test sets.
        
        Parameters
        ----------
        y : pd.Series
            Time series data
        test_size : float, default=0.2
            Proportion of data to use for testing
            
        Returns
        -------
        tuple
            (train_y, test_y)
        """
        from sktime.forecasting.model_selection import temporal_train_test_split
        self.train_y, self.test_y = temporal_train_test_split(y, test_size=test_size)
        return self.train_y, self.test_y
    
    
    
    def summary(self, include_base_models=True):
        """
        Print a summary of the ensemble results.
        
        Parameters
        ----------
        include_base_models : bool, default=True
            Whether to include base model information and weights in the summary
        """
        print("\n" + "="*50)
        print("ENSEMBLE FORECASTER SUMMARY")
        print("="*50)
        
        for ensemble_type in ['auto_ensemble', 'online_ensemble']:
            if ensemble_type in self.model_results:
                results = self.model_results[ensemble_type]
                print(f"\n{ensemble_type.upper()}")
                print("-"*30)
                print(f"Result ID: {results['result_id']}")
                
                print("\nMetrics:")
                for metric, value in results['metrics'].items():
                    print(f"  {metric.upper()}: {value:.4f}")
                
                if include_base_models and 'model_weights' in results:
                    print("\nModel Weights:")
                    for model, weight in results['model_weights'].items():
                        print(f"  {model}: {weight:.4f}")
        
        print("="*50 + "\n")


    def forecast_future(self, y, future_horizon=12):
        """
        Fit ensemble models and generate future predictions.
        
        Parameters
        ----------
        y : pd.Series
            The full time series dataset
        future_horizon : int, default=12
            Number of periods to forecast into the future
                
        Returns
        -------
        dict
            Dictionary containing ensemble results, forecasts, all metrics, and model weights
        """
        from datetime import datetime
        from sktime.performance_metrics.forecasting import (
            MeanAbsolutePercentageError,
            MeanSquaredError,
            MeanAbsoluteError
        )
        
        # Set up metrics
        metrics_map = {
            'rmse': MeanSquaredError(square_root=True),
            'mse': MeanSquaredError(square_root=False),
            'mae': MeanAbsoluteError(),
            'mape': MeanAbsolutePercentageError()
        }
        
        # Split data if not already done
        if self.train_y is None or self.test_y is None:
            self.split_data(y)
        
        # Fit both ensemble types
        print("Fitting Auto Ensemble Forecaster...")
        self.fit_auto_ensemble(self.train_y)
        
        print("Fitting Online Ensemble Forecaster...")
        self.fit_online_ensemble(self.train_y)
        
        # Generate forecasts
        future_fh = list(range(1, future_horizon + 1))
        test_fh = list(range(1, len(self.test_y) + 1))
        
        ensemble_results = {}
        
        # Auto Ensemble results
        if self.auto_ensemble is not None:
            print("Generating Auto Ensemble forecasts...")
            auto_test_pred = self.auto_ensemble.predict(fh=test_fh)
            
            # Calculate all metrics
            auto_metrics = {}
            for metric_name, metric_func in metrics_map.items():
                auto_metrics[metric_name] = metric_func(self.test_y, auto_test_pred)
            
            # Get model weights
            auto_weights = {}
            if hasattr(self.auto_ensemble, 'weights_'):
                for (name, _), weight in zip(self.forecasters, self.auto_ensemble.weights_):
                    auto_weights[name] = weight
            
            # Refit on full data and get future predictions
            self.auto_ensemble.fit(y)
            auto_future_pred = self.auto_ensemble.predict(fh=future_fh)
            
            result_id = f"auto_ensemble_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            ensemble_results['auto_ensemble'] = {
                "result_id": result_id,
                "forecaster": self.auto_ensemble,
                "test_predictions": auto_test_pred,
                "future_predictions": auto_future_pred,
                "metrics": auto_metrics,
                "model_weights": auto_weights,
                "base_forecasters": dict(self.forecasters)
            }
        
        # Online Ensemble results
        if self.online_ensemble is not None:
            print("Generating Online Ensemble forecasts...")
            online_test_pred = self.online_ensemble.update_predict_single(self.test_y, fh=test_fh)
            
            # Calculate all metrics
            online_metrics = {}
            for metric_name, metric_func in metrics_map.items():
                online_metrics[metric_name] = metric_func(self.test_y, online_test_pred)
            
            # Get model weights
            online_weights = {}
            if hasattr(self.online_ensemble.ensemble_algorithm, 'weights_'):
                for (name, _), weight in zip(self.forecasters, 
                                        self.online_ensemble.ensemble_algorithm.weights_):
                    online_weights[name] = weight
            
            # Refit on full data for future predictions
            self.online_ensemble.fit(y)
            online_future_pred = self.online_ensemble.predict(fh=future_fh)
            
            result_id = f"online_ensemble_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            ensemble_results['online_ensemble'] = {
                "result_id": result_id,
                "forecaster": self.online_ensemble,
                "test_predictions": online_test_pred,
                "future_predictions": online_future_pred,
                "metrics": online_metrics,
                "model_weights": online_weights,
                "base_forecasters": dict(self.forecasters)
            }
        
        return ensemble_results