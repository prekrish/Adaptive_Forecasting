"""
Module for creating and configuring forecasting models.
"""
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.arima import AutoARIMA
from sktime.forecasting.theta import ThetaForecaster
from sktime.forecasting.ets import AutoETS
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.forecasting.compose import TransformedTargetForecaster,Permute
from sktime.transformations.compose import OptionalPassthrough
from sktime.transformations.series.detrend import Deseasonalizer, Detrender
from sktime.transformations.series.boxcox import BoxCoxTransformer
from sktime.transformations.series.lag import Lag
from sktime.transformations.series.impute import Imputer
from sktime.transformations.series.adapt import TabularToSeriesAdaptor
from sklearn.preprocessing import RobustScaler
from sktime.forecasting.statsforecast import StatsForecastAutoARIMA, StatsForecastAutoETS, StatsForecastAutoTBATS
try:
    from sktime.forecasting.fbprophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
from abc import ABC, abstractmethod

__all__ = [
    'BaseForecaster',
    'NaiveAdapter',
    'AutoARIMAAdapter',
    'ExpSmoothingAdapter',
    'StatsAutoARIMAAdapter',
    'StatsAutoETSAdapter',
    'StatsAutoTBATSAdapter',
    'ProphetAdapter',
    'get_forecaster',
    'ALGORITHM_MAP'
]

class BaseForecaster(ABC):
    """Base class for all forecasting algorithms."""
    
    def __init__(self, seasonal_period=12):
        self.seasonal_period = seasonal_period
        self.base_transformers = [
            ("detrender", OptionalPassthrough(transformer=Detrender())),
            ("deseasonalizer", OptionalPassthrough(transformer=Deseasonalizer(sp=seasonal_period))),
            ("scaler", OptionalPassthrough(transformer=TabularToSeriesAdaptor(RobustScaler()))),
        ]
    
    @abstractmethod
    def get_forecaster(self):
        """Return configured forecaster"""
        pass
    
    @abstractmethod
    def get_param_grid(self):
        """Return parameter grid"""
        pass
    
    def get_base_param_grid(self):
        """Return base transformer parameter grid"""
        return {
            "estimator__detrender__passthrough": [True, False],
            "estimator__deseasonalizer__passthrough": [True, False],
            "estimator__scaler__passthrough": [True, False],
        }
    
    def create_pipeline(self):
        """Create forecasting pipeline with transformers"""
        return TransformedTargetForecaster(
            steps=self.base_transformers + [("forecaster", self.get_forecaster())]
        )

class NaiveAdapter(BaseForecaster):
    def get_forecaster(self):
        return NaiveForecaster(
            strategy="last",
            sp=self.seasonal_period
        )
    
    def get_param_grid(self):
        base_grid = self.get_base_param_grid()
        naive_grid = {
            "estimator__forecaster__strategy": ["last", "mean", "drift"],
            "estimator__forecaster__sp": [3,12],
            "estimator__forecaster__window_length": [None, 3, 6],
        }
        return {**base_grid, **naive_grid}

class AutoARIMAAdapter(BaseForecaster):
    def get_forecaster(self):
        return AutoARIMA(
            random_state=42,
            seasonal=True,
            sp=self.seasonal_period,
            max_order=5,
            stepwise=True
        )
    
    def get_param_grid(self):
        base_grid = self.get_base_param_grid()
        arima_grid = {
            "estimator__forecaster__d": [0, 1],
            "estimator__forecaster__D": [0, 1],
            "estimator__forecaster__start_p": [1],
            "estimator__forecaster__max_p": [3],
            "estimator__forecaster__start_q": [1],
            "estimator__forecaster__max_q": [3],
        }
        return {**base_grid, **arima_grid}

class ExpSmoothingAdapter(BaseForecaster):
    def get_forecaster(self):
        return ExponentialSmoothing(
            seasonal="add",
            sp=self.seasonal_period
        )
    
    def get_param_grid(self):
        base_grid = self.get_base_param_grid()
        exp_grid = {
            "estimator__forecaster__trend": [None, "add", "mul"],
            "estimator__forecaster__seasonal": [None, "add", "mul"],
            "estimator__forecaster__damped_trend": [True, False],
            "estimator__forecaster__sp": [12],
            "estimator__forecaster__initialization_method": ["estimated"],
        }
        return {**base_grid, **exp_grid}

class StatsAutoARIMAAdapter(BaseForecaster):
    def get_forecaster(self):
        return StatsForecastAutoARIMA(
            start_p=1,
            start_q=1,
            max_p=3,
            max_q=3,
            seasonal=True,
            stepwise=True,
            approximation=True
        )
    
    def get_param_grid(self):
        base_grid = self.get_base_param_grid()
        arima_grid = {
            "estimator__forecaster__d": [0, 1],
            "estimator__forecaster__D": [0, 1],
            "estimator__forecaster__max_P": [1],
            "estimator__forecaster__max_Q": [1],
            "estimator__forecaster__information_criterion": ['aicc'],
            "estimator__forecaster__seasonal": [True],
        }
        return {**base_grid, **arima_grid}

class StatsAutoETSAdapter(BaseForecaster):
    def get_forecaster(self):
        return StatsForecastAutoETS(
            season_length=self.seasonal_period,
            model="ZZZ"
        )
    
    def get_param_grid(self):
        base_grid = self.get_base_param_grid()
        ets_grid = {
            "estimator__forecaster__season_length": [4, 12],
            "estimator__forecaster__model": ["ZZZ", "ZZN", "ZNN", "ZAN"],
        }
        return {**base_grid, **ets_grid}

class StatsAutoTBATSAdapter(BaseForecaster):
    def get_forecaster(self):
        return StatsForecastAutoTBATS(
            use_boxcox=True,
            use_trend=True,
            use_damped_trend=True,
            seasonal_periods=[self.seasonal_period]
        )
    
    def get_param_grid(self):
        base_grid = self.get_base_param_grid()
        tbats_grid = {
            "estimator__forecaster__use_boxcox": [True, False],
            "estimator__forecaster__use_trend": [True, False],
            "estimator__forecaster__use_damped_trend": [True, False],
            "estimator__forecaster__seasonal_periods": [[4, 12]],
        }
        return {**base_grid, **tbats_grid}

class ProphetAdapter(BaseForecaster):
    def __init__(self, seasonal_period=12):
        super().__init__(seasonal_period)
        if not PROPHET_AVAILABLE:
            raise ImportError("Prophet is not available. Install it with: pip install prophet")
    
    def get_forecaster(self):
        return Prophet(
            seasonality_mode='multiplicative',
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            uncertainty_samples=0  # Disable uncertainty sampling for faster execution
        )
    
    def get_param_grid(self):
        base_grid = self.get_base_param_grid()
        prophet_grid = {
            "estimator__forecaster__seasonality_mode": ['additive', 'multiplicative'],
            "estimator__forecaster__changepoint_prior_scale": [0.05, 0.1],
            "estimator__forecaster__yearly_seasonality": [True],
        }
        return {**base_grid, **prophet_grid}

# Algorithm registry
ALGORITHM_MAP = {
    'naive': NaiveAdapter,
    'arima': AutoARIMAAdapter,
    'stats_arima': StatsAutoARIMAAdapter,
    'exp_smoothing': ExpSmoothingAdapter,
    'ets': StatsAutoETSAdapter,
    'tbats': StatsAutoTBATSAdapter,
    'prophet': ProphetAdapter,
}

class PermutedForecaster:
    """Wrapper class that combines Permute with param_grid access."""
    
    def __init__(self, forecaster):
        self.forecaster = forecaster
        self.permuted = Permute(forecaster.create_pipeline())
    
    def get_param_grid(self):
        """Get parameter grid from base forecaster."""
        return self.forecaster.get_param_grid()
    

def get_forecaster(algorithm: str, seasonal_period: int = 12):
    """Get forecaster by name"""
    if algorithm not in ALGORITHM_MAP:
        raise ValueError(f"Unknown algorithm: {algorithm}. Available: {list(ALGORITHM_MAP.keys())}")
    
    forecaster_class = ALGORITHM_MAP[algorithm]
    forecaster = forecaster_class(seasonal_period=seasonal_period)
    return PermutedForecaster(forecaster)  # Return wrapped forecaster