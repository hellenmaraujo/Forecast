from __future__ import annotations

from copy import deepcopy
from typing import Dict, Tuple

import pandas as pd
import pytest

from src.core.io import load_config
from src.core.logging_utils import configure_logging
from src.core.utils import set_seed
from src.data_prep.data_preparation import run_data_preparation
from src.data_prep.feature_engineering import build_features, persist_features
from src.forecast.forecasting_pipeline import generate_forecast
from src.forecast.train_models import train_models
from src.optimization.optimization_plan import run_optimization
from src.rrp.rrp_analysis import run_rrp_analysis


@pytest.fixture(scope="session")
def config() -> Dict:
    cfg = deepcopy(load_config("configs/config.yaml"))
    cfg["forecast"]["folds"] = 2
    cfg["forecast"]["models"]["random_forest"]["n_estimators"] = [50]
    cfg["forecast"]["models"]["xgboost"]["n_estimators"] = [100]
    cfg["forecast"]["horizon_months"] = 3
    configure_logging(cfg["project"]["paths"]["logging_config"])
    set_seed(cfg["project"]["seed"])
    return cfg


@pytest.fixture(scope="session")
def prepared_features(config: Dict) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    run_data_preparation(config)
    train_features, test_features, metadata = build_features(config)
    persist_features(train_features, test_features, metadata, config)
    return train_features, test_features, metadata


@pytest.fixture(scope="session")
def trained_models(config: Dict, prepared_features) -> None:
    train_models(config)


@pytest.fixture(scope="session")
def forecast_output(config: Dict, trained_models) -> pd.DataFrame:
    return generate_forecast(config)


@pytest.fixture(scope="session")
def optimized_plan(config: Dict, forecast_output) -> pd.DataFrame:
    return run_optimization(config)


@pytest.fixture(scope="session")
def rrp_summary(config: Dict, optimized_plan) -> pd.DataFrame:
    return run_rrp_analysis(config)
