from __future__ import annotations
import numpy as np, pandas as pd
from typing import Dict, Any, Tuple
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

def _mape(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = np.where(np.abs(y_true) < 1e-8, 1.0, np.abs(y_true))
    return np.mean(np.abs((y_true - y_pred) / denom)) * 100

def _rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred))**2)))

def _fit_model(name: str, X, y, params: Dict[str, Any]):
    if name == "random_forest":
        return RandomForestRegressor(
            n_estimators=params.get("n_estimators", 200),
            max_depth=params.get("max_depth", 10),
            min_samples_leaf=params.get("min_samples_leaf", 2),
            random_state=42, n_jobs=-1
        )
    if name == "xgboost":
        return XGBRegressor(
            n_estimators=params.get("n_estimators", 300),
            max_depth=params.get("max_depth", 6),
            learning_rate=params.get("learning_rate", 0.05),
            subsample=params.get("subsample", 0.8),
            colsample_bytree=params.get("colsample_bytree", 0.8),
            random_state=42, n_jobs=-1
        )
    raise ValueError(f"Modelo não suportado: {name}")

def cv_and_score(X: pd.DataFrame, y: pd.Series, model_name: str, grid: Dict[str, Any], folds: int = 3) -> Tuple[Dict[str, float], Any]:
    tscv = TimeSeriesSplit(n_splits=folds)
    best = {"mae": 1e18, "rmse": 1e18, "mape": 1e18}
    best_est = None
    for params in [grid]:
        maes, rmses, mapes = [], [], []
        for tr_idx, vl_idx in tscv.split(X):
            m = _fit_model(model_name, X.iloc[tr_idx], y.iloc[tr_idx], params)
            m.fit(X.iloc[tr_idx], y.iloc[tr_idx])
            pred = m.predict(X.iloc[vl_idx])
            maes.append(mean_absolute_error(y.iloc[vl_idx], pred))
            rmses.append(_rmse(y.iloc[vl_idx], pred))
            mapes.append(_mape(y.iloc[vl_idx], pred))
        scores = {"mae": float(np.mean(maes)), "rmse": float(np.mean(rmses)), "mape": float(np.mean(mapes))}
        if scores["mape"] < best["mape"]:
            best, best_est = scores, _fit_model(model_name, X, y, params)
    if best_est is None:
        best_est = _fit_model(model_name, X, y, grid)
    best_est.fit(X, y)
    return best, best_est
