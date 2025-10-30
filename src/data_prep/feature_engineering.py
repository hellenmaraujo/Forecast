from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd
from src.core.io import save_csv, save_json
from src.core.logging_utils import get_logger

log = get_logger(__name__)

def _add_time_features(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    df = df.copy()
    df["year"] = df[date_col].dt.year
    df["month"] = df[date_col].dt.month
    df["quarter"] = df[date_col].dt.quarter
    return df

def _add_lags_rollings(df: pd.DataFrame, cfg: Dict[str, Any], group_cols: list[str], target: str) -> pd.DataFrame:
    lags = cfg["data_prep"]["lags"]
    wins = cfg["data_prep"]["rolling_windows"]
    df = df.copy()
    df = df.sort_values(group_cols + [cfg["data_prep"]["date_column"]])
    for g, gdf in df.groupby(group_cols):
        idx = gdf.index
        for L in lags:
            df.loc[idx, f"lag_{L}"] = gdf[target].shift(L)
        for W in wins:
            df.loc[idx, f"rollmean_{W}"] = gdf[target].shift(1).rolling(W).mean()
    return df

def build_features(cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    paths = cfg["project"]["paths"]
    c = cfg["data_prep"]
    train = pd.read_csv(Path(paths["data_interim"]) / "train_raw.csv", parse_dates=[c["date_column"]])
    test  = pd.read_csv(Path(paths["data_interim"]) / "test_raw.csv", parse_dates=[c["date_column"]])
    group_cols = [c["family_column"], c["product_column"]]
    train = _add_time_features(train, c["date_column"])
    test = _add_time_features(test, c["date_column"])
    train = _add_lags_rollings(train, cfg, group_cols, c["target_column"])
    test = _add_lags_rollings(test, cfg, group_cols, c["target_column"])
    min_lag = max([0] + c["lags"] + c["rolling_windows"])
    train = (train.groupby(group_cols, group_keys=False).apply(lambda d: d.iloc[min_lag:]).reset_index(drop=True))
    feature_cols = [col for col in train.columns if col not in [c["target_column"], c["date_column"]] + group_cols]
    train.dropna(subset=feature_cols, inplace=True)
    test.dropna(subset=feature_cols, inplace=True)
    meta = {
        "timestamp_column": c["date_column"],
        "group_columns": group_cols,
        "target_column": c["target_column"]
    }
    return train, test, meta

def persist_features(train: pd.DataFrame, test: pd.DataFrame, meta: Dict[str, Any], cfg: Dict[str, Any]) -> None:
    paths = cfg["project"]["paths"]
    save_csv(train, Path(paths["data_processed"]) / "train_features.csv")
    save_csv(test,  Path(paths["data_processed"]) / "test_features.csv")
    save_json(meta, Path(paths["data_processed"]) / "feature_metadata.json")
    log.info("Features salvas.")
