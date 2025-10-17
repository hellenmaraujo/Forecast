from __future__ import annotations
import json, joblib
from pathlib import Path
from typing import Dict, Any
import pandas as pd, numpy as np
from sklearn.metrics import mean_absolute_error
from src.core.io import save_csv
from src.viz.visualization import plot_lines

def _rmse(y, p): return float(np.sqrt(np.mean((np.asarray(y)-np.asarray(p))**2)))
def _mape(y, p):
    y = np.asarray(y); p = np.asarray(p)
    denom = np.where(np.abs(y) < 1e-8, 1.0, np.abs(y))
    return float(np.mean(np.abs((y - p) / denom)) * 100)

def evaluate_models(cfg: Dict[str, Any]) -> None:
    paths = cfg["project"]["paths"]
    proc = Path(paths["data_processed"])
    df_test = pd.read_csv(proc / "test_features.csv")
    if df_test.empty:
        return
    meta = json.loads((proc / "feature_metadata.json").read_text(encoding="utf-8"))
    ts, target = meta["timestamp_column"], meta["target_column"]; fam_col = meta["group_columns"][0]
    results = []
    preds_all = []

    models_dir = Path(paths["models_dir"])
    for model_file in models_dir.glob("best_*.joblib"):
        fam = model_file.name.split("_")[1]
        bundle = joblib.load(model_file)
        feats = bundle["features"]; est = bundle["estimator"]
        gtest = df_test[df_test[fam_col] == fam].copy()
        if gtest.empty:
            continue
        y = gtest[target]; X = gtest[feats]
        pred = est.predict(X)
        mae = mean_absolute_error(y, pred); rmse = _rmse(y, pred); mape = _mape(y, pred)
        results.append({"familia": fam, "mae": mae, "rmse": rmse, "mape": mape})
        tmp = gtest[[ts]].copy(); tmp["real"] = y.values; tmp["previsto"] = pred
        tmp.sort_values(ts, inplace=True)
        preds_all.append(tmp.assign(familia=fam))
        plot_lines(tmp.rename(columns={ts: "data"}), "data", ["real", "previsto"],
                   f"Avaliação {fam}", Path(paths["reports_figures"]) / f"eval_{fam}.png")

    if results:
        save_csv(pd.DataFrame(results), Path(paths["reports_tables"]) / "metricas_teste.csv")
    if preds_all:
        save_csv(pd.concat(preds_all), Path(paths["reports_tables"]) / "predicoes_teste_detalhe.csv")
