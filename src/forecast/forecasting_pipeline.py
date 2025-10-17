from __future__ import annotations
import json, joblib
from pathlib import Path
from typing import Dict, Any
import pandas as pd
from dateutil.relativedelta import relativedelta
from src.core.io import save_csv
from src.viz.visualization import plot_lines

def generate_forecast(cfg: Dict[str, Any]) -> None:
    paths = cfg["project"]["paths"]
    proc = Path(paths["data_processed"])
    df_test = pd.read_csv(proc / "test_features.csv")
    if df_test.empty:
        return
    meta = json.loads((proc / "feature_metadata.json").read_text(encoding="utf-8"))
    ts, fam_col, target = meta["timestamp_column"], meta["group_columns"][0], meta["target_column"]
    horizon = cfg["forecast"]["horizon_months"]

    out_rows = []
    models_dir = Path(paths["models_dir"])
    for model_file in models_dir.glob("best_*.joblib"):
        fam = model_file.name.split("_")[1]
        bundle = joblib.load(model_file)
        feats = bundle["features"]; est = bundle["estimator"]
        gtest = df_test[df_test[fam_col] == fam].copy()
        if gtest.empty:
            continue
        gtest.sort_values(ts, inplace=True)
        ref_row = gtest.iloc[-1:].copy()
        last_date = pd.to_datetime(ref_row[ts].iloc[0])
        current_features = ref_row[feats].iloc[0].copy()
        for h in range(1, horizon+1):
            prediction = float(est.predict([current_features.values])[0])
            out_rows.append({
                "familia": fam,
                "data": (last_date + relativedelta(months=h)).date().isoformat(),
                "demanda_prevista": prediction
            })

    out = pd.DataFrame(out_rows)
    if out.empty:
        return
    out.sort_values(["familia", "data"], inplace=True)
    save_csv(out, Path(paths["reports_tables"]) / "demanda_prevista.csv")
    plot_lines(out.groupby("data", as_index=False)["demanda_prevista"].sum(), "data", ["demanda_prevista"],
               "Demanda Prevista (MVP)", Path(paths["reports_figures"]) / "demanda_prevista.png")
