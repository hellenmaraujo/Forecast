from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any
import joblib, pandas as pd
from src.core.io import save_csv
from src.core.logging_utils import get_logger
from src.forecast.model_selection import cv_and_score

log = get_logger(__name__)

def train_models(cfg: Dict[str, Any]) -> None:
    paths = cfg["project"]["paths"]
    proc = Path(paths["data_processed"])
    df = pd.read_csv(proc / "train_features.csv")
    meta = json.loads((proc / "feature_metadata.json").read_text(encoding="utf-8"))
    ts, target = meta["timestamp_column"], meta["target_column"]
    group_cols = meta["group_columns"]

    features = [c for c in df.columns if c not in group_cols + [ts, target]]
    results = []
    models_cfg = cfg["forecast"]["models"]

    for fam, gdf in df.groupby(group_cols[0]):
        X = gdf[features]; y = gdf[target]
        fam_best = {"model": None, "scores": {"mape": 1e18}, "estimator": None}
        for name, mcfg in models_cfg.items():
            if not mcfg.get("enabled", False):
                continue
            grid = {k: (v[0] if isinstance(v, list) else v) for k, v in mcfg.items() if k != "enabled"}
            scores, est = cv_and_score(X, y, name, grid, folds=cfg["forecast"]["folds"])
            results.append({"familia": fam, "modelo": name, **scores})
            if scores["mape"] < fam_best["scores"]["mape"]:
                fam_best = {"model": name, "scores": scores, "estimator": est}
        if fam_best["estimator"] is None:
            log.warning("Família %s sem modelo treinado.", fam)
            continue
        out_path = Path(paths["models_dir"]) / f"best_{fam}_{fam_best['model']}.joblib"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"estimator": fam_best["estimator"], "features": features}, out_path)
        log.info("Fam %s | best=%s | MAPE=%.2f", fam, fam_best["model"], fam_best["scores"]["mape"])

    if results:
        save_csv(pd.DataFrame(results), Path(paths["reports_tables"]) / "ranking_modelos.csv")
