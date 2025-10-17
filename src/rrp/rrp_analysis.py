from __future__ import annotations
from pathlib import Path
from typing import Dict, Any
import pandas as pd
from src.core.io import save_csv
from src.viz.visualization import plot_lines

def run_rrp_analysis(cfg: Dict[str, Any]) -> None:
    paths = cfg["project"]["paths"]
    plan_path = Path(paths["reports_tables"]) / "plano_agregado.csv"
    if not plan_path.exists():
        return
    plano = pd.read_csv(plan_path)
    if plano.empty:
        return
    prod_per_h = cfg["rrp"]["productivity_units_per_hour"]
    cap_hours = cfg["rrp"]["capacity_hours_per_period"]

    rows = []
    for data, group in plano.groupby("data"):
        hours_needed = 0.0
        for fam, fam_df in group.groupby("familia"):
            prod = max(0.0, float(fam_df["producao"].sum()))
            rate = float(prod_per_h.get(fam, 1.0))
            hours_needed += prod / rate if rate > 0 else 0.0
        capacity = float(cap_hours.get(data, cap_hours.get("default", 1e9)))
        utilization = hours_needed / capacity if capacity > 0 else 0.0
        rows.append({
            "data": data,
            "horas_necessarias": hours_needed,
            "capacidade_horas": capacity,
            "utilizacao": utilization,
        })

    out = pd.DataFrame(rows).sort_values("data")
    save_csv(out, Path(paths["reports_tables"]) / "rrp_resumo.csv")
    plot_lines(out, "data", ["horas_necessarias", "capacidade_horas"],
               "RRP - Horas vs Capacidade", Path(paths["reports_figures"]) / "rrp_resumo.png")
