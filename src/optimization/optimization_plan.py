from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd
import pulp
from src.core.io import save_csv
from src.viz.visualization import plot_lines

def _costs_for(fam: str, cfg: Dict[str, Any]) -> Tuple[float, float, float]:
    cp = cfg["optimization"]["cost_parameters"]
    params = cp.get(fam, cp["default"])
    return params["production"], params["inventory"], params["backlog"]

def _capacity_for(period: str, cfg: Dict[str, Any]) -> float:
    cap = cfg["optimization"]["capacity_per_period"]
    return cap.get(period, cap.get("default", 1e9))

def run_optimization(cfg: Dict[str, Any]) -> None:
    paths = cfg["project"]["paths"]
    demand_path = Path(paths["reports_tables"]) / "demanda_prevista.csv"
    if not demand_path.exists():
        return
    demand = pd.read_csv(demand_path)
    if demand.empty:
        return
    fams = sorted(demand["familia"].unique())
    periods = sorted(demand["data"].unique())
    dem = {(r.familia, r.data): float(r.demanda_prevista) for r in demand.itertuples()}

    model = pulp.LpProblem("PlanoAgregado", pulp.LpMinimize)
    prod = pulp.LpVariable.dicts("prod", (fams, periods), lowBound=0)
    inv = pulp.LpVariable.dicts("inv", (fams, periods), lowBound=0)
    backlog = pulp.LpVariable.dicts("backlog", (fams, periods), lowBound=0)

    # objective
    model += pulp.lpSum(
        prod[f][t] * _costs_for(f, cfg)[0]
        + inv[f][t] * _costs_for(f, cfg)[1]
        + backlog[f][t] * _costs_for(f, cfg)[2]
        for f in fams for t in periods
    )

    initial_inventory = cfg["optimization"]["initial_inventory"]
    for f in fams:
        prev_inv = initial_inventory.get(f, 0.0)
        prev_backlog = 0.0
        for t in periods:
            demand_value = dem.get((f, t), 0.0)
            model += prev_inv + prod[f][t] - demand_value + prev_backlog == inv[f][t] + backlog[f][t]
            prev_inv = inv[f][t]
            prev_backlog = backlog[f][t]

    for t in periods:
        model += pulp.lpSum(prod[f][t] for f in fams) <= _capacity_for(t, cfg)

    model.solve(pulp.PULP_CBC_CMD(msg=False))

    rows = []
    for f in fams:
        for t in periods:
            rows.append({
                "familia": f,
                "data": t,
                "demanda": dem.get((f, t), 0.0),
                "producao": pulp.value(prod[f][t]),
                "estoque": pulp.value(inv[f][t]),
                "atraso": pulp.value(backlog[f][t]),
            })
    out = pd.DataFrame(rows)
    save_csv(out, Path(paths["reports_tables"]) / "plano_agregado.csv")

    agg = out.groupby("data", as_index=False)[["demanda", "producao", "estoque"]].sum()
    plot_lines(agg, "data", ["demanda", "producao", "estoque"],
               "Demanda x Produção x Estoque (Agregado)",
               Path(paths["reports_figures"]) / "plano_agregado.png")
