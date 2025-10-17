from __future__ import annotations
from pathlib import Path
from typing import Dict, Any
import pandas as pd
from src.core.io import save_csv

def run_disaggregation(cfg: Dict[str, Any]) -> None:
    paths = cfg["project"]["paths"]
    plan_path = Path(paths["reports_tables"]) / "plano_agregado.csv"
    if not plan_path.exists():
        return
    plano = pd.read_csv(plan_path)
    if plano.empty:
        return
    proportions = cfg["disaggregation"]["proportions"]
    rows = []
    for r in plano.itertuples():
        fam = r.familia
        total = max(0.0, float(r.producao))
        for prod, weight in proportions.get(fam, {}).items():
            rows.append({
                "data": r.data,
                "familia": fam,
                "produto": prod,
                "producao": total * float(weight),
            })
    if not rows:
        return
    out = pd.DataFrame(rows)
    save_csv(out, Path(paths["reports_tables"]) / "plano_desagregado.csv")
