from __future__ import annotations

def test_optimization_solution(optimized_plan, config):
    from pathlib import Path
    import pandas as pd
    plan_path = Path(config["project"]["paths"]["reports_tables"]) / "plano_agregado.csv"
    assert plan_path.exists()
    plan_df = pd.read_csv(plan_path)
    assert not plan_df.empty
    assert (plan_df[["producao", "estoque", "atraso"]] >= 0).all().all()
