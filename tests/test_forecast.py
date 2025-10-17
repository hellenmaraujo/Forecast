from __future__ import annotations
from pathlib import Path

def test_training_metrics_saved(config, trained_models):
    import pandas as pd
    metrics_path = Path(config["project"]["paths"]["reports_tables"]) / "ranking_modelos.csv"
    assert metrics_path.exists()
    metrics_df = pd.read_csv(metrics_path)
    assert not metrics_df.empty


def test_forecast_output_shape(config, forecast_output):
    import pandas as pd
    forecast_path = Path(config["project"]["paths"]["reports_tables"]) / "demanda_prevista.csv"
    assert forecast_path.exists()
    forecast_df = pd.read_csv(forecast_path)
    assert len(forecast_df["data"].unique()) >= config["forecast"]["horizon_months"]
