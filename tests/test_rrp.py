from __future__ import annotations

def test_rrp_outputs(rrp_summary, config):
    from pathlib import Path
    import pandas as pd
    rrp_path = Path(config["project"]["paths"]["reports_tables"]) / "rrp_resumo.csv"
    assert rrp_path.exists()
    rrp_df = pd.read_csv(rrp_path)
    assert not rrp_df.empty
    assert (rrp_df["utilizacao"] >= 0).all()
