from __future__ import annotations
import numpy as np, pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Any
from dateutil.relativedelta import relativedelta
from src.core.io import ensure_directory, save_csv
from src.core.logging_utils import get_logger

log = get_logger(__name__)

def _generate_synthetic(cfg: Dict[str, Any]) -> pd.DataFrame:
    p = cfg["data_prep"]["synthetic"]
    start = pd.Period(p["start"], freq=cfg["data_prep"]["frequency"]).to_timestamp()
    periods = int(p["periods"])
    freq = cfg["data_prep"]["frequency"]
    idx = pd.date_range(start, periods=periods, freq=freq)
    rows = []
    saz_amp = float(p["sazonalidade_amplitude"])
    tend = float(p["tendencia"])
    noise = float(p["ruido"])
    rng = np.random.default_rng(cfg["project"]["seed"])
    for fam in p["familias"]:
        for k in range(int(p["produtos_por_familia"])):
            base = 100 + np.arange(periods) * tend * 100
            saz = (1 + saz_amp * np.sin(np.linspace(0, 2*np.pi, periods)))
            serie = base * saz * (1 + noise * rng.standard_normal(periods))
            rows += [{
                "data": idx[i].date().isoformat(),
                "familia": fam,
                "produto": f"{fam}_prod_{k+1}",
                "quantidade": max(0.0, float(serie[i]))
            } for i in range(periods)]
    return pd.DataFrame(rows)

def load_or_generate(cfg: Dict[str, Any]) -> pd.DataFrame:
    raw_dir = Path(cfg["project"]["paths"]["data_raw"])
    csv = raw_dir / "vendas_historicas.csv"
    if csv.exists():
        df = pd.read_csv(csv)
        log.info("Carregou dados reais: %s", csv)
    else:
        log.warning("Arquivo real não encontrado. Gerando dataset sintético.")
        ensure_directory(raw_dir)
        df = _generate_synthetic(cfg)
        save_csv(df, csv)
    return df

def clean_and_split(df: pd.DataFrame, cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    c = cfg["data_prep"]
    for col in [c["date_column"], c["family_column"], c["product_column"], c["target_column"]]:
        if col not in df.columns:
            raise ValueError(f"Coluna ausente: {col}")
    df = df.copy()
    df[c["date_column"]] = pd.to_datetime(df[c["date_column"]])
    df.sort_values(by=[c["family_column"], c["product_column"], c["date_column"]], inplace=True)
    freq = c["frequency"]
    last_date = df[c["date_column"]].max()
    test_end = last_date
    test_start = (test_end.to_period(freq).to_timestamp() - relativedelta(months=c["test_size_months"]-1)).normalize()
    train = df[df[c["date_column"]] < test_start].copy()
    test  = df[df[c["date_column"]] >= test_start].copy()
    return train, test

def run_data_preparation(cfg: Dict[str, Any]) -> None:
    paths = cfg["project"]["paths"]
    ensure_directory(paths["data_interim"])
    df = load_or_generate(cfg)
    train, test = clean_and_split(df, cfg)
    save_csv(df, Path(paths["data_interim"]) / "prepared_data.csv")
    save_csv(train, Path(paths["data_interim"]) / "train_raw.csv")
    save_csv(test, Path(paths["data_interim"]) / "test_raw.csv")
    log.info("Data prep ok: train=%d, test=%d", len(train), len(test))
