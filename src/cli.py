from __future__ import annotations
import argparse
from pathlib import Path
from src.core.io import load_config
from src.core.logging_utils import configure_logging
from src.core.utils import set_seed
from src.data_prep.data_preparation import run_data_preparation
from src.data_prep.feature_engineering import build_features, persist_features
from src.forecast.train_models import train_models
from src.forecast.evaluate import evaluate_models
from src.forecast.forecasting_pipeline import generate_forecast
from src.optimization.optimization_plan import run_optimization
from src.rrp.rrp_analysis import run_rrp_analysis
from src.disaggregation.disaggregation import run_disaggregation

def _build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    sub = p.add_subparsers(dest="command", required=True)
    for cmd in ["prepare-data","train","evaluate","forecast","optimize","rrp","disaggregate","all"]:
        sub.add_parser(cmd)
    return p

def cmd_prepare_data(cfg):
    run_data_preparation(cfg)
    tr, te, meta = build_features(cfg)
    persist_features(tr, te, meta, cfg)

def cmd_train(cfg): train_models(cfg)
def cmd_evaluate(cfg): evaluate_models(cfg)
def cmd_forecast(cfg): generate_forecast(cfg)
def cmd_optimize(cfg): run_optimization(cfg)
def cmd_rrp(cfg): run_rrp_analysis(cfg)
def cmd_disaggregate(cfg): run_disaggregation(cfg)
def cmd_all(cfg):
    cmd_prepare_data(cfg); cmd_train(cfg); cmd_evaluate(cfg)
    cmd_forecast(cfg); cmd_optimize(cfg); cmd_rrp(cfg); cmd_disaggregate(cfg)

def main():
    args = _build_parser().parse_args()
    cfg = load_config(Path(args.config))
    configure_logging(cfg["project"]["paths"]["logging_config"])
    set_seed(cfg["project"]["seed"])
    {
      "prepare-data": cmd_prepare_data,
      "train": cmd_train,
      "evaluate": cmd_evaluate,
      "forecast": cmd_forecast,
      "optimize": cmd_optimize,
      "rrp": cmd_rrp,
      "disaggregate": cmd_disaggregate,
      "all": cmd_all,
    }[args.command](cfg)

if __name__ == "__main__":
    main()
