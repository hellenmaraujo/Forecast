from __future__ import annotations
import logging, logging.config, yaml
from functools import lru_cache
from pathlib import Path

DEFAULT_LOGGING_CONFIG = Path("configs/logging.yaml")

@lru_cache(maxsize=1)
def configure_logging(config_path: str | Path | None = None) -> None:
    cfg = Path(config_path) if config_path else DEFAULT_LOGGING_CONFIG
    if not cfg.exists():
        logging.basicConfig(level=logging.INFO,
                            format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")
        return
    with cfg.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    # cria pasta do arquivo de log se necessário
    file_handler = data.get("handlers", {}).get("file", {})
    if "filename" in file_handler:
        Path(file_handler["filename"]).parent.mkdir(parents=True, exist_ok=True)
    logging.config.dictConfig(data)

def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
