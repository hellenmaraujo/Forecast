from __future__ import annotations
from pathlib import Path
from typing import Any, Dict
import json, yaml, pandas as pd

def ensure_directory(path: str | Path) -> Path:
    p = Path(path); p.mkdir(parents=True, exist_ok=True); return p

def load_config(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)

def save_csv(df: pd.DataFrame, path: str | Path) -> Path:
    p = Path(path); ensure_directory(p.parent); df.to_csv(p, index=False); return p

def save_json(obj: Dict[str, Any], path: str | Path) -> Path:
    p = Path(path); ensure_directory(p.parent)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    return p
