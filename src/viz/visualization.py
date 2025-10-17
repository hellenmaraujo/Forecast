from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

def plot_lines(df: pd.DataFrame, x: str, ys: list[str], title: str, output: str | Path) -> str:
    fig, ax = plt.subplots(figsize=(10, 4))
    for y in ys:
        ax.plot(df[x], df[y], label=y)
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel("valor")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    output = str(output)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=120)
    plt.close(fig)
    return output
