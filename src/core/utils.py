from __future__ import annotations
import random, numpy as np
from contextlib import contextmanager
from time import perf_counter
from typing import Iterator

def set_seed(seed: int = 42) -> None:
    random.seed(seed); np.random.seed(seed)

@contextmanager
def timer(name: str):
    start = perf_counter()
    yield
    end = perf_counter()
    print(f"[TIMER] {name}: {end - start:.3f}s")
