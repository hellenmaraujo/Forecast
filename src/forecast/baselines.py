import numpy as np
import pandas as pd

try:
    from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt
except Exception:
    SimpleExpSmoothing = None
    Holt = None


def _as_series(y):
    if isinstance(y, (pd.Series, pd.DataFrame)):
        return y.squeeze()
    return pd.Series(y)


class MovingAverageRegressor:
    """Baseline simples: prevê a última média móvel do y_train como constante."""
    def __init__(self, window=3):
        self.window = int(window)
        self.value_ = None

    def fit(self, X, y):
        y = _as_series(y).astype(float)
        w = max(1, min(self.window, len(y)))
        ma = y.rolling(window=w, min_periods=w).mean()
        self.value_ = ma.dropna().iloc[-1] if not ma.dropna().empty else float(y.mean())
        return self

    def predict(self, X):
        n = len(X)
        return np.full(shape=(n,), fill_value=self.value_, dtype=float)

class SESRegressor:
    """Suavização Exponencial Simples (statsmodels)"""
    def __init__(self, alpha=None, optimized=True):
        self.alpha = alpha
        self.optimized = optimized
        self._fit = None

    def fit(self, X, y):
        if SimpleExpSmoothing is None:
            raise ImportError("statsmodels é necessário para SES.")
        y = _as_series(y).astype(float)

        # regra: com poucas observações, use 'estimated'
        init_method = "heuristic" if len(y) >= 10 else "estimated"
        try:
            model = SimpleExpSmoothing(y, initialization_method=init_method)
            self._fit = model.fit(
                smoothing_level=self.alpha,
                optimized=self.optimized if self.alpha is None else False
            )
        except Exception:
            # fallback totalmente robusto
            model = SimpleExpSmoothing(y, initialization_method="estimated")
            self._fit = model.fit(optimized=True)
        return self

    def predict(self, X):
        n = len(X)
        fc = self._fit.forecast(n)
        return np.asarray(fc, dtype=float)


class HoltRegressor:
    """Holt linear (nível + tendência) com statsmodels"""
    def __init__(self, alpha=None, beta=None, optimized=True):
        self.alpha = alpha
        self.beta = beta
        self.optimized = optimized
        self._fit = None

    def fit(self, X, y):
        if Holt is None:
            raise ImportError("statsmodels é necessário para Holt.")
        y = _as_series(y).astype(float)

        # regra: com poucas observações, use 'estimated'
        init_method = "heuristic" if len(y) >= 10 else "estimated"
        try:
            model = Holt(y, initialization_method=init_method)
            self._fit = model.fit(
                smoothing_level=self.alpha,
                smoothing_trend=self.beta,
                optimized=self.optimized if (self.alpha is None or self.beta is None) else False
            )
        except Exception:
            # fallback: estima tudo automaticamente
            model = Holt(y, initialization_method="estimated")
            self._fit = model.fit(optimized=True)
        return self

    def predict(self, X):
        n = len(X)
        fc = self._fit.forecast(n)
        return np.asarray(fc, dtype=float)
