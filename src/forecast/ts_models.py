import numpy as np
import pandas as pd
from typing import Optional

# ARIMA
try:
    import pmdarima as pm
    _HAS_PM = True
except Exception:
    _HAS_PM = False

from sklearn.svm import SVR as _SKSVR

def _as_series(y):
    if isinstance(y, (pd.Series, pd.DataFrame)):
        return y.squeeze()
    return pd.Series(y)

class ARIMARegressor:
    """
    Wrapper para forecast via ARIMA/SARIMA.
    Ignora X (features) e modela diretamente y como série temporal.
    predict(X) retorna um forecast de len(X) passos à frente.
    """
    def __init__(self,
                 auto: bool = True,
                 seasonal: bool = False,
                 m: int = 1,
                 order: Optional[tuple] = None,
                 seasonal_order: Optional[tuple] = None):
        self.auto = auto
        self.seasonal = seasonal
        self.m = m
        self.order = order
        self.seasonal_order = seasonal_order
        self._model = None
        self._y_index = None

    def fit(self, X, y):
        y = _as_series(y).astype(float)
        self._y_index = y.index

        if self.auto:
            if not _HAS_PM:
                raise ImportError(
                    "pmdarima não instalado. Instale com: pip install pmdarima"
                )
            self._model = pm.auto_arima(
                y,
                seasonal=self.seasonal,
                m=self.m,
                error_action="ignore",
                suppress_warnings=True,
                stepwise=True,
                trace=False
            )
        else:
            # fallback simples se quiser usar ordem fixa
            if self.order is None:
                self.order = (1, 1, 1)
            import statsmodels.api as sm
            self._model = sm.tsa.SARIMAX(
                y,
                order=self.order,
                seasonal_order=self.seasonal_order if self.seasonal else (0, 0, 0, 0),
                enforce_stationarity=False,
                enforce_invertibility=False
            ).fit(disp=False)
        return self

    def predict(self, X):
        n = len(X)
        if n <= 0:
            return np.array([], dtype=float)

        if _HAS_PM and hasattr(self._model, "predict"):  # pmdarima
            fc = self._model.predict(n_periods=n)
            return np.asarray(fc, dtype=float)

        # statsmodels SARIMAXResults
        fc = self._model.get_forecast(steps=n).predicted_mean
        return np.asarray(fc, dtype=float)

class SVRRegressor:
    """
    Wrapper direto para SVR da sklearn.
    Aqui X é usado (lags/rolling features devem vir do data_prep).
    """
    def __init__(self, kernel="rbf", C=10.0, gamma="scale"):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self._svr = _SKSVR(kernel=self.kernel, C=self.C, gamma=self.gamma)

    def fit(self, X, y):
        # Assegura que arrays estão 2D/1D como o sklearn espera
        if isinstance(X, pd.DataFrame):
            X_fit = X.values
        else:
            X_fit = np.asarray(X)
        y_fit = _as_series(y).astype(float).values
        self._svr.fit(X_fit, y_fit)
        return self

    def predict(self, X):
        Xp = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        return self._svr.predict(Xp).astype(float)
