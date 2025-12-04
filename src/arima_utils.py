#!/usr/bin/env python3
# ============================================================================
# Proyecto: Agentificación de Modelos ARIMA con Aprendizaje Reforzado
# Archivo: arima_utils.py
# Descripción: Utilidades para ajuste y comparación de modelos ARIMA
# ============================================================================

import time
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# ============================================================================
# CLASE ARIMAModel
# ============================================================================

@dataclass
class ARIMAModel:
    """
    Wrapper ligero sobre statsmodels.ARIMA para facilitar comparación de modelos.

    Atributos principales:
        order: tupla (p, d, q)
        aic, bic: criterios de información
        rmse, mae: métricas de error sobre el conjunto de validación
        training_time: tiempo de ajuste del modelo
        failed: indica si el ajuste falló
    """
    order: Tuple[int, int, int]
    aic: float
    bic: float
    rmse: float
    mae: float
    training_time: float
    n_params: int
    failed: bool = False
    name: Optional[str] = None

    # Estos campos no se incluyen en el dataclass por simplicidad, pero se guardan aparte
    fitted_model: object = None
    forecast: Optional[np.ndarray] = None
    conf_int: Optional[np.ndarray] = None

    def to_dict(self) -> Dict:
        d = asdict(self)
        # quitar objetos grandes que no se pueden serializar
        d.pop("fitted_model", None)
        d.pop("forecast", None)
        d.pop("conf_int", None)
        return d


def fit_arima_model(
    train_series: pd.Series,
    val_series: Optional[pd.Series] = None,
    order: Tuple[int, int, int] = (1, 1, 1),
    alpha: float = 0.05,
    max_forecast_steps: Optional[int] = None,
) -> ARIMAModel:
    """
    Ajusta un modelo ARIMA(train_series, order) y evalúa sobre val_series (si se entrega).

    Args:
        train_series: Serie de entrenamiento (índice temporal o entero).
        val_series: Serie de validación. Si es None, sólo se ajusta el modelo.
        order: tupla (p, d, q).
        alpha: nivel para intervalos de confianza (por ejemplo 0.05 → 95%).
        max_forecast_steps: limitar pasos de forecast (por defecto len(val_series)).

    Returns:
        ARIMAModel con métricas y modelo ajustado.
    """
    p, d, q = order
    start_time = time.time()

    try:
        model = ARIMA(train_series, order=order)
        fitted = model.fit()

        training_time = time.time() - start_time
        aic = float(fitted.aic)
        bic = float(fitted.bic)
        n_params = p + q + 1  # aproximación simple

        rmse = np.nan
        mae = np.nan
        forecast = None
        conf_int = None

        if val_series is not None and len(val_series) > 0:
            steps = len(val_series)
            if max_forecast_steps is not None:
                steps = min(steps, max_forecast_steps)

            res = fitted.get_forecast(steps=steps)
            forecast = res.predicted_mean.values
            conf_int = res.conf_int(alpha=alpha).values

            rmse = float(np.sqrt(mean_squared_error(val_series.values[:steps], forecast)))
            mae = float(mean_absolute_error(val_series.values[:steps], forecast))

        result = ARIMAModel(
            order=order,
            aic=aic,
            bic=bic,
            rmse=rmse,
            mae=mae,
            training_time=training_time,
            n_params=n_params,
            failed=False,
            name=f"ARIMA{order}",
        )
        result.fitted_model = fitted
        result.forecast = forecast
        result.conf_int = conf_int

        return result

    except Exception as e:
        # Modelo falló: devolvemos ARIMAModel "malo" para poder compararlo
        training_time = time.time() - start_time
        print(f"⚠️  Error al ajustar ARIMA{order}: {e}")

        return ARIMAModel(
            order=order,
            aic=1e9,
            bic=1e9,
            rmse=1e6,
            mae=1e6,
            training_time=training_time,
            n_params=p + q + 1,
            failed=True,
            name=f"ARIMA{order}",
        )


# ============================================================================
# COMPARACIÓN DE MODELOS
# ============================================================================

def compare_models(
    train_series: pd.Series,
    val_series: pd.Series,
    configs: List[Tuple[int, int, int]],
    alpha: float = 0.05,
) -> Tuple[List[ARIMAModel], pd.DataFrame]:
    """
    Compara una lista de modelos ARIMA sobre la misma serie train/val.

    Args:
        train_series: Serie de entrenamiento.
        val_series: Serie de validación.
        configs: Lista de tuplas (p, d, q) a evaluar. Deben ser ≥ 3 según el PDF.
        alpha: Nivel para intervalos de confianza.

    Returns:
        (models_ordenados, df_resultados)
    """
    assert len(configs) >= 3, "Se requieren al menos 3 configuraciones (p,d,q) para la comparación."

    print("\n" + "=" * 80)
    print("📊 COMPARACIÓN DE MODELOS ARIMA (criterio AIC)")
    print("=" * 80)

    models: List[ARIMAModel] = []

    for order in configs:
        print(f"\n🔧 Ajustando ARIMA{order}...")
        m = fit_arima_model(train_series, val_series, order=order, alpha=alpha)
        models.append(m)
        print(f"   → AIC={m.aic:.2f}, BIC={m.bic:.2f}, RMSE={m.rmse:.2f}, MAE={m.mae:.2f}, "
              f"tiempo={m.training_time:.2f}s, failed={m.failed}")

    # Ordenar por AIC ascendente (mejor primero)
    models_sorted = sorted(models, key=lambda x: x.aic)
    best = models_sorted[0]

    print("\n✅ Mejor modelo según AIC:")
    print(f"   ARIMA{best.order} con AIC={best.aic:.2f}, RMSE={best.rmse:.2f}")

    # DataFrame resumen
    rows = [m.to_dict() for m in models_sorted]
    df_results = pd.DataFrame(rows)
    df_results["order_str"] = df_results["order"].apply(lambda t: f"ARIMA{tuple(t)}")
    df_results = df_results[
        ["order_str", "order", "aic", "bic", "rmse", "mae", "n_params", "training_time", "failed"]
    ]

    return models_sorted, df_results


# ============================================================================
# GRID SEARCH (BÚSQUEDA EXHAUSTIVA)
# ============================================================================

def grid_search_arima(
    train_series: pd.Series,
    val_series: pd.Series,
    p_range: Tuple[int, int] = (0, 5),
    d_range: Tuple[int, int] = (0, 2),
    q_range: Tuple[int, int] = (0, 4),
    alpha: float = 0.05,
    max_models: Optional[int] = None,
) -> Tuple[ARIMAModel, pd.DataFrame]:
    """
    Búsqueda exhaustiva simple sobre rangos de p,d,q, seleccionando el mejor según AIC.

    Args:
        train_series: Serie de entrenamiento.
        val_series: Serie de validación.
        p_range: (p_min, p_max)
        d_range: (d_min, d_max)
        q_range: (q_min, q_max)
        alpha: nivel para intervalos de confianza.
        max_models: límite opcional de modelos a evaluar (por si el espacio es muy grande).

    Returns:
        (mejor_modelo, df_resultados_completo)
    """
    p_min, p_max = p_range
    d_min, d_max = d_range
    q_min, q_max = q_range

    configs: List[Tuple[int, int, int]] = []
    for p in range(p_min, p_max + 1):
        for d in range(d_min, d_max + 1):
            for q in range(q_min, q_max + 1):
                configs.append((p, d, q))

    if max_models is not None:
        configs = configs[:max_models]

    print("\n" + "=" * 80)
    print("🔍 GRID SEARCH ARIMA (criterio AIC)")
    print("=" * 80)
    print(f"   Total de configuraciones a evaluar: {len(configs)}")

    models: List[ARIMAModel] = []

    for i, order in enumerate(configs, start=1):
        print(f"\n[{i}/{len(configs)}] Ajustando ARIMA{order}...")
        m = fit_arima_model(train_series, val_series, order=order, alpha=alpha)
        models.append(m)
        print(f"   → AIC={m.aic:.2f}, RMSE={m.rmse:.2f}, failed={m.failed}")

    # Ordenar por AIC y construir DataFrame
    models_sorted = sorted(models, key=lambda x: x.aic)
    best = models_sorted[0]

    print("\n✅ Mejor modelo encontrado por grid search:")
    print(f"   ARIMA{best.order} con AIC={best.aic:.2f}, RMSE={best.rmse:.2f}")

    rows = [m.to_dict() for m in models_sorted]
    df_results = pd.DataFrame(rows)
    df_results["order_str"] = df_results["order"].apply(lambda t: f"ARIMA{tuple(t)}")
    df_results = df_results[
        ["order_str", "order", "aic", "bic", "rmse", "mae", "n_params", "training_time", "failed"]
    ]

    return best, df_results

# ============================================================================
# TABLA PARA LA INTERFAZ STREAMLIT
# ============================================================================

def create_comparison_table(results) -> pd.DataFrame:
    """
    Recibe:
      - un DataFrame df_results
      - o la tupla (models_sorted, df_results)
    Devuelve la tabla final ordenada por AIC.
    """

    # Caso 1 — results es una tupla (salida estándar de compare_models)
    if isinstance(results, tuple):
        df_results = results[1].copy()

    # Caso 2 — results es directamente un DataFrame
    elif isinstance(results, pd.DataFrame):
        df_results = results.copy()

    else:
        raise ValueError("create_comparison_table recibe un tipo inválido.")

    if df_results.empty:
        raise ValueError("df_results está vacío.")

    # Agregar columna bonita
    if "order_str" not in df_results.columns:
        df_results["order_str"] = df_results["order"].apply(lambda t: f"ARIMA{tuple(t)}")

    # Ordenar por AIC
    df_results = df_results.sort_values("aic", ascending=True).reset_index(drop=True)

    return df_results


def forecast_with_intervals(model: ARIMAModel, steps: int = 10, alpha: float = 0.05):
    """
    Devuelve forecast + intervalo de confianza (% = 1-alpha).
    """
    if model.fitted_model is None:
        raise ValueError("Modelo no está ajustado. Llama .fit() primero.")

    res = model.fitted_model.get_forecast(steps=steps)
    forecast = res.predicted_mean.values
    conf_int = res.conf_int(alpha=alpha).values

    lower = conf_int[:, 0]
    upper = conf_int[:, 1]

    return forecast, lower, upper
