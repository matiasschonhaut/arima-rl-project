# ============================================================================
# Proyecto: Agentificación de Modelos ARIMA con Aprendizaje Reforzado
# Archivo: __init__.py
# Descripción: Inicialización del paquete src
# ============================================================================

__version__ = "1.0.0"
__author__ = "ARIMA-RL Project"
__description__ = "Optimización de hiperparámetros ARIMA mediante Aprendizaje Reforzado"

from .data_processor import TimeSeriesProcessor
from .arima_env import ARIMAHyperparamEnv, make_arima_env
from .rl_agent import ARIMAAgent

# Nuevas importaciones agregadas
from .arima_utils import (
    ARIMAModel,
    compare_models,
    grid_search_arima
)

__all__ = [
    'TimeSeriesProcessor',
    'ARIMAHyperparamEnv',
    'make_arima_env',
    'ARIMAAgent',
    'ARIMAModel',
    'compare_models',
    'grid_search_arima'
]

