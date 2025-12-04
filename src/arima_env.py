#!/usr/bin/env python3
# ============================================================================
# Proyecto: Agentificación de Modelos ARIMA con Aprendizaje Reforzado
# Archivo: arima_env.py
# Descripción: Entorno Gymnasium personalizado para optimización de
#              hiperparámetros ARIMA vía DQN (acción discreta)
# ============================================================================

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
import time

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class ARIMAHyperparamEnv(gym.Env):
    """
    Entorno Gymnasium para optimización de hiperparámetros ARIMA mediante RL.
    
    Espacio de Estados (8D, según diseño del proyecto):
        0: best_RMSE normalizado por std de los datos
        1: best_AIC normalizado
        2: p normalizado
        3: d normalizado
        4: q normalizado
        5: paso actual normalizado (progreso del episodio)
        6: best_AIC normalizado (otra vez como referencia global)
        7: número de parámetros (p+q+1) normalizado (coef)
        
    Espacio de Acciones:
        - Discreto(N), donde cada entero representa una tupla (p, d, q)
        - p: [0, p_max], d: [0, d_max], q: [0, q_max]
        
    Función de Recompensa:
        - Multiobjetivo: balancea precisión, complejidad y eficiencia
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        train_data,
        val_data,
        p_max=5,
        d_max=2,
        q_max=4,
        max_steps=50,
        reward_weights=None,
    ):
        """
        Inicializa el entorno ARIMA.
        
        Args:
            train_data: Array numpy con datos de entrenamiento
            val_data: Array numpy con datos de validación
            p_max: Máximo orden autorregresivo
            d_max: Máximo orden de diferenciación
            q_max: Máximo orden de media móvil
            max_steps: Máximo de pasos por episodio
            reward_weights: Dict con pesos para componentes de recompensa
        """
        super().__init__()

        self.train_data = np.asarray(train_data, dtype=float)
        self.val_data = np.asarray(val_data, dtype=float)
        self.p_max = p_max
        self.d_max = d_max
        self.q_max = q_max
        self.max_steps = max_steps

        # Pesos de la función de recompensa
        if reward_weights is None:
            self.reward_weights = {
                "accuracy": 1.0,
                "aic": 0.3,
                "time": 0.1,
                "diagnostics": 0.2,
            }
        else:
            self.reward_weights = reward_weights

        # Espacio de acción: Discreto, codificando (p,d,q)
        self.num_actions = (p_max + 1) * (d_max + 1) * (q_max + 1)
        self.action_space = spaces.Discrete(self.num_actions)

        # Espacio de observación (8 características)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32
        )

        # Variables de estado
        self.current_step = 0
        self.current_config = None
        self.best_aic = np.inf
        self.best_rmse = np.inf
        self.history = []

        # Normalización robusta (evitar divisiones por cero o signos raros)
        self.data_mean = float(np.mean(self.train_data))
        self.data_std = float(np.std(self.train_data))

        if abs(self.data_mean) < 1e-8:
            self.data_mean = 1.0
        if self.data_std < 1e-8:
            self.data_std = 1.0

    # ---------------------------------------------------------------------
    # Codificación / Decodificación de acciones
    # ---------------------------------------------------------------------
    def decode_action(self, action_id: int):
        """
        Convierte un entero del espacio Discrete en la tupla (p,d,q).
        """
        p = action_id // ((self.d_max + 1) * (self.q_max + 1))
        d = (action_id // (self.q_max + 1)) % (self.d_max + 1)
        q = action_id % (self.q_max + 1)
        return int(p), int(d), int(q)

    def encode_action(self, p: int, d: int, q: int) -> int:
        """
        Convierte una tupla (p,d,q) en un entero del espacio Discrete.
        """
        return int(
            p * (self.d_max + 1) * (self.q_max + 1)
            + d * (self.q_max + 1)
            + q
        )

    # ---------------------------------------------------------------------
    # API Gymnasium
    # ---------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        """
        Reinicia el entorno al estado inicial.
        
        Returns:
            observation: Estado inicial
            info: Información adicional
        """
        super().reset(seed=seed)

        self.current_step = 0
        self.current_config = [1, 1, 1]  # Configuración inicial por defecto
        self.best_aic = np.inf
        self.best_rmse = np.inf
        self.history = []

        # Estado inicial
        observation = self._get_observation()
        info = {}

        return observation, info

    def step(self, action):
        """
        Ejecuta una acción (entrenar ARIMA con configuración específica).
        
        Args:
            action: entero en [0, num_actions-1] que codifica (p, d, q)
            
        Returns:
            observation: Nuevo estado
            reward: Recompensa obtenida
            terminated: Si el episodio terminó
            truncated: Si el episodio fue truncado
            info: Información adicional
        """
        self.current_step += 1

        # Extraer configuración de la acción
        p, d, q = self.decode_action(int(action))
        self.current_config = [p, d, q]

        # Entrenar modelo ARIMA y obtener métricas
        try:
            metrics = self._train_and_evaluate_arima(p, d, q)

            # Calcular recompensa
            reward = self._compute_reward(metrics)

            # Actualizar mejor configuración
            if metrics["aic"] < self.best_aic:
                self.best_aic = metrics["aic"]
            if metrics["rmse"] < self.best_rmse:
                self.best_rmse = metrics["rmse"]

            # Guardar en historial
            self.history.append(
                {
                    "step": self.current_step,
                    "config": (p, d, q),
                    "metrics": metrics,
                    "reward": reward,
                }
            )

            success = True

        except Exception as e:
            # Si el modelo falla, penalizar fuertemente
            metrics = {
                "rmse": 1e6,
                "mae": 1e6,
                "mape": 1e6,
                "aic": 1e6,
                "bic": 1e6,
                "training_time": 0,
                "residuals_mean": 0.0,
                "residuals_std": 0.0,
                "n_params": p + q + 1,
                "failed": True,
                "error": str(e),
            }
            reward = -10.0
            success = False

        # Nuevo estado
        observation = self._get_observation()

        # Condiciones de terminación
        terminated = False
        truncated = self.current_step >= self.max_steps

        # Información adicional
        info = {
            "step": self.current_step,
            "config": (p, d, q),
            "metrics": metrics,
            "reward": reward,
            "success": success,
            "best_aic": self.best_aic,
            "best_rmse": self.best_rmse,
        }

        return observation, reward, terminated, truncated, info

    # ---------------------------------------------------------------------
    # Lógica interna
    # ---------------------------------------------------------------------
    def _train_and_evaluate_arima(self, p, d, q):
        """
        Entrena modelo ARIMA y calcula métricas de evaluación.
        
        Args:
            p, d, q: Hiperparámetros ARIMA
            
        Returns:
            dict: Métricas de desempeño
        """
        start_time = time.time()
        
        # Entrenar modelo
        model = ARIMA(self.train_data, order=(p, d, q))
        fitted_model = model.fit()
        
        training_time = time.time() - start_time
        
        # Predicciones en validación
        forecast = fitted_model.forecast(steps=len(self.val_data))
        
        # Métricas de precisión
        rmse = np.sqrt(mean_squared_error(self.val_data, forecast))
        mae = mean_absolute_error(self.val_data, forecast)

        # Evitar división por cero en MAPE
        denom = np.where(np.abs(self.val_data) < 1e-8, 1e-8, np.abs(self.val_data))
        mape = np.mean(np.abs((self.val_data - forecast) / denom)) * 100.0  # en porcentaje
        
        # Métricas de complejidad
        aic = fitted_model.aic
        bic = fitted_model.bic
        
        # Diagnóstico de residuos
        residuals = fitted_model.resid
        residuals_mean = float(np.mean(residuals))
        residuals_std = float(np.std(residuals))
        
        # Ljung-Box para autocorrelación de residuos
        try:
            if len(residuals) > 5:
                lag = min(10, len(residuals) - 1)
                lb_stat, lb_pvalue = acorr_ljungbox(residuals, lags=[lag], return_df=False)
                ljung_pvalue = float(lb_pvalue[-1])
            else:
                # Muy pocos residuos: asumir "no evidencia" de autocorrelación
                ljung_pvalue = 1.0
        except Exception:
            # Si algo falla en el test, no penalizamos
            ljung_pvalue = 1.0
        
        metrics = {
            'rmse': rmse,
            'mae': mae,
            'mape': mape,  # porcentaje
            'aic': aic,
            'bic': bic,
            'training_time': training_time,
            'residuals_mean': residuals_mean,
            'residuals_std': residuals_std,
            'ljung_pvalue': ljung_pvalue,
            'n_params': p + q + 1,
            'failed': False
        }
        
        return metrics

    def _compute_reward(self, metrics):
        """
        Calcula recompensa multiobjetivo según diseño del reporte:
        
        rt = w_acc * r_accuracy + w_aic * r_AIC + w_time * r_time + w_diag * r_diagnostics
        
        donde:
          r_accuracy = -(0.5 * RMSE/σY + 0.3 * MAE/μY + 0.2 * MAPE)
          r_AIC      = -0.1 * AIC/1000
          r_time     = -0.01 * (t_train / t_max)
          r_diag     = 0.5 * [I(|media_residuos| < 0.1) + I(p_Ljung > 0.05)]
        """
        if metrics.get('failed', False):
            return -10.0
        
        # Constantes del reporte
        alpha1, alpha2, alpha3 = 0.5, 0.3, 0.2
        beta = 0.1
        gamma = 0.01
        delta = 0.5
        t_max = 10.0  # tiempo máximo razonable para normalizar (segundos)
        
        # --- 1) Precisión de predicción (r_accuracy) ---
        rmse_norm = metrics['rmse'] / self.data_std
        mae_norm = metrics['mae'] / abs(self.data_mean)
        mape_norm = metrics['mape'] / 100.0  # pasar de % a [0,1]
        
        r_accuracy = -(alpha1 * rmse_norm + alpha2 * mae_norm + alpha3 * mape_norm)
        
        # --- 2) Complejidad del modelo (r_AIC) ---
        aic_norm = metrics['aic'] / 1000.0
        r_aic = -beta * aic_norm
        
        # --- 3) Eficiencia computacional (r_time) ---
        time_norm = metrics['training_time'] / t_max
        r_time = -gamma * time_norm
        
        # --- 4) Calidad de residuos (r_diagnostics) ---
        residuals_mean = metrics.get('residuals_mean', 0.0)
        ljung_pvalue = metrics.get('ljung_pvalue', 1.0)
        
        series_std = np.std(self.train_data)
        cond_mean_ok = abs(residuals_mean) < 0.25 * series_std   # criterio robusto
        cond_ljung_ok = ljung_pvalue > 0.05
        
        r_diagnostics = delta * (int(cond_mean_ok) + int(cond_ljung_ok))
        
        # --- Combinación con pesos w_acc, w_aic, w_time, w_diag ---
        reward = (
            self.reward_weights['accuracy'] * r_accuracy +
            self.reward_weights['aic'] * r_aic +
            self.reward_weights['time'] * r_time +
            self.reward_weights['diagnostics'] * r_diagnostics
        )
        
        # Bonus por mejorar el mejor AIC y RMSE del episodio
        if metrics['aic'] < self.best_aic:
            reward += 1.0
        if metrics['rmse'] < self.best_rmse:
            reward += 0.5
        
        return reward

    def _get_observation(self):
        """
        Construye el vector de observación (estado).
        
        Returns:
            np.array: Vector de estado (8 características)
        """
        # Si aún no hay configuración, usar algo neutro
        if self.current_config is None:
            p, d, q = 1, 1, 1
        else:
            p, d, q = self.current_config

        # Best RMSE normalizado (si no hay aún, usar un valor grande fijo)
        if np.isinf(self.best_rmse):
            rmse_norm = 10.0
        else:
            rmse_norm = self.best_rmse / self.data_std

        # Best AIC normalizado (si no hay aún, usar valor grande)
        if np.isinf(self.best_aic):
            best_aic_norm = 10.0
        else:
            best_aic_norm = self.best_aic / 1000.0

        # p, d, q normalizados
        p_norm = p / max(1, self.p_max)
        d_norm = d / max(1, self.d_max)
        q_norm = q / max(1, self.q_max)

        # Progreso del episodio
        step_norm = self.current_step / max(1, self.max_steps)

        # Número de parámetros (coeficientes del modelo ARIMA): p + q + 1 (sigma^2)
        n_params = p + q + 1
        n_params_max = self.p_max + self.q_max + 1
        coef_norm = n_params / max(1, n_params_max)

        observation = np.array(
            [
                rmse_norm,      # 0
                best_aic_norm,  # 1
                p_norm,         # 2
                d_norm,         # 3
                q_norm,         # 4
                step_norm,      # 5
                best_aic_norm,  # 6 (referencia global de AIC)
                coef_norm,      # 7
            ],
            dtype=np.float32,
        )

        return observation

    # ---------------------------------------------------------------------
    # Utilidades
    # ---------------------------------------------------------------------
    def render(self, mode="human"):
        """
        Renderiza el estado actual del entorno (imprime información).
        """
        if mode == "human":
            print(f"\n{'=' * 60}")
            print(f"Step: {self.current_step}/{self.max_steps}")
            if self.current_config is not None:
                print(
                    f"Current Config: p={self.current_config[0]}, "
                    f"d={self.current_config[1]}, q={self.current_config[2]}"
                )
            print(f"Best AIC: {self.best_aic:.2f}")
            print(f"Best RMSE: {self.best_rmse:.2f}")
            if len(self.history) > 0:
                last = self.history[-1]
                print(f"Last Reward: {last['reward']:.4f}")
            print(f"{'=' * 60}")

    def get_best_config(self):
        """
        Retorna la mejor configuración encontrada hasta ahora.
        
        Returns:
            dict: {config, aic, rmse, mae, step}
        """
        if len(self.history) == 0:
            return None

        # Encontrar configuración con mejor AIC
        best_entry = min(self.history, key=lambda x: x["metrics"]["aic"])

        return {
            "config": best_entry["config"],
            "aic": best_entry["metrics"]["aic"],
            "rmse": best_entry["metrics"]["rmse"],
            "mae": best_entry["metrics"]["mae"],
            "step": best_entry["step"],
        }


# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================
def make_arima_env(train_data, val_data, **kwargs):
    """
    Función factory para crear el entorno ARIMA.
    """
    return ARIMAHyperparamEnv(train_data, val_data, **kwargs)


def test_env():
    """
    Función de prueba del entorno con configuraciones aleatorias.
    """
    print("🧪 Probando entorno ARIMA...")

    # Datos sintéticos
    np.random.seed(42)
    train_data = np.random.randn(48) * 5 + 50
    val_data = np.random.randn(6) * 5 + 50

    # Crear entorno
    env = ARIMAHyperparamEnv(train_data, val_data)

    print("✅ Entorno creado")
    print(f"   Espacio de acciones: {env.action_space}")
    print(f"   Espacio de observación: {env.observation_space}")

    # Ejecutar episodio de prueba
    observation, info = env.reset()
    print(f"\n✅ Estado inicial: {observation}")

    for step in range(5):
        # Acción aleatoria (entero)
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        p, d, q = env.decode_action(action)

        print(f"\n   Step {step + 1}:")
        print(f"   Acción (id={action}): p={p}, d={d}, q={q}")
        print(f"   Recompensa: {reward:.4f}")
        print(f"   AIC: {info['metrics']['aic']:.2f}")

        if terminated or truncated:
            break

    # Mejor configuración
    best = env.get_best_config()
    print("\n✅ Mejor configuración encontrada:")
    print(f"   (p, d, q) = {best['config']}")
    print(f"   AIC: {best['aic']:.2f}")
    print(f"   RMSE: {best['rmse']:.2f}")

    print("\n✅ Prueba completada!")


if __name__ == "__main__":
    test_env()
