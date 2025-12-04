#!/usr/bin/env python3
# ============================================================================
# Proyecto: Agentificación de Modelos ARIMA con Aprendizaje Reforzado
# Archivo: rl_agent.py
# Descripción: Agente DQN con Stable-Baselines3 para optimización ARIMA
# ============================================================================

import os
import argparse
import numpy as np
import yaml

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv

from src.arima_env import ARIMAHyperparamEnv
from src.data_processor import TimeSeriesProcessor


# ============================================================================
# CLASE PRINCIPAL DEL AGENTE
# ============================================================================

class ARIMAAgent:
    """
    Agente de Aprendizaje Reforzado para optimización de hiperparámetros ARIMA.
    Usa algoritmo DQN (Deep Q-Network) de Stable-Baselines3.
    """

    def __init__(self, train_data, val_data, config=None):
        """
        Inicializa el agente RL.

        Args:
            train_data: Datos de entrenamiento (numpy array)
            val_data:   Datos de validación (numpy array)
            config:     Dict con configuración combinada (rl_agent + environment)
        """
        self.train_data = np.asarray(train_data, dtype=float)
        self.val_data = np.asarray(val_data, dtype=float)

        # Configuración por defecto
        default_config = {
            # Entorno
            "p_max": 5,
            "d_max": 2,
            "q_max": 4,
            "max_steps": 50,
            "reward_weights": {
                "accuracy": 1.0,
                "aic": 0.3,
                "time": 0.1,
                "diagnostics": 0.2,
            },
            # RL
            "policy": "MlpPolicy",
            "total_timesteps": 50_000,
            "learning_rate": 0.00025,
            "buffer_size": 50000,
            "learning_starts": 1000,
            "batch_size": 64,
            "tau": 1.0,
            "gamma": 0.99,
            "train_freq": 4,
            "gradient_steps": 1,
            "target_update_interval": 2000,
            "exploration_fraction": 0.5,
            "exploration_initial_eps": 1.0,
            "exploration_final_eps": 0.02,
            "policy_kwargs": {
                "net_arch": [128, 128],   "activation_fn": "relu"
            },
            "device": "auto",
            "verbose": 1,
            "model_save_path": "models/arima_dqn_agent",
            "tensorboard_log": "models/tensorboard_logs",
            "checkpoint_freq": 2000,
        }

        # ============================
        # FUSIÓN SEGURA DE CONFIG
        # ============================

        # copiar defaults
        merged = default_config.copy()

        if config is not None:
            # sobreescribir defaults con config externo
            merged.update(config)

            # merge de policy_kwargs si existe
            if "policy_kwargs" in config:
                merged["policy_kwargs"].update(config["policy_kwargs"])

        # convertir activation_fn str → clase torch.nn
        act = merged["policy_kwargs"].get("activation_fn", None)
        if isinstance(act, str):
            import torch.nn as nn
            act_map = {
                "relu": nn.ReLU,
                "tanh": nn.Tanh,
                "elu": nn.ELU,
                "leaky_relu": nn.LeakyReLU,
                "sigmoid": nn.Sigmoid,
            }
            if act not in act_map:
                raise ValueError(f"activation_fn '{act}' no reconocido.")
            merged["policy_kwargs"]["activation_fn"] = act_map[act]

        # asignar
        self.config = merged


        # Crear entorno base y entorno vectorizado
        self.env_base, self.vec_env = self._create_envs()

        # Modelo DQN (se inicializa en train() o load())
        self.model = None

    # ----------------------------------------------------------------------
    # Creación de entornos
    # ----------------------------------------------------------------------
    def _make_base_env(self):
        """
        Crea una instancia del entorno ARIMA (no vectorizado).
        """
        env = ARIMAHyperparamEnv(
            train_data=self.train_data,
            val_data=self.val_data,
            p_max=self.config["p_max"],
            d_max=self.config["d_max"],
            q_max=self.config["q_max"],
            max_steps=self.config["max_steps"],
            reward_weights=self.config.get("reward_weights", None),
        )
        return env

    def _create_envs(self):
        """
        Crea y configura el entorno base y su versión vectorizada.
        """
        # Entorno base para verificación
        env_base = self._make_base_env()

        try:
            check_env(env_base, warn=True)
            print("✅ Entorno verificado correctamente con Gymnasium")
        except Exception as e:
            print(f"⚠️ Advertencia en verificación de entorno: {e}")

        # Versión vectorizada (1 entorno) para SB3
        def make_env():
            return Monitor(self._make_base_env())

        vec_env = DummyVecEnv([make_env])

        return env_base, vec_env

    # ----------------------------------------------------------------------
    # Entrenamiento
    # ----------------------------------------------------------------------
    def train(
        self,
        total_timesteps=None,
        save_path=None,
        tensorboard_log=None,
        save_freq=None,
    ):
        """
        Entrena el agente DQN.

        Args:
            total_timesteps: Número total de timesteps de entrenamiento
            save_path:       Ruta base para guardar el modelo entrenado (sin .zip)
            tensorboard_log: Directorio para logs de TensorBoard
            save_freq:       Frecuencia de guardado de checkpoints
        """
        # Resolver parámetros desde config si no se pasan explícitos
        if total_timesteps is None:
            total_timesteps = self.config.get("total_timesteps", 50_000)
        if save_path is None:
            save_path = self.config.get("model_save_path", "models/arima_dqn_agent")
        if tensorboard_log is None:
            tensorboard_log = self.config.get(
                "tensorboard_log", "models/tensorboard_logs"
            )
        if save_freq is None:
            save_freq = self.config.get("checkpoint_freq", 5000)

        print("\n" + "=" * 80)
        print("🚀 INICIANDO ENTRENAMIENTO DEL AGENTE RL")
        print("=" * 80)

        print("\n⚙️  Configuración RL:")
        print(f"   Total timesteps: {total_timesteps}")
        print(f"   Learning rate:   {self.config['learning_rate']}")
        print(f"   Buffer size:     {self.config['buffer_size']}")
        print(f"   Batch size:      {self.config['batch_size']}")
        print(f"   Exploración:     frac={self.config['exploration_fraction']}, "
              f"eps_ini={self.config['exploration_initial_eps']}, "
              f"eps_fin={self.config['exploration_final_eps']}")
        print(f"   Red:             {self.config['policy_kwargs']['net_arch']}")

        # Crear directorios
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        os.makedirs(tensorboard_log, exist_ok=True)

        # Crear modelo DQN
        self.model = DQN(
            policy=self.config.get("policy", "MlpPolicy"),
            env=self.vec_env,
            learning_rate=self.config["learning_rate"],
            buffer_size=self.config["buffer_size"],
            learning_starts=self.config["learning_starts"],
            batch_size=self.config["batch_size"],
            tau=self.config["tau"],
            gamma=self.config["gamma"],
            train_freq=self.config["train_freq"],
            gradient_steps=self.config["gradient_steps"],
            target_update_interval=self.config["target_update_interval"],
            exploration_fraction=self.config["exploration_fraction"],
            exploration_initial_eps=self.config["exploration_initial_eps"],
            exploration_final_eps=self.config["exploration_final_eps"],
            policy_kwargs=self.config["policy_kwargs"],
            tensorboard_log=tensorboard_log,
            verbose=self.config.get("verbose", 1),
            device=self.config.get("device", "auto"),
        )

        print(f"\n✅ Modelo DQN creado (device={self.model.device})")

        # Callbacks: Checkpoints + Evaluación periódica
        checkpoint_callback = CheckpointCallback(
            save_freq=save_freq,
            save_path=os.path.dirname(save_path),
            name_prefix="arima_dqn_checkpoint",
        )

        eval_env = DummyVecEnv([lambda: Monitor(self._make_base_env())])
        eval_callback = EvalCallback(
            eval_env=eval_env,
            best_model_save_path=os.path.dirname(save_path),
            log_path=os.path.dirname(save_path),
            eval_freq=save_freq,
            deterministic=True,
            render=False,
        )

        callbacks = [checkpoint_callback, eval_callback]

        print("\n🎓 Entrenando agente...")
        print(f"   (Para ver métricas: tensorboard --logdir {tensorboard_log})")

        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            log_interval=100,
            progress_bar=True,
        )

        # Guardar modelo final
        self.model.save(save_path)
        print(f"\n✅ Modelo final guardado en: {save_path}.zip")

        print("\n" + "=" * 80)
        print("✅ ENTRENAMIENTO COMPLETADO")
        print("=" * 80)

        return self.model

    # ----------------------------------------------------------------------
    # Carga de modelo
    # ----------------------------------------------------------------------
    def load(self, model_path="models/arima_dqn_agent.zip"):
        """
        Carga un modelo previamente entrenado.

        Args:
            model_path: Ruta al modelo guardado (.zip)
        """
        if not os.path.exists(model_path):
                raise FileNotFoundError(f"Modelo no encontrado: {model_path}")
        self.model = DQN.load(model_path, env=self.vec_env)
        return self.model
    # ----------------------------------------------------------------------
    # Evaluación y predicción de mejor configuración
    # ----------------------------------------------------------------------
    def _run_episode_with_model(self, deterministic=True):
        """
        Ejecuta un episodio completo con el modelo actual y retorna la
        mejor configuración observada en ese episodio.
        """
        if self.model is None:
            raise ValueError(
                "Modelo no entrenado/cargado. Ejecute train() o load() primero."
            )

        eval_env = self._make_base_env()
        obs, _ = eval_env.reset()
        done = False

        while not done:
            action, _ = self.model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated

        best = eval_env.get_best_config()
        return best

    def predict_best_config(self, deterministic=True):
        """
        Obtiene la mejor configuración ARIMA propuesta por el agente,
        ejecutando un episodio completo de interacción.

        Returns:
            (p, d, q)
        """
        best = self._run_episode_with_model(deterministic=deterministic)
        config = best["config"]
        p, d, q = config

        print("\n🤖 Agente RL predice configuración óptima (en un episodio):")
        print(f"   (p, d, q) = ({p}, {d}, {q})")
        print(f"   AIC: {best['aic']:.2f}, RMSE: {best['rmse']:.4f}")

        return (p, d, q)

    def evaluate(self, n_episodes=10):
        """
        Evalúa el agente entrenado en múltiples episodios.

        Args:
            n_episodes: Número de episodios de evaluación

        Returns:
            dict con estadísticas: mean_reward, mean_aic, best_config, etc.
        """
        if self.model is None:
            raise ValueError("Modelo no entrenado/cargado. Ejecute train() o load().")

        print(f"\n📊 Evaluando agente en {n_episodes} episodios...")

        episode_rewards = []
        episode_aic = []
        episode_configs = []

        for ep in range(n_episodes):
            best = self._run_episode_with_model(deterministic=True)
            # Usamos -AIC como aproximación de "reward de episodio" (para reporting)
            episode_rewards.append(-best["aic"])
            episode_aic.append(best["aic"])
            episode_configs.append(best["config"])

        stats = {
            "mean_reward": float(np.mean(episode_rewards)),
            "std_reward": float(np.std(episode_rewards)),
            "mean_aic": float(np.mean(episode_aic)),
            "std_aic": float(np.std(episode_aic)),
            "best_aic": float(np.min(episode_aic)),
            "best_config": episode_configs[int(np.argmin(episode_aic))],
        }

        print("\n📈 Resultados de evaluación:")
        print(
            f"   Recompensa promedio (aprox -AIC): "
            f"{stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}"
        )
        print(
            f"   AIC promedio: {stats['mean_aic']:.2f} ± {stats['std_aic']:.2f}"
        )
        print(f"   Mejor AIC: {stats['best_aic']:.2f}")
        print(f"   Mejor configuración: {stats['best_config']}")

        return stats


# ============================================================================
# FUNCIONES AUXILIARES PARA CONFIG Y PIPELINE
# ============================================================================

def load_full_config(config_path: str):
    """
    Carga config YAML completa y devuelve:
        cfg     (dict completo)
        rl_cfg  (sección rl_agent)
        env_cfg (sección environment)
        data_cfg(sección data)
    """
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    rl_cfg = cfg.get("rl_agent", {})
    env_cfg = cfg.get("environment", {})
    data_cfg = cfg.get("data", {})

    return cfg, rl_cfg, env_cfg, data_cfg


def build_agent_from_config(data_path: str, config_path: str):
    """
    Construye el ARIMAAgent usando los datos y el config.yaml.
    """
    print("📂 Cargando configuración desde:", config_path)
    cfg, rl_cfg, env_cfg, data_cfg = load_full_config(config_path)

    # Cargar y preparar datos
    print("📂 Cargando datos...")
    processor = TimeSeriesProcessor(data_path)
    processor.load_data()

    # Preprocesamiento según config
    handle_missing = data_cfg.get("handle_missing", "interpolate")
    outlier_threshold = data_cfg.get("outlier_threshold", 3.0)
    processor.preprocess(
        handle_missing=handle_missing,
        outlier_threshold=outlier_threshold,
    )

    processor.split_data()
    train_data, val_data, test_data = processor.get_train_val_test()
    print(
        f"✅ Datos preparados: {len(train_data)} train, "
        f"{len(val_data)} val, {len(test_data)} test"
    )

    # Combinar rl_agent + environment en un solo dict de config
    merged_config = {}
    merged_config.update(env_cfg)
    merged_config.update(rl_cfg)
    # Asegurar reward_weights dentro de config del agente
    if "reward_weights" in env_cfg:
        merged_config["reward_weights"] = env_cfg["reward_weights"]

    agent = ARIMAAgent(train_data, val_data, config=merged_config)
    return agent, cfg, rl_cfg, env_cfg, data_cfg


def train_agent_from_file(
    data_path="data/germany_monthly_power.csv",
    timesteps=None,
    output_dir="models",
    config_path="config/config.yaml",
):
    """
    Entrena agente RL desde archivo CSV usando config.yaml.

    Args:
        data_path:   Ruta al archivo de datos
        timesteps:   Número de timesteps de entrenamiento (si None, usa config)
        output_dir:  Directorio de salida para modelos
        config_path: Ruta al archivo de configuración YAML
    """
    agent, cfg, rl_cfg, env_cfg, data_cfg = build_agent_from_config(
        data_path, config_path
    )

    total_timesteps = timesteps if timesteps is not None else rl_cfg.get(
        "total_timesteps", 50_000
    )
    save_path = rl_cfg.get("model_save_path", os.path.join(output_dir, "arima_dqn_agent"))
    tensorboard_log = rl_cfg.get(
        "tensorboard_log", os.path.join(output_dir, "tensorboard_logs")
    )
    checkpoint_freq = rl_cfg.get("checkpoint_freq", 5000)

    agent.train(
        total_timesteps=total_timesteps,
        save_path=save_path,
        tensorboard_log=tensorboard_log,
        save_freq=checkpoint_freq,
    )

    # Evaluar agente
    print("\n🧪 Evaluando agente entrenado...")
    stats = agent.evaluate(n_episodes=5)

    # Guardar estadísticas
    os.makedirs(output_dir, exist_ok=True)
    stats_file = os.path.join(output_dir, "training_stats.txt")
    with open(stats_file, "w") as f:
        f.write("Estadísticas de Entrenamiento del Agente RL\n")
        f.write("=" * 60 + "\n\n")
        for key, value in stats.items():
            f.write(f"{key}: {value}\n")

    print(f"✅ Estadísticas guardadas en: {stats_file}")

    return agent


# ============================================================================
# INTERFAZ DE LÍNEA DE COMANDOS
# ============================================================================

def main():
    """
    Función principal para entrenamiento / evaluación desde línea de comandos.
    """
    parser = argparse.ArgumentParser(
        description="Entrenar/evaluar agente RL para optimización de hiperparámetros ARIMA"
    )

    parser.add_argument(
        "--train",
        action="store_true",
        help="Entrenar un nuevo agente",
    )

    parser.add_argument(
        "--data",
        type=str,
        default="data/germany_monthly_power.csv",
        help="Ruta al archivo de datos CSV mensual",
    )

    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Número de timesteps de entrenamiento (si None, usa config.yaml)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Directorio de salida para modelos y logs",
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Ruta al archivo de configuración YAML",
    )

    parser.add_argument(
        "--eval",
        action="store_true",
        help="Evaluar agente existente",
    )

    parser.add_argument(
        "--model-path",
        type=str,
        default="models/arima_dqn_agent.zip",
        help="Ruta al modelo para evaluación (.zip)",
    )

    args = parser.parse_args()

    if args.train:
        # Entrenar nuevo agente
        agent = train_agent_from_file(
            data_path=args.data,
            timesteps=args.timesteps,
            output_dir=args.output_dir,
            config_path=args.config,
        )

        best_config = agent.predict_best_config()
        print(f"\n🎯 Configuración recomendada por el agente: (p, d, q) = {best_config}")

    elif args.eval:
        # Evaluar agente existente
        print("📂 Cargando agente y datos para evaluación...")
        agent, cfg, rl_cfg, env_cfg, data_cfg = build_agent_from_config(
            args.data, args.config
        )
        agent.load(args.model_path)

        stats = agent.evaluate(n_episodes=10)
        best_config = agent.predict_best_config()
        print(f"\n🎯 Configuración recomendada: (p, d, q) = {best_config}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
