#!/usr/bin/env python3
# ============================================================================
# Proyecto: Agentificación de Modelos ARIMA con Aprendizaje Reforzado
# Archivo: app_streamlit.py
# Descripción: Interfaz web con Streamlit para explorar datos, probar ARIMA y
#              utilizar el agente RL para sugerir hiperparámetros (p, d, q).
# ============================================================================

import os
from typing import Tuple, List

import numpy as np
import pandas as pd
import streamlit as st
import yaml

from src.data_processor import TimeSeriesProcessor
from src.rl_agent import ARIMAAgent
from src.arima_utils import (
    compare_models,
    fit_arima_model,   # función definida en arima_utils.py
)


# ============================================================================
# CARGA DE CONFIGURACIÓN Y DATOS
# ============================================================================

@st.cache_data
def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


@st.cache_data
def load_data_and_splits(config_path: str = "config/config.yaml"):
    """
    Carga config.yaml, prepara datos con TimeSeriesProcessor y retorna:
        cfg, df_completo, train_df, val_df, test_df
    """
    cfg = load_config(config_path)
    data_cfg = cfg.get("data", {})

    data_path = data_cfg.get("source", "data/germany_monthly_power.csv")
    handle_missing = data_cfg.get("handle_missing", "interpolate")
    outlier_thr = data_cfg.get("outlier_threshold", 3.0)

    processor = TimeSeriesProcessor(data_path)
    processor.load_data()
    processor.preprocess(handle_missing=handle_missing, outlier_threshold=outlier_thr)
    processor.split_data()

    df_full = processor.df.copy()
    train_df = processor.train.copy()
    val_df = processor.val.copy()
    test_df = processor.test.copy()

    return cfg, df_full, train_df, val_df, test_df


@st.cache_resource
def load_trained_agent(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    cfg: dict,
) -> Tuple[ARIMAAgent, bool]:
    """
    Crea un ARIMAAgent con la config (rl_agent + environment) y, si existe,
    carga el modelo entrenado desde disco.
    """
    train_values = train_df["value"].values.astype(float)
    val_values = val_df["value"].values.astype(float)

    rl_cfg = cfg.get("rl_agent", {})
    env_cfg = cfg.get("environment", {})

    merged = {}
    merged.update(env_cfg)
    merged.update(rl_cfg)
    if "reward_weights" in env_cfg:
        merged["reward_weights"] = env_cfg["reward_weights"]

    agent = ARIMAAgent(train_values, val_values, config=merged)

    model_base = rl_cfg.get("model_save_path", "models/arima_dqn_agent")
    model_path = model_base if model_base.endswith(".zip") else model_base + ".zip"

    has_model = os.path.exists(model_path)

    if has_model:
        agent.load(model_path)

    return agent, has_model


# ============================================================================
# FUNCIONES AUXILIARES PARA LA APP
# ============================================================================

def make_forecast_with_order(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    order: Tuple[int, int, int],
    alpha: float = 0.05,
):
    """
    Ajusta ARIMA con orden dado sobre train + val y pronostica sobre test.
    Retorna:
        modelo (ARIMAModel),
        forecast_index (Index),
        forecast (np.ndarray),
        conf_int (np.ndarray),
        metrics (dict con RMSE, MAE, AIC, BIC)
    """
    # Entrenamos el modelo usando train+val para luego testear en test
    full_train = pd.concat([train_df, val_df])
    full_train_series = full_train["value"]
    test_series = test_df["value"]

    model = fit_arima_model(full_train_series, test_series, order=order, alpha=alpha)

    forecast = model.forecast
    conf_int = model.conf_int
    forecast_index = test_series.index[: len(forecast)] if forecast is not None else None

    metrics = {
        "AIC": model.aic,
        "BIC": model.bic,
        "RMSE": model.rmse,
        "MAE": model.mae,
    }

    return model, forecast_index, forecast, conf_int, metrics


def plot_series_with_split(df_full, train_df, val_df, test_df):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df_full.index, df_full["value"], label="Serie completa", alpha=0.3)

    ax.plot(train_df.index, train_df["value"], label="Train", linewidth=2)
    ax.plot(val_df.index, val_df["value"], label="Val", linewidth=2)
    ax.plot(test_df.index, test_df["value"], label="Test", linewidth=2)

    ax.set_title("Serie mensual de consumo eléctrico (GWh)")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Consumo (unidades normalizadas)")
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)


def plot_forecast(test_df, forecast_index, forecast, conf_int=None, title="Forecast"):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(test_df.index, test_df["value"], label="Real (test)", linewidth=2)

    if forecast is not None and forecast_index is not None:
        ax.plot(forecast_index, forecast, label="Pronóstico ARIMA", linewidth=2)

        if conf_int is not None and len(conf_int) == len(forecast):
            ax.fill_between(
                forecast_index,
                conf_int[:, 0],
                conf_int[:, 1],
                alpha=0.2,
                label="Intervalo de confianza",
            )

    ax.set_title(title)
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Consumo")
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)


# ============================================================================
# APLICACIÓN STREAMLIT
# ============================================================================

def main():
    st.set_page_config(
        page_title="ARIMA + RL Dashboard",
        layout="wide",
    )

    st.title("⚡ Agentificación de Modelos ARIMA vía Aprendizaje Reforzado")
    st.write(
        "Esta interfaz permite:\n"
        "- Explorar la serie temporal mensual de consumo eléctrico.\n"
        "- Probar el agente RL entrenado para sugerir hiperparámetros ARIMA.\n"
        "- Comparar manualmente distintas configuraciones ARIMA usando AIC / RMSE.\n"
    )

    # ----------------------------------------------------------------------
    # Carga de datos
    # ----------------------------------------------------------------------
    cfg, df_full, train_df, val_df, test_df = load_data_and_splits("config/config.yaml")

    st.sidebar.header("Datos")
    st.sidebar.write(f"**Archivo:** {cfg.get('data', {}).get('source', 'data/germany_monthly_power.csv')}")
    st.sidebar.write(f"Total observaciones: {len(df_full)}")
    st.sidebar.write(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    st.sidebar.write(f"Rango: {df_full.index[0].strftime('%Y-%m')} → {df_full.index[-1].strftime('%Y-%m')}")

    # Carga del agente RL entrenado
    agent, has_model = load_trained_agent(train_df, val_df, cfg)

    # ----------------------------------------------------------------------
    # Tabs principales
    # ----------------------------------------------------------------------
    tab1, tab2, tab3 = st.tabs(["📈 Serie y datos", "🤖 Agente RL", "⚔️ Comparar modelos"])

    # ===========================
    # TAB 1: SERIE Y DATOS
    # ===========================
    with tab1:
        st.subheader("Serie mensual y partición Train / Val / Test")

        plot_series_with_split(df_full, train_df, val_df, test_df)

        st.markdown("### Estadísticas básicas")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Media (train)", f"{train_df['value'].mean():.2f}")
            st.metric("Desv. estándar (train)", f"{train_df['value'].std():.2f}")
        with col2:
            st.metric("Media (val)", f"{val_df['value'].mean():.2f}")
            st.metric("Media (test)", f"{test_df['value'].mean():.2f}")
        with col3:
            st.metric("Mínimo (full)", f"{df_full['value'].min():.2f}")
            st.metric("Máximo (full)", f"{df_full['value'].max():.2f}")

        with st.expander("Ver primeros datos"):
            st.dataframe(df_full.head())

    # ===========================
    # TAB 2: AGENTE RL
    # ===========================
    with tab2:
        st.subheader("Agente de Aprendizaje Reforzado para ARIMA")

        if not has_model:
            st.warning(
                "No se encontró un modelo entrenado del agente RL.\n\n"
                "Entrena primero desde la línea de comandos:\n\n"
                "```bash\n"
                "python -m src.rl_agent --train --timesteps 50000 --config config/config.yaml\n"
                "```"
            )
        else:
            st.success("Modelo del agente RL cargado correctamente desde disco.")

            st.markdown("### Configuración sugerida por el agente")

            if st.button("🔮 Obtener configuración (p, d, q) recomendada"):
                p, d, q = agent.predict_best_config(deterministic=True)

                st.write(f"**Configuración sugerida:** ARIMA({p}, {d}, {q})")

                # Ajustar modelo con esa configuración y pronosticar sobre test
                with st.spinner("Ajustando modelo ARIMA y generando pronóstico..."):
                    model, fc_index, fc_values, conf_int, metrics = make_forecast_with_order(
                        train_df, val_df, test_df, (p, d, q)
                    )

                st.markdown("#### Métricas en conjunto de test (modelo RL-sugerido)")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("AIC", f"{metrics['AIC']:.2f}")
                col2.metric("BIC", f"{metrics['BIC']:.2f}")
                col3.metric("RMSE", f"{metrics['RMSE']:.2f}")
                col4.metric("MAE", f"{metrics['MAE']:.2f}")

                st.markdown("#### Pronóstico vs datos reales (test)")
                plot_forecast(
                    test_df, fc_index, fc_values, conf_int,
                    title=f"Pronóstico ARIMA({p},{d},{q}) sugerido por RL"
                )

    # ===========================
    # TAB 3: COMPARACIÓN DE MODELOS
    # ===========================
    with tab3:
        st.subheader("Comparación de modelos ARIMA (criterio AIC)")

        st.markdown(
            "Selecciona al menos **tres** configuraciones (p,d,q) para comparar. "
            "Puedes incluir la configuración sugerida por RL u otras manuales."
        )

        # Valores por defecto
        default_configs: List[Tuple[int, int, int]] = [(1, 1, 1), (2, 1, 2), (3, 1, 4)]

        cols = st.columns(3)
        configs = []
        for i, col in enumerate(cols):
            with col:
                st.markdown(f"**Modelo {i+1}**")
                p = st.number_input(f"p_{i+1}", min_value=0, max_value=5, value=default_configs[i][0], step=1)
                d = st.number_input(f"d_{i+1}", min_value=0, max_value=2, value=default_configs[i][1], step=1)
                q = st.number_input(f"q_{i+1}", min_value=0, max_value=4, value=default_configs[i][2], step=1)
                configs.append((int(p), int(d), int(q)))

        if st.button("⚔️ Comparar modelos"):
            train_series = train_df["value"]
            val_series = val_df["value"]

            with st.spinner("Ajustando modelos ARIMA y calculando métricas..."):
                models, table = compare_models(train_series, val_series, configs)

            st.markdown("### Resultados ordenados por AIC (menor es mejor)")
            st.dataframe(table.style.highlight_min(subset=["aic"], color="#c7f7c7"))

            best_order = models[0].order
            st.success(f"Mejor modelo según AIC: ARIMA{best_order} (AIC={models[0].aic:.2f})")

            # Opcional: pronóstico del mejor modelo sobre test
            st.markdown("#### Pronóstico del mejor modelo sobre el conjunto de test")
            with st.spinner("Ajustando mejor modelo sobre train+val y pronosticando test..."):
                _, fc_index, fc_values, conf_int, metrics = make_forecast_with_order(
                    train_df, val_df, test_df, best_order
                )

            st.markdown(
                f"**ARIMA{best_order}** – AIC={metrics['AIC']:.2f}, "
                f"RMSE={metrics['RMSE']:.2f}, MAE={metrics['MAE']:.2f}"
            )
            plot_forecast(
                test_df,
                fc_index,
                fc_values,
                conf_int,
                title=f"Pronóstico del mejor modelo ARIMA{best_order}",
            )


if __name__ == "__main__":
    main()
