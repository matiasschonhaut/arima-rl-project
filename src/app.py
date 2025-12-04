#!/usr/bin/env python3
# ============================================================================
# Proyecto: Agentificación de Modelos ARIMA con Aprendizaje Reforzado
# Archivo: app.py
# Descripción: Interfaz web interactiva con Streamlit
# ============================================================================

import os
import sys

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from statsmodels.tsa.stattools import acf, pacf, adfuller
from statsmodels.stats.stattools import jarque_bera
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy import stats

# Añadir directorio raíz al path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_processor import TimeSeriesProcessor
from src.arima_utils import (
    ARIMAModel,
    compare_models,
    fit_arima_model,
    create_comparison_table,
    forecast_with_intervals,
)
from src.rl_agent import ARIMAAgent

# ============================================================================
# CONFIGURACIÓN DE LA PÁGINA
# ============================================================================

st.set_page_config(
    page_title="Agente RL-ARIMA",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS personalizado
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2ca02c;
        margin-top: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.25rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
        border-radius: 0.25rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""",
    unsafe_allow_html=True,
)



# ========================================================================
# RUTAS ABSOLUTAS (solución al problema del archivo no encontrado)
# ========================================================================

# BASE_DIR = directorio donde está app.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Ruta absoluta al archivo del modelo RL
MODEL_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "models", "arima_dqn_agent.zip"))

# Ruta absoluta al dataset por defecto
DEFAULT_DATA_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "data", "germany_monthly_power.csv"))


# ========================================================================
# CARGA DE DATOS Y MODELO RL CON RUTAS ABSOLUTAS
# ========================================================================

@st.cache_data
def load_data(path=DEFAULT_DATA_PATH):
    processor = TimeSeriesProcessor(path)
    processor.load_data()
    processor.split_data()
    return processor


@st.cache_resource
def load_rl_agent(data_path=DEFAULT_DATA_PATH, model_path=MODEL_PATH):

    if not os.path.exists(model_path):
        return None

    try:
        processor = load_data(data_path)
        train_data = processor.train['value'].values
        val_data = processor.val['value'].values

        agent = ARIMAAgent(train_data, val_data)
        agent.load(model_path)
        return agent

    except Exception:
        return None


# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def plot_time_series(processor: TimeSeriesProcessor) -> go.Figure:
    """Visualiza serie temporal completa con división train/val/test."""
    train = processor.train
    val = processor.val
    test = processor.test

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=train.index,
            y=train["value"],
            mode="lines+markers",
            name="Train",
            line=dict(color="#1f77b4", width=2),
            marker=dict(size=4),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=val.index,
            y=val["value"],
            mode="lines+markers",
            name="Validation",
            line=dict(color="#ff7f0e", width=2),
            marker=dict(size=4),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=test.index,
            y=test["value"],
            mode="lines+markers",
            name="Test",
            line=dict(color="#2ca02c", width=2),
            marker=dict(size=4),
        )
    )

    fig.update_layout(
        title="Serie Temporal: Consumo Eléctrico Alemán (60 meses)",
        xaxis_title="Fecha",
        yaxis_title="Consumo (GWh)",
        hovermode="x unified",
        height=500,
    )

    return fig


def plot_acf_pacf(series: pd.Series, nlags: int = 12) -> go.Figure:
    """
    Visualiza ACF y PACF controlando que nlags cumpla:
    nlags <= 0.5 * N - 1 (requisito de statsmodels para PACF).
    """
    n = len(series)
    max_pacf_lags = max(1, (n // 2) - 1)

    if nlags > max_pacf_lags:
        nlags = max_pacf_lags

    acf_vals = acf(series, nlags=nlags, fft=False)
    pacf_vals = pacf(series, nlags=nlags, method="ywm")

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Autocorrelation Function (ACF)",
            "Partial Autocorrelation Function (PACF)",
        ),
    )

    # ACF
    fig.add_trace(
        go.Bar(x=list(range(nlags + 1)), y=acf_vals, name="ACF"), row=1, col=1
    )

    # PACF
    fig.add_trace(
        go.Bar(x=list(range(nlags + 1)), y=pacf_vals, name="PACF"), row=1, col=2
    )

    conf_interval = 1.96 / np.sqrt(n)
    for col in [1, 2]:
        fig.add_hline(
            y=conf_interval,
            line_dash="dash",
            line_color="red",
            row=1,
            col=col,
        )
        fig.add_hline(
            y=-conf_interval,
            line_dash="dash",
            line_color="red",
            row=1,
            col=col,
        )

    fig.update_xaxes(title_text="Lag", row=1, col=1)
    fig.update_xaxes(title_text="Lag", row=1, col=2)
    fig.update_yaxes(title_text="Correlation", row=1, col=1)
    fig.update_yaxes(title_text="Correlation", row=1, col=2)

    fig.update_layout(height=400, showlegend=False)
    return fig


def plot_forecast(
    train: pd.DataFrame,
    val: pd.DataFrame,
    forecast: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    title: str = "Pronóstico ARIMA",
) -> go.Figure:
    """Visualiza pronóstico con intervalos de confianza."""
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=train.index,
            y=train["value"],
            mode="lines",
            name="Histórico (Train)",
            line=dict(color="#1f77b4", width=2),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=val.index,
            y=val["value"],
            mode="lines+markers",
            name="Real (Validation)",
            line=dict(color="#2ca02c", width=2),
            marker=dict(size=6),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=val.index,
            y=forecast,
            mode="lines+markers",
            name="Pronóstico",
            line=dict(color="#d62728", width=2, dash="dash"),
            marker=dict(size=6),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=val.index.tolist() + val.index.tolist()[::-1],
            y=upper.tolist() + lower.tolist()[::-1],
            fill="toself",
            fillcolor="rgba(214, 39, 40, 0.2)",
            line=dict(color="rgba(255,255,255,0)"),
            name="IC 95%",
            showlegend=True,
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Fecha",
        yaxis_title="Consumo (GWh)",
        hovermode="x unified",
        height=500,
    )

    return fig


def plot_residuals(residuals: np.ndarray) -> go.Figure:
    """Visualiza diagnóstico de residuos."""
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Residuos vs Tiempo",
            "Histograma de Residuos",
            "Q-Q Plot",
            "ACF de Residuos",
        ),
    )

    # 1. Residuos vs tiempo
    fig.add_trace(
        go.Scatter(
            x=list(range(len(residuals))),
            y=residuals,
            mode="lines+markers",
            name="Residuos",
        ),
        row=1,
        col=1,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)

    # 2. Histograma
    fig.add_trace(
        go.Histogram(x=residuals, nbinsx=20, name="Histograma"), row=1, col=2
    )

    # 3. Q-Q plot
    (osm, osr), (slope, intercept, r) = stats.probplot(residuals, dist="norm")
    fig.add_trace(
        go.Scatter(x=osm, y=osr, mode="markers", name="Q-Q"), row=2, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=osm,
            y=slope * osm + intercept,
            mode="lines",
            name="Línea teórica",
            line=dict(color="red", dash="dash"),
        ),
        row=2,
        col=1,
    )

    # 4. ACF residuos
    nlags = min(20, len(residuals) // 4)
    acf_vals = acf(residuals, nlags=nlags, fft=False)
    fig.add_trace(
        go.Bar(x=list(range(len(acf_vals))), y=acf_vals, name="ACF"),
        row=2,
        col=2,
    )
    conf_interval = 1.96 / np.sqrt(len(residuals))
    fig.add_hline(
        y=conf_interval, line_dash="dash", line_color="red", row=2, col=2
    )
    fig.add_hline(
        y=-conf_interval, line_dash="dash", line_color="red", row=2, col=2
    )

    fig.update_layout(height=700, showlegend=False)
    return fig


def compute_residual_diagnostics(residuals: np.ndarray) -> dict:
    """Calcula diagnósticos básicos de residuos."""
    res = {}
    res["residuals_mean"] = float(np.mean(residuals))
    res["residuals_std"] = float(np.std(residuals, ddof=1))

    # Jarque-Bera
    jb_stat, jb_pvalue, skew, kurtosis = jarque_bera(residuals)
    res["jb_stat"] = float(jb_stat)
    res["jb_pvalue"] = float(jb_pvalue)
    res["skew"] = float(skew)
    res["kurtosis"] = float(kurtosis)
    res["is_normal"] = jb_pvalue > 0.05

    # Ljung-Box: primera lag razonable
    lag = max(1, min(10, len(residuals) // 2))
    lb_df = acorr_ljungbox(residuals, lags=[lag], return_df=True)
    lb_pvalue = lb_df["lb_pvalue"].iloc[0]
    res["lb_pvalue"] = float(lb_pvalue)
    res["no_autocorrelation"] = lb_pvalue > 0.05

    # Estabilidad de varianza (ratio entre mitades)
    if len(residuals) >= 10:
        mid = len(residuals) // 2
        var1 = np.var(residuals[:mid], ddof=1)
        var2 = np.var(residuals[mid:], ddof=1)
        if var2 == 0:
            variance_ratio = np.inf
        else:
            variance_ratio = var1 / var2
    else:
        variance_ratio = 1.0

    res["variance_ratio"] = float(variance_ratio)
    res["variance_stable"] = 0.5 < variance_ratio < 2.0

    return res


def adf_test_wrapper(series: pd.Series, name: str = "Serie") -> dict:
    """
    Wrapper que usa:
    - processor.test_stationarity si existe
    - sino, adfuller directamente.
    """
    # Intentar usar método del processor si existe
    # (esta función se llamará pasando processor.train['value'])
    try:
        # si quien llama tiene processor con test_stationarity, usará eso
        # (la firma real se maneja en main)
        raise AttributeError  # forzamos fallback aquí; se maneja en main
    except AttributeError:
        result = adfuller(series, autolag="AIC")
        test_stat, p_value, *_ = result
        return {
            "name": name,
            "test_statistic": float(test_stat),
            "p_value": float(p_value),
            "is_stationary": p_value < 0.05,
        }


# ============================================================================
# INTERFAZ PRINCIPAL
# ============================================================================


def main():
    """Función principal de la aplicación Streamlit."""

    # Header
    st.markdown(
        '<p class="main-header">🤖 Agente RL-ARIMA para Forecasting de Series Temporales</p>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/300x100/1f77b4/ffffff?text=RL-ARIMA")
        st.title("⚙️ Configuración")

        data_path = st.text_input(
            "Ruta de datos", "data/germany_monthly_power.csv"
        )

        if os.path.exists(data_path):
            st.success("✅ Datos encontrados")
            processor = load_data(data_path)
        else:
            st.error("❌ Archivo de datos no encontrado")
            st.info("💡 Ejecuta: `python data/download_data.py`")
            st.stop()

        st.markdown("---")

        # Info dataset
        st.subheader("📊 Información del Dataset")
        st.metric("Total de meses", len(processor.df))
        st.metric(
            "Período",
            f"{processor.df.index[0].strftime('%Y-%m')} - {processor.df.index[-1].strftime('%Y-%m')}",
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Train", f"{len(processor.train)}")
        with c2:
            st.metric("Val", f"{len(processor.val)}")
        with c3:
            st.metric("Test", f"{len(processor.test)}")

        st.markdown("---")

        # Modelo RL
        if os.path.exists(MODEL_PATH):
            st.success("✅ Modelo RL disponible")
            rl_agent = load_rl_agent(data_path=DEFAULT_DATA_PATH, model_path=MODEL_PATH)

        else:
            st.warning("⚠️ Modelo RL no entrenado")
            st.info(
                "Entrena el agente para usar Modo Automático:\n\n"
                "`python -m src.rl_agent --train --timesteps 50000`"
            )
            rl_agent = None

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "📊 Exploración de Datos",
            "🤖 Agente RL / Manual",
            "📈 Comparación de Modelos",
            "🔍 Diagnósticos",
        ]
    )

    # ========================================================================
    # TAB 1: EXPLORACIÓN DE DATOS
    # ========================================================================
    with tab1:
        st.markdown(
            '<p class="sub-header">📊 Exploración de Datos</p>',
            unsafe_allow_html=True,
        )

        # Serie temporal
        st.plotly_chart(plot_time_series(processor), use_container_width=True)

        # Estadísticas descriptivas
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📈 Estadísticas Descriptivas")
            stats_df = pd.DataFrame(
                {
                    "Métrica": [
                        "Media",
                        "Desv. Estándar",
                        "Mínimo",
                        "Máximo",
                        "Mediana",
                    ],
                    "Valor (GWh)": [
                        f"{processor.df['value'].mean():.2f}",
                        f"{processor.df['value'].std():.2f}",
                        f"{processor.df['value'].min():.2f}",
                        f"{processor.df['value'].max():.2f}",
                        f"{processor.df['value'].median():.2f}",
                    ],
                }
            )
            st.dataframe(stats_df, use_container_width=True, hide_index=True)

        with col2:
            st.subheader("🔍 Test de Estacionariedad (ADF)")
            # Intentamos usar método del processor si existe, sino fallback
            try:
                adf_result = processor.test_stationarity(
                    processor.train["value"], name="Training Set"
                )
            except AttributeError:
                adf_result = adf_test_wrapper(
                    processor.train["value"], name="Training Set"
                )

            adf_df = pd.DataFrame(
                {
                    "Métrica": [
                        "Estadístico ADF",
                        "P-value",
                        "Estacionaria",
                    ],
                    "Valor": [
                        f"{adf_result['test_statistic']:.4f}",
                        f"{adf_result['p_value']:.4f}",
                        "✅ Sí"
                        if adf_result["is_stationary"]
                        else "❌ No",
                    ],
                }
            )
            st.dataframe(adf_df, use_container_width=True, hide_index=True)

        # ACF / PACF
        if st.checkbox("📊 Mostrar funciones ACF/PACF"):
            max_lags = max(1, (len(processor.train) // 2) - 1)
            nlags = st.slider(
                "Número de lags",
                min_value=1,
                max_value=max_lags,
                value=min(12, max_lags),
            )

            st.plotly_chart(
                plot_acf_pacf(processor.train["value"], nlags=nlags),
                use_container_width=True,
            )

    # ========================================================================
    # TAB 2: AGENTE RL / MANUAL
    # ========================================================================
    with tab2:
        st.markdown(
            '<p class="sub-header">🤖 Predicción con Agente RL / Modo Manual</p>',
            unsafe_allow_html=True,
        )

        mode = st.radio(
            "Seleccione modo:",
            ["🤖 Modo Automático (Agente RL)", "🎛️ Modo Manual (Sliders)"],
            horizontal=True,
        )

        train_series = processor.train["value"]
        val_series = processor.val["value"]

        # -------------------- MODO RL --------------------
        if mode == "🤖 Modo Automático (Agente RL)":
            st.markdown("### 🤖 Modo Automático con Agente RL")

            if rl_agent is None:
                st.error("❌ Agente RL no disponible.")
                st.code(
                    "python -m src.rl_agent --train --timesteps 50000",
                    language="bash",
                )
            else:
                if st.button("🎯 Predecir Mejor Configuración", type="primary"):
                    with st.spinner("🤖 Agente RL analizando..."):
                        p_pred, d_pred, q_pred = rl_agent.predict_best_config()
                        st.session_state["rl_config"] = (p_pred, d_pred, q_pred)
                        st.success(
                            f"✅ Agente RL recomienda: ARIMA({p_pred}, {d_pred}, {q_pred})"
                        )

                if "rl_config" in st.session_state:
                    p_rl, d_rl, q_rl = st.session_state["rl_config"]
                    st.info(
                        f"📋 Configuración propuesta: **ARIMA({p_rl}, {d_rl}, {q_rl})**"
                    )

                    if st.button("▶️ Entrenar Modelo ARIMA Propuesto"):
                        with st.spinner("⏳ Entrenando modelo..."):
                            try:
                                model = fit_arima_model(
                                    train_series,
                                    val_series,
                                    order=(p_rl, d_rl, q_rl),
                                )

                                # Forecast e intervalos
                                steps = len(val_series)
                                forecast, lower, upper = forecast_with_intervals(
                                    model, steps=steps, alpha=0.05
                                )

                                # Residuales en validación
                                residuals = (
                                    val_series.values[:steps] - forecast
                                )

                                # Guardamos en session_state para diagnósticos
                                st.session_state["rl_model"] = model
                                st.session_state["rl_residuals"] = residuals

                                # Métricas
                                c1, c2, c3, c4 = st.columns(4)
                                with c1:
                                    st.metric("AIC", f"{model.aic:.2f}")
                                with c2:
                                    st.metric("BIC", f"{model.bic:.2f}")
                                with c3:
                                    st.metric("RMSE", f"{model.rmse:.2f}")
                                with c4:
                                    st.metric("MAE", f"{model.mae:.2f}")

                                st.plotly_chart(
                                    plot_forecast(
                                        processor.train,
                                        processor.val,
                                        forecast,
                                        lower,
                                        upper,
                                        title=f"Pronóstico ARIMA({p_rl}, {d_rl}, {q_rl})",
                                    ),
                                    use_container_width=True,
                                )

                            except Exception as e:
                                st.error(f"❌ Error al entrenar modelo: {e}")

        # -------------------- MODO MANUAL --------------------
        else:
            st.markdown("### 🎛️ Modo Manual con Sliders")

            c1, c2, c3 = st.columns(3)
            with c1:
                p_manual = st.slider("p (Orden AR)", 0, 5, 1)
            with c2:
                d_manual = st.slider("d (Diferenciación)", 0, 2, 1)
            with c3:
                q_manual = st.slider("q (Orden MA)", 0, 4, 1)

            st.info(
                f"📋 Configuración seleccionada: **ARIMA({p_manual}, {d_manual}, {q_manual})**"
            )

            if st.button("🚀 Entrenar y Evaluar Modelo", type="primary"):
                with st.spinner("⏳ Entrenando modelo..."):
                    try:
                        model = fit_arima_model(
                            train_series,
                            val_series,
                            order=(p_manual, d_manual, q_manual),
                        )

                        steps = len(val_series)
                        forecast, lower, upper = forecast_with_intervals(
                            model, steps=steps, alpha=0.05
                        )

                        residuals = val_series.values[:steps] - forecast

                        st.session_state["manual_model"] = model
                        st.session_state["manual_residuals"] = residuals

                        c1, c2, c3, c4 = st.columns(4)
                        with c1:
                            st.metric("AIC", f"{model.aic:.2f}")
                        with c2:
                            st.metric("BIC", f"{model.bic:.2f}")
                        with c3:
                            st.metric("RMSE", f"{model.rmse:.2f}")
                        with c4:
                            st.metric("MAE", f"{model.mae:.2f}")

                        st.plotly_chart(
                            plot_forecast(
                                processor.train,
                                processor.val,
                                forecast,
                                lower,
                                upper,
                                title=f"Pronóstico ARIMA({p_manual}, {d_manual}, {q_manual})",
                            ),
                            use_container_width=True,
                        )

                    except Exception as e:
                        st.error(f"❌ Error al entrenar modelo: {e}")

    # ========================================================================
    # TAB 3: COMPARACIÓN DE MODELOS
    # ========================================================================
    with tab3:
        st.markdown(
            '<p class="sub-header">📈 Comparación de Modelos ARIMA</p>',
            unsafe_allow_html=True,
        )

        st.markdown("### ⚙️ Configurar Modelos a Comparar")

        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("**Modelo 1**")
            p1 = st.number_input("p", 0, 5, 1, key="p1")
            d1 = st.number_input("d", 0, 2, 1, key="d1")
            q1 = st.number_input("q", 0, 4, 1, key="q1")

        with c2:
            st.markdown("**Modelo 2**")
            p2 = st.number_input("p", 0, 5, 2, key="p2")
            d2 = st.number_input("d", 0, 2, 1, key="d2")
            q2 = st.number_input("q", 0, 4, 1, key="q2")

        with c3:
            st.markdown("**Modelo 3**")
            p3 = st.number_input("p", 0, 5, 1, key="p3")
            d3 = st.number_input("d", 0, 2, 1, key="d3")
            q3 = st.number_input("q", 0, 4, 2, key="q3")

        if st.button("📊 Comparar Modelos", type="primary"):
            configs = [(p1, d1, q1), (p2, d2, q2), (p3, d3, q3)]

            with st.spinner("⏳ Comparando modelos..."):
                models_sorted, df_results = compare_models(
                    processor.train["value"],
                    processor.val["value"],
                    configs,
                )

                comparison_df = create_comparison_table(df_results)

                st.markdown("### 📊 Tabla Comparativa")

                def highlight_best(row):
                    return [
                        "background-color: #d4edda" if row.name == 0 else ""
                        for _ in row
                    ]

                st.dataframe(
                    comparison_df.style.apply(
                        highlight_best, axis=1
                    ).format(
                        {
                            "aic": "{:.2f}",
                            "bic": "{:.2f}",
                            "rmse": "{:.2f}",
                            "mae": "{:.2f}",
                            "training_time": "{:.2f}",
                        }
                    ),
                    use_container_width=True,
                )

                best_row = comparison_df.iloc[0]
                st.success(
                    f"🏆 Mejor modelo: {best_row['order_str']} "
                    f"con AIC = {best_row['aic']:.2f}"
                )

                # Gráfica comparativa (AIC/BIC/RMSE/MAE)
                fig_comp = go.Figure()
                metrics = ["aic", "bic", "rmse", "mae"]
                metric_labels = ["AIC", "BIC", "RMSE", "MAE"]

                for _, row in comparison_df.iterrows():
                    order = row.get("order")
                    if isinstance(order, (list, tuple)) and len(order) == 3:
                        p, d, q = order
                        name = f"ARIMA({p},{d},{q})"
                    else:
                        name = row.get("order_str", "ARIMA")

                    fig_comp.add_trace(
                        go.Bar(
                            name=name,
                            x=metric_labels,
                            y=[row[m] for m in metrics],
                        )
                    )

                fig_comp.update_layout(
                    title="Comparación de Métricas por Modelo",
                    xaxis_title="Métrica",
                    yaxis_title="Valor",
                    barmode="group",
                    height=500,
                )

                st.plotly_chart(fig_comp, use_container_width=True)

                # Exportar CSV
                csv = comparison_df.to_csv(index=False)
                st.download_button(
                    label="📥 Descargar Tabla como CSV",
                    data=csv,
                    file_name="comparacion_modelos_arima.csv",
                    mime="text/csv",
                )

    # ========================================================================
    # TAB 4: DIAGNÓSTICOS
    # ========================================================================
    with tab4:
        st.markdown(
            '<p class="sub-header">🔍 Diagnóstico de Residuos</p>',
            unsafe_allow_html=True,
        )

        model_option = st.selectbox(
            "Seleccione modelo para diagnosticar:",
            ["Modelo RL (si disponible)", "Modelo Manual (si disponible)"],
        )

        model_obj = None
        residuals = None
        config_name = ""

        if (
            model_option == "Modelo RL (si disponible)"
            and "rl_model" in st.session_state
            and "rl_residuals" in st.session_state
        ):
            model_obj = st.session_state["rl_model"]
            residuals = st.session_state["rl_residuals"]
            config_name = f"RL: ARIMA{model_obj.order}"

        elif (
            model_option == "Modelo Manual (si disponible)"
            and "manual_model" in st.session_state
            and "manual_residuals" in st.session_state
        ):
            model_obj = st.session_state["manual_model"]
            residuals = st.session_state["manual_residuals"]
            config_name = f"Manual: ARIMA{model_obj.order}"

        if model_obj is not None and residuals is not None:
            st.markdown(f"### 📋 Modelo: {config_name}")

            diagnostics = compute_residual_diagnostics(residuals)

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric(
                    "Media", f"{diagnostics['residuals_mean']:.4f}"
                )
            with c2:
                st.metric(
                    "Desv. Estándar",
                    f"{diagnostics['residuals_std']:.4f}",
                )
            with c3:
                normal_status = (
                    "✅" if diagnostics["is_normal"] else "❌"
                )
                st.metric(
                    "Normalidad (JB)",
                    f"{normal_status} (p={diagnostics['jb_pvalue']:.4f})",
                )
            with c4:
                autocorr_status = (
                    "✅" if diagnostics["no_autocorrelation"] else "❌"
                )
                st.metric(
                    "Sin Autocorr (LB)",
                    f"{autocorr_status} (p={diagnostics['lb_pvalue']:.4f})",
                )

            st.markdown("### 📊 Interpretación")

            checks = [
                (
                    "✅"
                    if abs(diagnostics["residuals_mean"]) < 0.05 * abs(processor.train["value"].mean())
                    else "⚠️",
                    "Media de residuos aceptable (sesgo bajo)",
                    f"Media = {diagnostics['residuals_mean']:.4f}  —  umbral = {0.05 * abs(processor.train['value'].mean()):.2f}",
                ),

                (
                    "✅" if diagnostics["is_normal"] else "⚠️",
                    "Residuos siguen distribución normal",
                    f"Test JB p-value = {diagnostics['jb_pvalue']:.4f}",
                ),
                (
                    "✅" if diagnostics["no_autocorrelation"] else "⚠️",
                    "No hay autocorrelación en residuos",
                    f"Test LB p-value = {diagnostics['lb_pvalue']:.4f}",
                ),
                (
                    "✅"
                    if diagnostics["variance_stable"]
                    else "⚠️",
                    "Varianza de residuos estable",
                    f"Ratio = {diagnostics['variance_ratio']:.2f}",
                ),
            ]

            for status, check, detail in checks:
                st.markdown(f"{status} **{check}** - {detail}")

            st.markdown("### 📈 Gráficas de Diagnóstico")
            st.plotly_chart(
                plot_residuals(residuals), use_container_width=True
            )

        else:
            st.info(
                "ℹ️ Entrena un modelo primero en la pestaña "
                "'Agente RL / Manual' para ver diagnósticos."
            )


if __name__ == "__main__":
    main()
