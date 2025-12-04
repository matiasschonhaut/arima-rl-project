#!/usr/bin/env python3
# ============================================================================
# Proyecto: Agentificación de Modelos ARIMA con Aprendizaje Reforzado
# Archivo: generate_figures.py
# Descripción: Genera figuras para el informe:
#   01_serie_train_val_test.png      -> Serie con partición train/val/test
#   02_acf_pacf.png                  -> ACF/PACF de la serie train
#   03_adf_test.png                  -> Test ADF + rolling mean/std
#   04_residuos_hist_acf.png         -> Residuos, histograma, ACF
#   05_residuos_qqplot.png           -> QQ-plot de residuos
#   06_forecast_test.png             -> Forecast sobre test con IC 95%
#   07_env_rl_diagram.png            -> Diagrama del entorno RL–ARIMA
#   08_dqn_architecture.png          -> Diagrama de la arquitectura DQN
# ============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.graphics.gofplots import qqplot
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import jarque_bera

from matplotlib.patches import Rectangle, FancyArrow

FIG_DIR = "assets/figures"
DATA_PATH = "data/germany_monthly_power.csv"


# ============================================================================
# CARGA Y SPLIT DE DATOS
# ============================================================================

def load_and_split_data():
    """
    Carga germany_monthly_power.csv y divide en:
        - train: 28 meses (2015-01 a 2017-04)
        - val:   3 meses (2017-05 a 2017-07)
        - test:  5 meses (2017-08 a 2017-12)
    """
    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)

    if "load_gwh" in df.columns:
        series = df["load_gwh"]
    else:
        series = df.iloc[:, 0]
        series.name = "value"

    # Split fijo como se usó en rl_agent
    train = series.iloc[:28]
    val = series.iloc[28:31]
    test = series.iloc[31:36]

    print("✅ Datos cargados para figuras:")
    print(f"   Total: {len(series)} meses")
    print(f"   Train: {len(train)} meses ({train.index[0].strftime('%Y-%m')} → {train.index[-1].strftime('%Y-%m')})")
    print(f"   Val:   {len(val)} meses ({val.index[0].strftime('%Y-%m')} → {val.index[-1].strftime('%Y-%m')})")
    print(f"   Test:  {len(test)} meses ({test.index[0].strftime('%Y-%m')} → {test.index[-1].strftime('%Y-%m')})")

    return series, train, val, test


# ============================================================================
# FIGURA 1: SERIE + TRAIN/VAL/TEST
# ============================================================================

def plot_series_with_split(series, train, val, test, save=True):
    os.makedirs(FIG_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(series.index, series.values, color="lightgray",
            label="Serie completa", linewidth=1.5)
    ax.plot(train.index, train.values, label="Train", linewidth=2)
    ax.plot(val.index, val.values, label="Validation", linewidth=2)
    ax.plot(test.index, test.values, label="Test", linewidth=2)

    ax.set_title("Consumo eléctrico mensual – Partición Train / Val / Test")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Consumo [GWh]")
    ax.legend()
    fig.tight_layout()

    if save:
        path = os.path.join(FIG_DIR, "01_serie_train_val_test.png")
        fig.savefig(path, dpi=300)
        print(f"✅ Figura guardada: {path}")

    plt.close(fig)


# ============================================================================
# FIGURA 2: ACF / PACF
# ============================================================================

def plot_acf_pacf(series_train, lags=24, save=True):
    """
    ACF y PACF sobre la parte de entrenamiento.
    Por restricción de statsmodels: nlags < N/2.
    """
    os.makedirs(FIG_DIR, exist_ok=True)

    safe_lags = min(lags, len(series_train) // 2 - 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    plot_acf(series_train, ax=axes[0], lags=safe_lags)
    plot_pacf(series_train, ax=axes[1], lags=safe_lags)

    axes[0].set_title(f"ACF (Train, lags={safe_lags})")
    axes[1].set_title(f"PACF (Train, lags={safe_lags})")

    fig.tight_layout()

    if save:
        path = os.path.join(FIG_DIR, "02_acf_pacf.png")
        fig.savefig(path, dpi=300)
        print(f"✅ Figura guardada: {path}")

    plt.close(fig)


# ============================================================================
# FIGURA 3: TEST ADF + ROLLING MEAN/STD
# ============================================================================

def plot_adf_test(series_train, window=6, save=True):
    """
    Test de Dickey-Fuller aumentado sobre el conjunto de entrenamiento.
    Muestra la serie, la media móvil y la desviación estándar móvil,
    junto con los resultados numéricos del test.
    """
    os.makedirs(FIG_DIR, exist_ok=True)

    # ADF Test
    adf_result = adfuller(series_train, autolag="AIC")
    adf_stat = adf_result[0]
    p_value = adf_result[1]
    crit_values = adf_result[4]

    print("\n📊 Resultado Test ADF (Train):")
    print(f"   Estadístico ADF: {adf_stat:.4f}")
    print(f"   p-value:         {p_value:.4f}")
    for k, v in crit_values.items():
        print(f"   Valor crítico {k}%: {v:.4f}")

    # Rolling mean & std
    rolling_mean = series_train.rolling(window=window).mean()
    rolling_std = series_train.rolling(window=window).std()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(series_train.index, series_train.values, label="Train", linewidth=1.8)
    ax.plot(rolling_mean.index, rolling_mean.values, label=f"Media móvil ({window})", linestyle="--")
    ax.plot(rolling_std.index, rolling_std.values, label=f"Desv. estándar móvil ({window})", linestyle=":")

    ax.set_title("Test de Estacionariedad (ADF) – Train")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Consumo [GWh]")
    ax.legend(loc="upper left")

    # Cuadro de texto con resultados ADF
    textstr = "\n".join([
        f"ADF: {adf_stat:.3f}",
        f"p-value: {p_value:.3f}",
        "Valores críticos:",
        *[f"  {level}%: {val:.3f}" for level, val in crit_values.items()]
    ])

    ax.text(0.02, 0.02, textstr,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="bottom",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    fig.tight_layout()

    if save:
        path = os.path.join(FIG_DIR, "03_adf_test.png")
        fig.savefig(path, dpi=300)
        print(f"✅ Figura guardada: {path}")

    plt.close(fig)


# ============================================================================
# AJUSTE ARIMA Y RESIDUOS
# ============================================================================

def fit_arima_and_residuals(series, order=(3, 1, 4)):
    """
    Ajusta un ARIMA(p,d,q) sobre la serie completa (36 meses)
    y retorna el modelo y sus residuos.
    """
    print(f"\n🧠 Ajustando ARIMA{order} sobre la serie completa para diagnóstico de residuos...")
    model = ARIMA(series, order=order)
    result = model.fit()
    residuals = result.resid

    print(f"   AIC = {result.aic:.2f}")
    print(f"   BIC = {result.bic:.2f}")

    return result, residuals


# ============================================================================
# FIGURA 4 y 5: DIAGNÓSTICO DE RESIDUOS
# ============================================================================

def plot_residual_diagnostics(residuals, save=True):
    """
    Figura 4: Residuos en el tiempo + histograma + ACF.
    Figura 5: QQ-plot de residuos.
    Además imprime Ljung-Box y Jarque-Bera.
    """
    os.makedirs(FIG_DIR, exist_ok=True)

    # --- Tests estadísticos ---
    lb = acorr_ljungbox(residuals, lags=[12], return_df=True)
    jb_stat, jb_pvalue, skew, kurt = jarque_bera(residuals)

    print("\n📊 Diagnóstico de residuos:")
    print("   Ljung-Box (lag=12):")
    print(f"     estadístico = {lb['lb_stat'].iloc[0]:.3f}, p-value = {lb['lb_pvalue'].iloc[0]:.3f}")
    print("   Jarque-Bera:")
    print(f"     JB = {jb_stat:.3f}, p-value = {jb_pvalue:.3f}")
    print(f"     skew = {skew:.3f}, kurtosis = {kurt:.3f}")

    # Figura 4: residuos + histograma + ACF
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Residuos vs tiempo
    axes[0].plot(residuals.index, residuals.values, linewidth=1.5)
    axes[0].axhline(0, color="black", linewidth=1)
    axes[0].set_title("Residuos del modelo ARIMA(3,1,4)")
    axes[0].set_xlabel("Fecha")

    # Histograma
    axes[1].hist(residuals.values, bins=10, edgecolor="black", alpha=0.7)
    axes[1].set_title("Histograma de residuos")

    # ACF de residuos
    plot_acf(residuals, ax=axes[2], lags=min(12, len(residuals)//2 - 1))
    axes[2].set_title("ACF de residuos")

    fig.tight_layout()
    if save:
        path = os.path.join(FIG_DIR, "04_residuos_hist_acf.png")
        fig.savefig(path, dpi=300)
        print(f"✅ Figura guardada: {path}")
    plt.close(fig)

    # Figura 5: QQ-plot
    fig_qq = qqplot(residuals, line="s")
    plt.title("QQ-plot de residuos ARIMA(3,1,4)")
    plt.tight_layout()
    if save:
        path = os.path.join(FIG_DIR, "05_residuos_qqplot.png")
        fig_qq.savefig(path, dpi=300)
        print(f"✅ Figura guardada: {path}")
    plt.close(fig_qq)


# ============================================================================
# FIGURA 6: FORECAST SOBRE TEST (IC 95%)
# ============================================================================

def plot_forecast_on_test(series, train, val, test, order=(3, 1, 4),
                          steps_ahead=None, save=True):
    os.makedirs(FIG_DIR, exist_ok=True)

    if steps_ahead is None:
        steps_ahead = len(test)

    full_train = pd.concat([train, val])
    print(f"\n📈 Ajustando ARIMA{order} sobre train+val y pronosticando test...")

    model = ARIMA(full_train, order=order)
    result = model.fit()

    forecast_res = result.get_forecast(steps=steps_ahead)
    forecast_mean = forecast_res.predicted_mean
    conf_int = forecast_res.conf_int(alpha=0.05)  # 95%

    fc_index = test.index[:steps_ahead]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(full_train.index, full_train.values,
            label="Train+Val", linewidth=2)
    ax.plot(test.index, test.values,
            label="Real (Test)", linewidth=2)
    ax.plot(fc_index, forecast_mean.values,
            label="Forecast ARIMA(3,1,4)", linewidth=2)

    ax.fill_between(
        fc_index,
        conf_int.iloc[:, 0].values,
        conf_int.iloc[:, 1].values,
        alpha=0.2,
        label="IC 95%"
    )

    ax.set_title("Pronóstico sobre conjunto de test – ARIMA(3,1,4)")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Consumo [GWh]")
    ax.legend()
    fig.tight_layout()

    if save:
        path = os.path.join(FIG_DIR, "06_forecast_test.png")
        fig.savefig(path, dpi=300)
        print(f"✅ Figura guardada: {path}")

    plt.close(fig)


# ============================================================================
# FIGURA 7: DIAGRAMA ENTORNO RL–ARIMA
# ============================================================================

def plot_env_diagram(save=True):
    os.makedirs(FIG_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")

    # Coordenadas básicas
    y = 0.5
    x_state = 0.05
    x_agent = 0.3
    x_action = 0.55
    x_env = 0.8

    # Cajas
    boxes = [
        (x_state, y, 0.18, 0.2, "Estado $s_t$\n(8 features)"),
        (x_agent, y, 0.18, 0.2, "Agente DQN\n(red neuronal)"),
        (x_action, y, 0.18, 0.2, "Acción $a_t$\n$(p,d,q)$"),
        (x_env, y, 0.18, 0.2, "Entorno ARIMA\n(Ajuste y evaluación)"),
    ]

    for (x, yb, w, h, text) in boxes:
        rect = Rectangle((x, yb - h/2), w, h,
                         linewidth=1.5, edgecolor="black", facecolor="white")
        ax.add_patch(rect)
        ax.text(x + w/2, yb, text, ha="center", va="center", fontsize=9)

    # Flechas principales
    def arrow(x1, y1, x2, y2, text=None):
        ax.add_patch(FancyArrow(x1, y1, x2 - x1, y2 - y1,
                                width=0.005, length_includes_head=True,
                                head_width=0.03, head_length=0.03))
        if text:
            ax.text((x1 + x2) / 2, y1 + 0.07, text,
                    ha="center", va="bottom", fontsize=8)

    arrow(x_state + 0.18, y, x_agent, y, "observación $s_t$")
    arrow(x_agent + 0.18, y, x_action, y, "acción $a_t$")
    arrow(x_action + 0.18, y, x_env, y, "$(p,d,q)$")

    # Flecha de retorno (recompensa y siguiente estado)
    arrow(x_env + 0.18, y - 0.05, x_state + 0.18, y - 0.05,
          "$r_t$, $s_{t+1}$")

    ax.set_title("Diagrama del entorno RL–ARIMA")

    fig.tight_layout()
    if save:
        path = os.path.join(FIG_DIR, "07_env_rl_diagram.png")
        fig.savefig(path, dpi=300)
        print(f"✅ Figura guardada: {path}")
    plt.close(fig)


# ============================================================================
# FIGURA 8: ARQUITECTURA DQN
# ============================================================================

def plot_dqn_architecture(n_state=8, n_actions=90, save=True):
    """
    Dibuja una arquitectura típica de DQN:
        Input(8) -> Dense(256, ReLU) -> Dense(256, ReLU) -> Output(90)
    """
    os.makedirs(FIG_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis("off")

    # Coordenadas
    x_input = 0.05
    x_hidden1 = 0.3
    x_hidden2 = 0.55
    x_output = 0.8
    y = 0.5
    w, h = 0.18, 0.2

    layers = [
        (x_input, "Entrada\nEstado (8)"),
        (x_hidden1, "Capa oculta 1\nDense(256, ReLU)"),
        (x_hidden2, "Capa oculta 2\nDense(256, ReLU)"),
        (x_output, f"Salida\nQ(s,a) para {n_actions} acciones"),
    ]

    for (x, text) in layers:
        rect = Rectangle((x, y - h/2), w, h,
                         linewidth=1.5, edgecolor="black", facecolor="white")
        ax.add_patch(rect)
        ax.text(x + w/2, y, text, ha="center", va="center", fontsize=9)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrow(x1, y1, x2 - x1, y2 - y1,
                                width=0.005, length_includes_head=True,
                                head_width=0.03, head_length=0.03))

    arrow(x_input + w, y, x_hidden1, y)
    arrow(x_hidden1 + w, y, x_hidden2, y)
    arrow(x_hidden2 + w, y, x_output, y)

    ax.set_title("Arquitectura de la red DQN para selección de ARIMA")

    fig.tight_layout()
    if save:
        path = os.path.join(FIG_DIR, "08_dqn_architecture.png")
        fig.savefig(path, dpi=300)
        print(f"✅ Figura guardada: {path}")
    plt.close(fig)


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("📂 Cargando datos y realizando split train/val/test...")
    series, train, val, test = load_and_split_data()

    print("\n📊 Generando Figura 1: Serie con partición train/val/test...")
    plot_series_with_split(series, train, val, test)

    print("\n📊 Generando Figura 2: ACF y PACF (Train)...")
    plot_acf_pacf(train, lags=24)

    print("\n📊 Generando Figura 3: Test ADF + rolling mean/std (Train)...")
    plot_adf_test(train, window=6)

    print("\n📊 Ajustando ARIMA(3,1,4) y generando diagnóstico de residuos...")
    result, residuals = fit_arima_and_residuals(series, order=(3, 1, 4))
    plot_residual_diagnostics(residuals)

    print("\n📊 Generando Figura 6: Forecast sobre el conjunto de test...")
    plot_forecast_on_test(series, train, val, test, order=(3, 1, 4))

    print("\n📊 Generando Figura 7: Diagrama del entorno RL–ARIMA...")
    plot_env_diagram()

    print("\n📊 Generando Figura 8: Arquitectura de la red DQN...")
    plot_dqn_architecture(n_state=8, n_actions=90)

    print("\n🎉 ¡Todas las figuras han sido generadas y guardadas en assets/figures/!")


if __name__ == "__main__":
    main()
