#!/usr/bin/env python3
# ============================================================================
# Proyecto: Agentificación de Modelos ARIMA con Aprendizaje Reforzado
# Archivo: data_processor.py
# Descripción: Carga, limpieza y división de series temporales para ARIMA + RL
# ============================================================================

import pandas as pd
import numpy as np
import os
from statsmodels.tsa.stattools import adfuller


class TimeSeriesProcessor:
    """
    Clase para cargar y procesar datos mensuales del consumo eléctrico alemán.

    Flujo:
        1. Cargar CSV mensual (60 o 36 meses dependiendo del origen)
        2. Limpiar valores faltantes / outliers
        3. Dividir en train (28), val (3), test (5)
    """

    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.train = None
        self.val = None
        self.test = None

    # ----------------------------------------------------------------------
    def load_data(self):
        print(f"📂 Cargando datos desde {self.data_path}...")

        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Archivo no encontrado: {self.data_path}")

        self.df = pd.read_csv(self.data_path, parse_dates=True, index_col=0)

        # Detectar nombre real de la columna
        if "load_gwh" in self.df.columns:
            col = "load_gwh"
        elif "value" in self.df.columns:
            col = "value"
        else:
            raise ValueError(f"El CSV debe contener columna 'load_gwh' o 'value'.")

        self.df = self.df[[col]].rename(columns={col: "value"})

        print(f"✅ {len(self.df)} observaciones cargadas")
        print(f"   Período: {self.df.index[0].strftime('%Y-%m')} a {self.df.index[-1].strftime('%Y-%m')}\n")

    # ----------------------------------------------------------------------
    def preprocess(self, handle_missing="interpolate", outlier_threshold=3.0):
        if self.df is None:
            raise RuntimeError("Debes ejecutar load_data() antes de preprocess()")

        # Missing
        if handle_missing == "interpolate":
            self.df["value"] = self.df["value"].interpolate()
        elif handle_missing == "forward_fill":
            self.df["value"] = self.df["value"].fillna(method="ffill")
        elif handle_missing == "drop":
            self.df = self.df.dropna()

        # Outliers
        if outlier_threshold is not None:
            mean = self.df["value"].mean()
            std = self.df["value"].std()
            limit = outlier_threshold * std

            high_mask = self.df["value"] > (mean + limit)
            low_mask = self.df["value"] < (mean - limit)

            if high_mask.any() or low_mask.any():
                print("⚠️ Corrigiendo outliers extremos...")
                self.df.loc[high_mask, "value"] = mean + limit
                self.df.loc[low_mask, "value"] = mean - limit

    # ----------------------------------------------------------------------
    def split_data(self):
        if self.df is None:
            raise RuntimeError("Debes ejecutar load_data() antes de split_data()")

        total = len(self.df)

        print("✂️  División de datos:")

        if total < 36:
            raise ValueError(
                f"Dataset demasiado pequeño ({total} meses)."
            )

        self.train = self.df.iloc[:28]
        self.val = self.df.iloc[28:31]
        self.test = self.df.iloc[31:36]

        print(f"   Train: {len(self.train)} meses")
        print(f"   Val:   {len(self.val)} meses")
        print(f"   Test:  {len(self.test)} meses\n")

    # ----------------------------------------------------------------------
    def test_stationarity(self, series, name="Serie"):
        """
        Test ADF para verificar estacionariedad.
        """
        adf = adfuller(series)

        return {
            "test_statistic": adf[0],
            "p_value": adf[1],
            "is_stationary": adf[1] < 0.05
        }

    # ----------------------------------------------------------------------
    def get_train_val_test(self):
        if self.train is None or self.val is None or self.test is None:
            raise RuntimeError("Debes ejecutar split_data() antes de get_train_val_test()")

        return (
            self.train["value"].values.astype(float),
            self.val["value"].values.astype(float),
            self.test["value"].values.astype(float),
        )
