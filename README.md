# 🤖 Agente RL-ARIMA para Forecasting de Series Temporales

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-green.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.27+-red.svg)](https://streamlit.io/)

## 📋 Descripción

Sistema completo de **Aprendizaje Reforzado (RL)** para la optimización automática de hiperparámetros de modelos **ARIMA** aplicado al pronóstico de consumo eléctrico. El proyecto implementa un agente **DQN** (Deep Q-Network) que aprende a seleccionar configuraciones óptimas (p, d, q) mediante interacción con un entorno **Gymnasium** personalizado.

### 🎯 Características Principales

- �?**Agente RL entrenado** con 50k timesteps usando Stable-Baselines3
- �?**Entorno Gymnasium personalizado** con función de recompensa multiobjetivo
- �?**Interfaz web interactiva** con Streamlit (dual mode: automático/manual)
- �?**Comparación de múltiples modelos** ARIMA con métricas completas
- �?**Diagnóstico de residuos** completo (ACF, Q-Q Plot, Ljung-Box, Jarque-Bera)
- �?**Containerización Docker** con generación automática de datos
- �?**Dataset real**: 60 meses de consumo eléctrico alemán (OPSD)

---

## 📊 Fundamento Técnico

### Problema a Resolver

La selección manual de hiperparámetros ARIMA (p, d, q) es:
- �?**Consume tiempo**: Requiere 60+ iteraciones típicamente
- 🎓 **Requiere expertise**: Análisis de ACF/PACF y pruebas de estacionariedad
- 🔄 **Proceso iterativo**: Ajuste basado en AIC, BIC, RMSE

### Solución Propuesta

Usar un **agente de aprendizaje reforzado** que:
1. **Aprende políticas óptimas** mediante exploración del espacio de configuraciones
2. **Reduce tiempo en 50-70%** vs. grid search exhaustivo
3. **Generaliza** a nuevas series temporales con fine-tuning mínimo

### Arquitectura del Sistema

```
┌─────────────────────────────────────────────────────────�?�?                 INTERFAZ WEB (Streamlit)               �?�? ┌─────────────�? ┌──────────────�? ┌───────────────�?�?�? �?Exploración �? �? Agente RL/  �? �? Comparación  �?�?�? �?   Datos    �? �?   Manual    �? �?   Modelos    �?�?�? └─────────────�? └──────────────�? └───────────────�?�?└─────────────────────────────────────────────────────────�?                            �?┌─────────────────────────────────────────────────────────�?�?             AGENTE RL (DQN - Stable-Baselines3)        �?�? �?Red neuronal: [128, 128]                             �?�? �?Exploration: ε-greedy (1.0 �?0.05)                   �?�? �?Replay buffer: 10,000 experiencias                   �?└─────────────────────────────────────────────────────────�?                            �?┌─────────────────────────────────────────────────────────�?�?        ENTORNO GYMNASIUM (ARIMAHyperparamEnv)          �?�? �?Estados: [RMSE, AIC, p, d, q, step, ...]            �?�? �?Acciones: (p, d, q) discreto                         �?�? �?Recompensa: f(accuracy, AIC, time, diagnostics)      �?└─────────────────────────────────────────────────────────�?                            �?┌─────────────────────────────────────────────────────────�?�?             MODELOS ARIMA (Statsmodels)                �?�? �?Entrenamiento en 48 meses                            �?�? �?Validación en 6 meses                                �?�? �?Prueba en 6 meses                                    �?└─────────────────────────────────────────────────────────�?```

---

## 🚀 Instalación Rápida

### Opción 1: Docker (Recomendado) �?
```bash
# 1. Descomprimir
unzip arima-rl-project.zip
cd arima-rl-project

# 2. Dar permisos
chmod +x scripts/*.sh

# 3. Construir y ejecutar
./scripts/build_docker.sh build
./scripts/build_docker.sh run

# 4. Acceder
# http://localhost:8501
```

**Tiempo de instalación**: 5-10 minutos  
**Espacio requerido**: ~2 GB

### Opción 2: Instalación Local

```bash
# 1. Descomprimir
unzip arima-rl-project.zip
cd arima-rl-project

# 2. Crear entorno virtual
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Generar datos
python data/download_data.py

# 5. Ejecutar aplicación
streamlit run src/app.py
```

---

## 📚 Uso del Sistema

### 1️⃣ Exploración de Datos (Tab 1)

- 📊 Visualización de serie temporal completa
- 📈 Estadísticas descriptivas
- 🔍 Prueba de estacionariedad (ADF)
- 📉 Funciones ACF/PACF interactivas

### 2️⃣ Agente RL / Modo Manual (Tab 2)

#### 🤖 Modo Automático (Agente RL)

1. Clic en **"🎯 Predecir Mejor Configuración"**
2. El agente analiza la serie y propone (p, d, q) óptimo
3. Clic en **"▶️ Entrenar Modelo ARIMA Propuesto"**
4. Ver métricas (AIC, BIC, RMSE, MAE) y pronóstico

> ⚠️ **Nota**: Requiere modelo RL entrenado (ver sección **Entrenamiento**)

#### 🎛�?Modo Manual (Sliders)

1. Ajustar sliders p, d, q manualmente
2. Clic en **"🚀 Entrenar y Evaluar Modelo"**
3. Explorar diferentes configuraciones

### 3️⃣ Comparación de Modelos (Tab 3)

1. Configurar hasta 3 modelos ARIMA diferentes
2. Clic en **"📊 Comparar Modelos"**
3. Ver tabla ordenada por AIC (mejor resaltado)
4. Exportar resultados a CSV

### 4️⃣ Diagnóstico de Residuos (Tab 4)

- 📊 Gráficas de diagnóstico (residuos, histograma, Q-Q, ACF)
- 🔢 Estadísticas (media, desv. estándar, tests)
- �?Verificación de supuestos ARIMA

---

## 🎓 Entrenamiento del Agente RL

El agente RL **NO** se entrena automáticamente durante el build Docker (para reducir tiempo). Entrenar después de iniciar el sistema:

### Dentro de Docker

```bash
# Acceder al contenedor
docker exec -it arima-rl-container bash

# Entrenar
python -m src.rl_agent --train --timesteps 50000

# Salir
exit
```

### Instalación Local

```bash
# Opción A: Con script
chmod +x scripts/train_agent.sh
./scripts/train_agent.sh

# Opción B: Comando directo
python -m src.rl_agent --train --timesteps 50000

# Opción C: Entrenamiento rápido (pruebas)
python -m src.rl_agent --train --timesteps 10000
```

**Tiempos de entrenamiento**:
- 10k timesteps: ~5-10 minutos
- 50k timesteps: ~30-60 minutos

**Modelo guardado**: `models/arima_dqn_agent.zip`

---

## 📊 Dataset: Consumo Eléctrico Alemán

### Fuente de Datos

- **Origen**: [Open Power System Data (OPSD)](https://open-power-system-data.org/)
- **Período**: 2013-2017 (60 meses)
- **Frecuencia**: Mensual (agregado desde datos horarios)
- **Unidades**: GWh (Gigawatt-hora)

### División de Datos

| Conjunto   | Meses | Porcentaje | Uso                          |
|------------|-------|------------|------------------------------|
| Train      | 48    | 80%        | Entrenamiento ARIMA y RL     |
| Validation | 6     | 10%        | Selección de hiperparámetros |
| Test       | 6     | 10%        | Evaluación final             |

### Generación Automática

El script `data/download_data.py`:
1. �?Intenta descargar datos reales de OPSD
2. �?Si falla, genera datos sintéticos realistas
3. �?Convierte a frecuencia mensual
4. �?Divide en train/val/test
5. �?Guarda CSVs individuales

---

## ⚙️ Configuración Avanzada

### Archivo `config/config.yaml`

```yaml
# Agente RL
rl_agent:
  total_timesteps: 50000
  learning_rate: 0.0001
  buffer_size: 10000
  exploration_fraction: 0.3

# Entorno
environment:
  p_max: 5
  d_max: 2
  q_max: 4
  max_steps: 50
  reward_weights:
    accuracy: 1.0
    aic: 0.3
    time: 0.1
    diagnostics: 0.2

# ARIMA
arima:
  confidence_level: 0.95
  max_training_time: 30
```

---

## 📈 Métricas de Evaluación

### Calidad del Pronóstico

- **RMSE** (Root Mean Squared Error): Error cuadrático medio
- **MAE** (Mean Absolute Error): Error absoluto medio
- **MAPE** (Mean Absolute Percentage Error): Error porcentual
- **R²**: Coeficiente de determinación

### Selección de Modelo

- **AIC** (Akaike Information Criterion): Balance ajuste/complejidad
- **BIC** (Bayesian Information Criterion): Penaliza más la complejidad
- **AICc**: AIC corregido para muestras pequeñas

### Diagnóstico de Residuos

- **Test de Normalidad**: Jarque-Bera (p > 0.05)
- **Test de Autocorrelación**: Ljung-Box (p > 0.05)
- **Homocedasticidad**: Ratio de varianzas < 2.0
- **Media de residuos**: �?0

---

## 🔧 Comandos Útiles

### Docker

```bash
# Ver logs en tiempo real
docker logs -f arima-rl-container

# Detener contenedor
./scripts/build_docker.sh stop

# Limpiar todo (contenedor + imagen)
./scripts/build_docker.sh clean

# Estado del sistema
./scripts/build_docker.sh status

# Abrir shell
./scripts/build_docker.sh shell
```

### TensorBoard (Monitoreo de Entrenamiento)

```bash
tensorboard --logdir models/tensorboard_logs
# Acceder a http://localhost:6006
```

### Evaluar Agente Entrenado

```bash
python -m src.rl_agent --eval --model-path models/arima_dqn_agent.zip
```

---

## 📁 Estructura del Proyecto

```
arima-rl-project/
├── README.md                  # Este archivo
├── QUICKSTART.md             # Guía de inicio rápido
├── Dockerfile                # Containerización completa
├── requirements.txt          # Dependencias Python
├── .dockerignore            # Archivos excluidos de Docker
├── .gitignore               # Archivos excluidos de Git
�?├── data/
�?  ├── download_data.py     # Script de descarga/generación de datos
�?  ├── germany_monthly_power.csv    # 60 meses completos
�?  ├── train.csv            # 48 meses
�?  ├── validation.csv       # 6 meses
�?  ├── test.csv             # 6 meses
�?  └── metadata.txt         # Información del dataset
�?├── src/
�?  ├── __init__.py          # Inicialización del paquete
�?  ├── data_processor.py    # Procesamiento de series temporales
�?  ├── arima_env.py         # Entorno Gymnasium personalizado
�?  ├── rl_agent.py          # Agente DQN (Stable-Baselines3)
�?  ├── arima_utils.py       # Utilidades ARIMA
�?  └── app.py               # Interfaz web Streamlit
�?├── scripts/
�?  ├── build_docker.sh      # Construcción/ejecución Docker
�?  ├── run_app.sh           # Ejecución de aplicación
�?  └── train_agent.sh       # Entrenamiento del agente RL
�?├── config/
�?  └── config.yaml          # Configuración completa
�?├── assets/
�?  ├── style.css            # Estilos CSS personalizados
�?  ├── custom.js            # JavaScript personalizado
�?  └── logs/                # Logs de aplicación
�?└── models/                  # Modelos entrenados
    ├── arima_dqn_agent.zip  # Modelo RL principal
    └── tensorboard_logs/    # Logs de TensorBoard
```

---

## 🎯 Resultados Esperados

### Agente RL vs. Grid Search

| Método       | Configuraciones | Tiempo | AIC Óptimo | Convergencia |
|--------------|-----------------|--------|------------|--------------|
| Grid Search  | 60              | 100%   | Garantizado| N/A          |
| Agente RL    | ~30-40          | 30-50% | 95-98%     | 30k steps    |

### Mejoras Reportadas en Literatura

- **Hyp-RL (2019)**: RL supera optimización bayesiana con 50 datasets
- **ARIMA-LSTM (2023)**: Mejoras de 13% en MAE sobre modelos individuales
- **RLMC (AAAI 2022)**: Combinación dinámica de modelos con RL

---

## 🐛 Solución de Problemas

### Problema: Datos no encontrados

```bash
python data/download_data.py
```

### Problema: Modelo RL no encontrado

```bash
python -m src.rl_agent --train --timesteps 10000
```

### Problema: Puerto 8501 ocupado

```bash
# Cambiar puerto
PORT=8502 ./scripts/run_app.sh

# O liberar puerto (Linux/Mac)
lsof -ti:8501 | xargs kill -9
```

### Problema: Error de memoria durante entrenamiento

```yaml
# Editar config/config.yaml
rl_agent:
  buffer_size: 5000  # Reducir de 10000
```

---

## 📚 Referencias

### Papers Principales

1. **Hyp-RL**: Jomaa et al. (2019) - Hyperparameter Optimization by RL
2. **RLMC**: Fu et al. (2022) - RL Based Dynamic Model Combination
3. **ARIMA-LSTM**: Wang & Li (2023) - Peak Electrical Energy Consumption Prediction

### Documentación Técnica

- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)
- [Statsmodels ARIMA](https://www.statsmodels.org/stable/)
- [Streamlit](https://docs.streamlit.io/)
- [Gymnasium](https://gymnasium.farama.org/)

### Datos

- [Open Power System Data](https://open-power-system-data.org/)
- [PyPSA](https://pypsa.org/)

---

## 🤝 Contribuciones

Este proyecto es parte de un reporte técnico académico sobre "Agentificación de Modelos ARIMA con Aprendizaje Reforzado". Consulte el reporte PDF completo para fundamentos matemáticos detallados.

---

## 📄 Licencia

Este proyecto se distribuye bajo licencia MIT. Ver archivo `LICENSE` para más detalles.

---

## 📞 Soporte

Para preguntas o problemas:
1. Consulte `QUICKSTART.md` para guía rápida
2. Revise las instrucciones detalladas en `INSTRUCCIONES_DESPLIEGUE.txt`
3. Consulte el reporte técnico PDF para fundamentos

---

## �?Características Futuras (Roadmap)

- [ ] Soporte para SARIMA (estacionalidad)
- [ ] Variables exógenas (ARIMAX/SARIMAX)
- [ ] Ensemble RL-ARIMA-LSTM
- [ ] Meta-learning cross-dataset
- [ ] Optimización multi-objetivo (NSGA-II)
- [ ] API REST para integración
- [ ] Dashboard de monitoreo en tiempo real

---

**Desarrollado con ❤️ para optimización automática de series temporales mediante Aprendizaje Reforzado**
