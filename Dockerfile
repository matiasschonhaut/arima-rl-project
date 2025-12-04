# =============================================================================
# Proyecto: Agentificación de Modelos ARIMA con Aprendizaje Reforzado
# Imagen Docker FINAL (modelo preentrenado)
# =============================================================================

FROM python:3.10-slim

LABEL maintainer="Tomás Stevenson & Matías Schonhaut"
LABEL description="Optimización de Modelos ARIMA vía RL (pretrained agent)"
LABEL version="2.0.0"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    APP_HOME=/app \
    PORT=8501

# ============================================================================  
# Dependencias de sistema para statsmodels, pmdarima y PyTorch
# ============================================================================
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    gfortran \
    libblas-dev \
    liblapack-dev \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR $APP_HOME

# ============================================================================  
# Instalar dependencias Python
# ============================================================================
COPY requirements.txt .

# PyTorch CPU
RUN pip install --upgrade pip && \
    pip install torch==2.1.0+cpu torchvision==0.16.0+cpu --extra-index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r requirements.txt

# ============================================================================  
# Copiar código
# ============================================================================
RUN mkdir -p src data config assets models

COPY src/ src/
COPY data/ data/
COPY config/ config/
COPY assets/ assets/

# Copiar modelo RL entrenado
COPY models/arima_dqn_agent.zip models/arima_dqn_agent.zip

# Permisos si fueran necesarios
RUN chmod +x data/download_data.py || true

# Generar dataset automáticamente
RUN python data/download_data.py

# ============================================================================  
# Configurar Streamlit
# ============================================================================
EXPOSE $PORT

HEALTHCHECK --interval=20s --timeout=10s --start-period=20s --retries=3 \
    CMD curl --fail http://localhost:$PORT/_stcore/health || exit 1

# Ejecutar Streamlit
CMD ["streamlit", "run", "src/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
