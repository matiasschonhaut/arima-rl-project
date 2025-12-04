#!/bin/bash
# ============================================================================
# Proyecto: Agentificación de Modelos ARIMA con Aprendizaje Reforzado
# Archivo: run_app.sh
# Descripción: Script para ejecutar aplicación Streamlit
# ============================================================================

set -e

# Colores
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}🚀 Iniciando Aplicación RL-ARIMA${NC}"
echo -e "${BLUE}================================${NC}"
echo ""

# Verificar que existan los datos
if [ ! -f "data/germany_monthly_power.csv" ]; then
    echo -e "${YELLOW}⚠️  Datos no encontrados. Descargando...${NC}"
    python data/download_data.py
    echo ""
fi

# Puerto (puede ser sobrescrito por variable de entorno)
PORT=${PORT:-8501}

echo -e "${GREEN}✅ Datos disponibles${NC}"
echo -e "${BLUE}📡 Puerto: $PORT${NC}"
echo ""
echo -e "${YELLOW}Iniciando Streamlit...${NC}"
echo ""

# Ejecutar Streamlit
streamlit run src/app.py \
    --server.port=$PORT \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --browser.gatherUsageStats=false
