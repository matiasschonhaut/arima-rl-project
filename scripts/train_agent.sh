#!/bin/bash
# ============================================================================
# Proyecto: Agentificación de Modelos ARIMA con Aprendizaje Reforzado
# Archivo: train_agent.sh
# Descripción: Script para entrenar agente RL
# ============================================================================

set -e

# Colores
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}🎓 Entrenamiento Agente RL-ARIMA${NC}"
echo -e "${BLUE}================================${NC}"
echo ""

# Verificar que existan los datos
if [ ! -f "data/germany_monthly_power.csv" ]; then
    echo -e "${RED}❌ Datos no encontrados${NC}"
    echo -e "${YELLOW}💡 Ejecute primero: python data/download_data.py${NC}"
    exit 1
fi

# Parámetros de entrenamiento
DATA_PATH=${DATA_PATH:-"data/germany_monthly_power.csv"}
TIMESTEPS=${TIMESTEPS:-50000}
OUTPUT_DIR=${OUTPUT_DIR:-"models"}

echo -e "${GREEN}⚙️  Configuración de Entrenamiento:${NC}"
echo -e "   Datos: $DATA_PATH"
echo -e "   Timesteps: $TIMESTEPS"
echo -e "   Directorio salida: $OUTPUT_DIR"
echo ""

# Crear directorio de salida
mkdir -p $OUTPUT_DIR

echo -e "${YELLOW}🚀 Iniciando entrenamiento...${NC}"
echo -e "${YELLOW}   (Esto puede tardar 30-60 minutos)${NC}"
echo ""

# Entrenar agente
python -m src.rl_agent \
    --train \
    --data "$DATA_PATH" \
    --timesteps $TIMESTEPS \
    --output-dir "$OUTPUT_DIR"

# Verificar que el modelo se haya guardado
if [ -f "$OUTPUT_DIR/arima_dqn_agent.zip" ]; then
    echo ""
    echo -e "${GREEN}================================${NC}"
    echo -e "${GREEN}✅ Entrenamiento completado${NC}"
    echo -e "${GREEN}================================${NC}"
    echo ""
    echo -e "${BLUE}📦 Modelo guardado en: $OUTPUT_DIR/arima_dqn_agent.zip${NC}"
    echo -e "${BLUE}📊 Logs de TensorBoard: $OUTPUT_DIR/tensorboard_logs${NC}"
    echo ""
    echo -e "${YELLOW}💡 Siguiente paso:${NC}"
    echo -e "   Ejecute la aplicación web: ./scripts/run_app.sh"
    echo -e "   O visualice logs: tensorboard --logdir $OUTPUT_DIR/tensorboard_logs"
else
    echo ""
    echo -e "${RED}================================${NC}"
    echo -e "${RED}❌ Error en el entrenamiento${NC}"
    echo -e "${RED}================================${NC}"
    echo -e "${YELLOW}💡 Revise los logs arriba para más detalles${NC}"
    exit 1
fi
