# 🚀 QUICKSTART - Agente RL-ARIMA

Guía de inicio rápido en cinco (5) minutos.

## ⚡ Instalación Express (Docker)

```bash
unzip arima-rl-project.zip
cd arima-rl-project
chmod +x scripts/*.sh
./scripts/build_docker.sh build
./scripts/build_docker.sh run
```

Acceder: **http://localhost:8501**

## 📋 Primeros Pasos

### 1. Explorar Datos (Tab 1)
- Ver serie temporal de 60 meses
- Revisar estadísticas
- Analizar estacionariedad

### 2. Modo Manual (Tab 2)
- Ajustar sliders p, d, q
- Entrenar modelo ARIMA
- Ver pronóstico

### 3. Entrenar Agente RL (Opcional)

```bash
docker exec -it arima-rl-container python -m src.rl_agent --train --timesteps 50000
```

Tiempo: ~30-60 minutos

### 4. Usar Modo Automático
- Clic en "🎯 Predecir Mejor Configuración"
- Entrenar modelo propuesto

## 🔧 Comandos Útiles

```bash
# Ver logs
docker logs -f arima-rl-container

# Detener
./scripts/build_docker.sh stop

# Reiniciar
./scripts/build_docker.sh run
```

## 📊 Qué Esperar

- **Datos**: 60 meses de consumo eléctrico alemán
- **División**: 48 train + 6 val + 6 test
- **Modelos**: ARIMA con múltiples configuraciones
- **Agente RL**: Propone hiperparámetros óptimos

## 🆘 Problemas Comunes

**Puerto ocupado**: `PORT=8502 ./scripts/run_app.sh`  
**Sin datos**: `python data/download_data.py`  
**Sin modelo RL**: Modo Manual funciona sin entrenamiento

## 📚 Más Info

Ver `README.md` para documentación completa.
