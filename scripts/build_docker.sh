#!/bin/bash
# ============================================================================
# Proyecto: Agentificación de Modelos ARIMA con Aprendizaje Reforzado
# Archivo: build_docker.sh
# Descripción: Script para construir y ejecutar el contenedor Docker
# ============================================================================

set -e

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Variables
IMAGE_NAME="arima-rl-agent"
CONTAINER_NAME="arima-rl-container"
PORT=8501

# Funciones
print_header() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ️  $1${NC}"
}

# Función: build
build_image() {
    print_header "CONSTRUYENDO IMAGEN DOCKER"
    
    echo "Imagen: $IMAGE_NAME"
    echo "Puerto: $PORT"
    echo ""
    
    docker build -t $IMAGE_NAME . || {
        print_error "Fallo al construir imagen Docker"
        exit 1
    }
    
    print_success "Imagen Docker construida exitosamente"
    echo ""
    docker images | grep $IMAGE_NAME
}

# Función: run
run_container() {
    print_header "EJECUTANDO CONTENEDOR"
    
    # Verificar si ya existe un contenedor con el mismo nombre
    if [ "$(docker ps -aq -f name=$CONTAINER_NAME)" ]; then
        print_info "Deteniendo contenedor existente..."
        docker stop $CONTAINER_NAME 2>/dev/null || true
        docker rm $CONTAINER_NAME 2>/dev/null || true
    fi
    
    echo "Iniciando contenedor..."
    docker run -d \
        --name $CONTAINER_NAME \
        -p $PORT:8501 \
        -v "$(pwd)/models:/app/models" \
        -v "$(pwd)/data:/app/data" \
        --health-cmd="curl --fail http://localhost:8501/_stcore/health || exit 1" \
        --health-interval=30s \
        --health-timeout=10s \
        --health-retries=3 \
        $IMAGE_NAME || {
            print_error "Fallo al ejecutar contenedor"
            exit 1
        }
    
    print_success "Contenedor iniciado exitosamente"
    echo ""
    print_info "Esperando que el contenedor esté saludable..."
    sleep 5
    
    # Mostrar logs
    echo ""
    print_info "Logs del contenedor:"
    docker logs $CONTAINER_NAME
    
    echo ""
    print_success "Aplicación disponible en: http://localhost:$PORT"
    print_info "Ver logs en tiempo real: docker logs -f $CONTAINER_NAME"
}

# Función: stop
stop_container() {
    print_header "DETENIENDO CONTENEDOR"
    
    if [ "$(docker ps -q -f name=$CONTAINER_NAME)" ]; then
        docker stop $CONTAINER_NAME
        print_success "Contenedor detenido"
    else
        print_info "Contenedor no está corriendo"
    fi
}

# Función: clean
clean_all() {
    print_header "LIMPIANDO TODO"
    
    # Detener y eliminar contenedor
    if [ "$(docker ps -aq -f name=$CONTAINER_NAME)" ]; then
        print_info "Eliminando contenedor..."
        docker stop $CONTAINER_NAME 2>/dev/null || true
        docker rm $CONTAINER_NAME 2>/dev/null || true
        print_success "Contenedor eliminado"
    fi
    
    # Eliminar imagen
    if [ "$(docker images -q $IMAGE_NAME)" ]; then
        print_info "Eliminando imagen..."
        docker rmi $IMAGE_NAME
        print_success "Imagen eliminada"
    fi
    
    print_success "Limpieza completada"
}

# Función: logs
show_logs() {
    print_header "LOGS DEL CONTENEDOR"
    
    if [ "$(docker ps -q -f name=$CONTAINER_NAME)" ]; then
        docker logs -f $CONTAINER_NAME
    else
        print_error "Contenedor no está corriendo"
        exit 1
    fi
}

# Función: status
show_status() {
    print_header "ESTADO DEL SISTEMA"
    
    echo "Imagen:"
    docker images | grep $IMAGE_NAME || print_info "Imagen no encontrada"
    
    echo ""
    echo "Contenedor:"
    docker ps -a | grep $CONTAINER_NAME || print_info "Contenedor no encontrado"
    
    echo ""
    echo "Health Check:"
    docker inspect --format='{{json .State.Health}}' $CONTAINER_NAME 2>/dev/null | python3 -m json.tool || print_info "Contenedor no está corriendo"
}

# Función: shell
open_shell() {
    print_header "ABRIENDO SHELL EN CONTENEDOR"
    
    if [ "$(docker ps -q -f name=$CONTAINER_NAME)" ]; then
        docker exec -it $CONTAINER_NAME /bin/bash
    else
        print_error "Contenedor no está corriendo"
        exit 1
    fi
}

# Main script
show_usage() {
    echo "Uso: $0 [COMANDO]"
    echo ""
    echo "Comandos disponibles:"
    echo "  build   - Construir imagen Docker"
    echo "  run     - Ejecutar contenedor"
    echo "  stop    - Detener contenedor"
    echo "  clean   - Limpiar contenedor e imagen"
    echo "  logs    - Mostrar logs en tiempo real"
    echo "  status  - Mostrar estado del sistema"
    echo "  shell   - Abrir shell en contenedor"
    echo ""
    echo "Ejemplos:"
    echo "  $0 build"
    echo "  $0 run"
    echo "  $0 stop"
}

# Procesar comandos
case "$1" in
    build)
        build_image
        ;;
    run)
        run_container
        ;;
    stop)
        stop_container
        ;;
    clean)
        clean_all
        ;;
    logs)
        show_logs
        ;;
    status)
        show_status
        ;;
    shell)
        open_shell
        ;;
    *)
        show_usage
        exit 1
        ;;
esac
