// ============================================================================
// Proyecto: Agentificación de Modelos ARIMA con Aprendizaje Reforzado
// Archivo: custom.js
// Descripción: Funcionalidades JavaScript personalizadas
// ============================================================================

(function() {
    'use strict';
    
    // ========================================================================
    // INICIALIZACIÓN
    // ========================================================================
    
    console.log('🤖 RL-ARIMA App Initialized');
    
    // ========================================================================
    // ANIMACIONES Y TRANSICIONES
    // ========================================================================
    
    /**
     * Añade animación de fade-in a elementos cuando se cargan
     */
    function addFadeInAnimation() {
        const elements = document.querySelectorAll('.stMarkdown, .stDataFrame, .stPlotlyChart');
        elements.forEach((el, index) => {
            setTimeout(() => {
                el.classList.add('fade-in');
            }, index * 100);
        });
    }
    
    /**
     * Smooth scroll para navegación
     */
    function enableSmoothScroll() {
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {
            anchor.addEventListener('click', function (e) {
                e.preventDefault();
                const target = document.querySelector(this.getAttribute('href'));
                if (target) {
                    target.scrollIntoView({
                        behavior: 'smooth',
                        block: 'start'
                    });
                }
            });
        });
    }
    
    // ========================================================================
    // MEJORAS DE TABLAS
    // ========================================================================
    
    /**
     * Resalta la mejor fila en tablas comparativas
     */
    function highlightBestRow() {
        const tables = document.querySelectorAll('.dataframe');
        tables.forEach(table => {
            const rows = table.querySelectorAll('tbody tr');
            if (rows.length > 0) {
                rows[0].classList.add('best-row');
            }
        });
    }
    
    /**
     * Añade tooltip a celdas de tabla
     */
    function addTableTooltips() {
        const cells = document.querySelectorAll('.dataframe td');
        cells.forEach(cell => {
            cell.setAttribute('title', cell.textContent);
        });
    }
    
    // ========================================================================
    // MEJORAS DE BOTONES
    // ========================================================================
    
    /**
     * Añade efecto de loading a botones
     */
    function addButtonLoadingEffect() {
        const buttons = document.querySelectorAll('.stButton > button');
        buttons.forEach(button => {
            button.addEventListener('click', function() {
                this.classList.add('loading');
                this.disabled = true;
                
                // Simular proceso (en realidad Streamlit maneja esto)
                setTimeout(() => {
                    this.classList.remove('loading');
                    this.disabled = false;
                }, 500);
            });
        });
    }
    
    // ========================================================================
    // MEJORAS DE GRÁFICAS PLOTLY
    // ========================================================================
    
    /**
     * Mejora interactividad de gráficas Plotly
     */
    function enhancePlotlyCharts() {
        // Configuración global de Plotly
        if (typeof Plotly !== 'undefined') {
            Plotly.setPlotConfig({
                displayModeBar: true,
                displaylogo: false,
                modeBarButtonsToRemove: ['sendDataToCloud', 'lasso2d', 'select2d'],
                toImageButtonOptions: {
                    format: 'png',
                    filename: 'arima_chart',
                    height: 800,
                    width: 1200,
                    scale: 2
                }
            });
        }
    }
    
    // ========================================================================
    // NOTIFICACIONES
    // ========================================================================
    
    /**
     * Muestra notificación temporal
     */
    function showNotification(message, type = 'info') {
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.textContent = message;
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 15px 20px;
            border-radius: 8px;
            background-color: ${type === 'success' ? '#2ca02c' : type === 'error' ? '#d62728' : '#1f77b4'};
            color: white;
            font-weight: 600;
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
            z-index: 9999;
            animation: slideInRight 0.3s ease-out;
        `;
        
        document.body.appendChild(notification);
        
        setTimeout(() => {
            notification.style.animation = 'slideOutRight 0.3s ease-out';
            setTimeout(() => {
                document.body.removeChild(notification);
            }, 300);
        }, 3000);
    }
    
    // ========================================================================
    // TECLADO Y ATAJOS
    // ========================================================================
    
    /**
     * Añade atajos de teclado
     */
    function addKeyboardShortcuts() {
        document.addEventListener('keydown', function(e) {
            // Ctrl/Cmd + K: Focus en búsqueda
            if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
                e.preventDefault();
                const searchInput = document.querySelector('input[type="text"]');
                if (searchInput) searchInput.focus();
            }
            
            // Ctrl/Cmd + R: Refrescar (prevenir comportamiento por defecto)
            if ((e.ctrlKey || e.metaKey) && e.key === 'r') {
                // Streamlit maneja esto automáticamente
            }
        });
    }
    
    // ========================================================================
    // ALMACENAMIENTO LOCAL
    // ========================================================================
    
    /**
     * Guarda preferencias del usuario
     */
    function saveUserPreferences() {
        const preferences = {
            theme: 'light',
            lastVisit: new Date().toISOString(),
            viewCount: (parseInt(localStorage.getItem('viewCount')) || 0) + 1
        };
        
        Object.keys(preferences).forEach(key => {
            localStorage.setItem(key, preferences[key]);
        });
    }
    
    /**
     * Carga preferencias del usuario
     */
    function loadUserPreferences() {
        const viewCount = localStorage.getItem('viewCount');
        if (viewCount) {
            console.log(`👋 Bienvenido de vuelta! Visita #${viewCount}`);
        }
    }
    
    // ========================================================================
    // ANÁLISIS Y TRACKING (OPCIONAL)
    // ========================================================================
    
    /**
     * Track eventos de usuario (solo para desarrollo)
     */
    function trackEvent(category, action, label) {
        if (window.location.hostname === 'localhost') {
            console.log(`📊 Event: ${category} / ${action} / ${label}`);
        }
    }
    
    // ========================================================================
    // UTILIDADES
    // ========================================================================
    
    /**
     * Formatea números con separadores de miles
     */
    function formatNumber(num) {
        return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
    }
    
    /**
     * Copia texto al portapapeles
     */
    function copyToClipboard(text) {
        navigator.clipboard.writeText(text).then(() => {
            showNotification('✅ Copiado al portapapeles', 'success');
        }).catch(() => {
            showNotification('❌ Error al copiar', 'error');
        });
    }
    
    /**
     * Detecta tema oscuro del sistema
     */
    function detectDarkMode() {
        return window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
    }
    
    // ========================================================================
    // OBSERVADOR DE MUTACIONES (para elementos dinámicos de Streamlit)
    // ========================================================================
    
    const observer = new MutationObserver(function(mutations) {
        mutations.forEach(function(mutation) {
            if (mutation.addedNodes.length) {
                highlightBestRow();
                addTableTooltips();
            }
        });
    });
    
    // Observar cambios en el DOM
    observer.observe(document.body, {
        childList: true,
        subtree: true
    });
    
    // ========================================================================
    // INICIALIZACIÓN AL CARGAR LA PÁGINA
    // ========================================================================
    
    document.addEventListener('DOMContentLoaded', function() {
        console.log('📄 DOM Content Loaded');
        
        addFadeInAnimation();
        enableSmoothScroll();
        highlightBestRow();
        addTableTooltips();
        addButtonLoadingEffect();
        enhancePlotlyCharts();
        addKeyboardShortcuts();
        saveUserPreferences();
        loadUserPreferences();
        
        // Track página vista
        trackEvent('Page', 'View', window.location.pathname);
    });
    
    // ========================================================================
    // MANEJO DE ERRORES GLOBAL
    // ========================================================================
    
    window.addEventListener('error', function(e) {
        console.error('❌ Error global:', e.message);
        // Aquí podrías enviar errores a un servicio de logging
    });
    
    // ========================================================================
    // EXPORTAR FUNCIONES PÚBLICAS
    // ========================================================================
    
    window.ARIMAApp = {
        showNotification,
        copyToClipboard,
        formatNumber,
        trackEvent
    };
    
})();

// ============================================================================
// CSS PARA ANIMACIONES (se inyecta dinámicamente)
// ============================================================================

const style = document.createElement('style');
style.textContent = `
    @keyframes slideInRight {
        from {
            transform: translateX(100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    @keyframes slideOutRight {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(100%);
            opacity: 0;
        }
    }
    
    .loading {
        position: relative;
        pointer-events: none;
    }
    
    .loading::after {
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        width: 20px;
        height: 20px;
        margin: -10px 0 0 -10px;
        border: 3px solid rgba(255, 255, 255, 0.3);
        border-top-color: white;
        border-radius: 50%;
        animation: spin 0.8s linear infinite;
    }
    
    @keyframes spin {
        to { transform: rotate(360deg); }
    }
`;

document.head.appendChild(style);
