import streamlit as st
import math
import matplotlib.pyplot as plt
from PIL import Image
import io

# Configuración de la página
st.set_page_config(
    page_title="Herramienta de Suelos ITCR",
    page_icon="🌍",
    layout="wide"
)

# Logo y título
col1, col2 = st.columns([1, 4])
with col1:
    st.image("https://www.itcr.ac.cr/wp-content/themes/portal-itcr/images/logo.png", width=100)
with col2:
    st.title("Herramienta de Análisis de Suelos")
    st.caption("Proyecto Final - ITCR")

# Menú de pestañas
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📐 Carga Circular", 
    "🟦 Rectangular", 
    "• Puntual", 
    "📏 Lineal", 
    "🔺 Trapezoidal", 
    "⚖️ Esfuerzos", 
    "📉 Consolidación"
])

# ========== PESTAÑA CARGA CIRCULAR ==========
with tab1:
    st.header("Carga Circular")
    col1, col2 = st.columns(2)
    
    with col1:
        R = st.number_input("Radio R (m)", value=5.0, min_value=0.1)
        z = st.number_input("Profundidad z (m)", value=2.0, min_value=0.1)
        q = st.number_input("Carga q (kPa)", value=100.0)
        E = st.number_input("Módulo Elasticidad E (kPa)", value=15000.0)
        v = st.number_input("Coeficiente Poisson ν", value=0.25, min_value=0.0, max_value=0.5, step=0.01)
        
        if st.button("Calcular", key="calc_circular"):
            Iz = (1 - 1/((1 + (R/z)**2)**1.5))**2
            sigma_z = q * Iz
            asiento = (q * R * (1 - v**2)) / E * 1000
            
            st.success(f"""
            **RESULTADOS:**  
            Factor de influencia (Iz): `{Iz:.4f}`  
            Esfuerzo vertical (σz): `{sigma_z:.2f} kPa`  
            Asentamiento estimado: `{asiento:.2f} mm`
            """)
    
    with col2:
        # Gráfico interactivo
        profundidades = [i/10 for i in range(1, 101)]
        factores = [(1 - 1/((1 + (R/z)**2)**1.5))**2 for z in profundidades]
        
        fig, ax = plt.subplots()
        ax.plot(profundidades, factores, 'b-', linewidth=2)
        ax.set_title("Variación del Factor de Influencia con la Profundidad")
        ax.set_xlabel("Profundidad z (m)")
        ax.set_ylabel("Factor de Influencia Iz")
        ax.grid(True)
        st.pyplot(fig)

# ========== PESTAÑA ESFUERZOS ========== 
with tab6:
    st.header("Cálculo de Esfuerzos por Estratos")
    
    # Configuración inicial
    col1, col2 = st.columns(2)
    with col1:
        nf = st.number_input("Nivel Freático (m)", value=2.0, min_value=0.0)
        unidad_gamma = st.selectbox("Unidad peso específico", ["kN/m³", "kg/m³"])
    
    # Lista para almacenar estratos
    if 'estratos' not in st.session_state:
        st.session_state.estratos = [{'espesor': 1.0, 'gamma': 18.0, 'k': 1e-5}]
    
    # Interfaz para agregar estratos
    with st.expander("🏗️ Configuración de Estratos"):
        for i, estrato in enumerate(st.session_state.estratos):
            cols = st.columns(3)
            with cols[0]:
                st.session_state.estratos[i]['espesor'] = st.number_input(
                    f"Espesor Estrato {i+1} (m)", 
                    value=estrato['espesor'], 
                    min_value=0.1,
                    key=f"espesor_{i}"
                )
            with cols[1]:
                st.session_state.estratos[i]['gamma'] = st.number_input(
                    f"γ Estrato {i+1}", 
                    value=estrato['gamma'], 
                    min_value=0.1,
                    key=f"gamma_{i}"
                )
            with cols[2]:
                st.session_state.estratos[i]['k'] = st.number_input(
                    f"k Estrato {i+1} (m/s)", 
                    value=estrato['k'], 
                    format="%e",
                    key=f"k_{i}"
                )
        
        if st.button("➕ Agregar Estrato"):
            st.session_state.estratos.append({'espesor': 1.0, 'gamma': 18.0, 'k': 1e-5})
            st.rerun()
    
    # Cálculo y resultados
    if st.button("📊 Calcular Esfuerzos"):
        profundidad_acum = 0
        esfuerzo_total = 0
        resultados = []
        
        for estrato in st.session_state.estratos:
            espesor = estrato['espesor']
            gamma = estrato['gamma']
            k = estrato['k']
            
            if unidad_gamma == "kg/m³":
                gamma /= 1000  # Conversión a kN/m³
            
            # Cálculos
            esfuerzo_total += gamma * espesor
            u = 9.81 * min(profundidad_acum + espesor, nf)
            esfuerzo_efectivo = esfuerzo_total - u
            
            resultados.append({
                'profundidad': f"{profundidad_acum:.2f}-{profundidad_acum + espesor:.2f} m",
                'esfuerzo_total': esfuerzo_total,
                'presion_poros': u,
                'esfuerzo_efectivo': esfuerzo_efectivo,
                'permeabilidad': k
            })
            
            profundidad_acum += espesor
        
        # Mostrar resultados en tabla
        st.dataframe(
            resultados,
            column_config={
                'profundidad': "Profundidad",
                'esfuerzo_total': st.column_config.NumberColumn("Esf. Total (kPa)", format="%.2f"),
                'presion_poros': st.column_config.NumberColumn("Pres. Poros (kPa)", format="%.2f"),
                'esfuerzo_efectivo': st.column_config.NumberColumn("Esf. Efectivo (kPa)", format="%.2f"),
                'permeabilidad': st.column_config.NumberColumn("k (m/s)", format="%.2e")
            },
            hide_index=True
        )
        
        # Gráfico de esfuerzos
        fig, ax = plt.subplots(figsize=(6, 8))
        profundidades = [0]
        sigmas = [0]
        u = [0]
        sigma_eff = [0]
        
        current_depth = 0
        for estrato in st.session_state.estratos:
            current_depth += estrato['espesor']
            profundidades.append(current_depth)
            sigmas.append(sigmas[-1] + estrato['gamma'] * estrato['espesor'])
            u.append(9.81 * min(current_depth, nf))
            sigma_eff.append(sigmas[-1] - u[-1])
        
        ax.plot(sigmas, profundidades, 'r-', label='Esfuerzo Total')
        ax.plot(u, profundidades, 'b-', label='Presión de Poros')
        ax.plot(sigma_eff, profundidades, 'g--', label='Esfuerzo Efectivo')
        
        ax.set_ylim(max(profundidades), 0)  # Invertir eje Y
        ax.set_xlabel("Esfuerzo (kPa)")
        ax.set_ylabel("Profundidad (m)")
        ax.legend()
        ax.grid(True)
        ax.set_title("Distribución de Esfuerzos Verticales")
        st.pyplot(fig)

# ========== PESTAÑA CONSOLIDACIÓN ==========
with tab7:
    st.header("Cálculo de Consolidación Primaria")
    
    col1, col2 = st.columns(2)
    with col1:
        h = st.number_input("Altura de drenaje (m)", value=5.0, min_value=0.1)
        cv = st.number_input("Coeficiente consolidación Cv (m²/s)", value=1e-7, format="%e")
        u = st.slider("Grado consolidación U (%)", 1, 99, 50)
    
    if st.button("⏱️ Calcular Tiempo"):
        if u < 60:
            tv = (math.pi/4) * (u/100)**2
        else:
            tv = 1.781 - 0.933 * math.log(1 - u/100)
        
        t = tv * h**2 / cv
        
        # Conversión a unidades legibles
        if t < 60:
            tiempo = f"{t:.2f} segundos"
        elif t < 3600:
            tiempo = f"{t/60:.2f} minutos"
        elif t < 86400:
            tiempo = f"{t/3600:.2f} horas"
        else:
            tiempo = f"{t/86400:.2f} días"
        
        st.success(f"""
        **RESULTADOS:**  
        - Factor tiempo (Tv): `{tv:.4f}`  
        - Tiempo requerido: `{tiempo}`  
        - Equivalente: `{t:.2f} segundos`
        """)
    
    with col2:
        # Gráfico de consolidación
        u_values = list(range(1, 100))
        tv_values = []
        for u_val in u_values:
            if u_val < 60:
                tv = (math.pi/4) * (u_val/100)**2
            else:
                tv = 1.781 - 0.933 * math.log(1 - u_val/100)
            tv_values.append(tv)
        
        fig, ax = plt.subplots()
        ax.plot(u_values, tv_values, 'b-')
        ax.axvline(x=u, color='r', linestyle='--')
        ax.set_title("Curva Teórica de Consolidación")
        ax.set_xlabel("Grado de Consolidación U (%)")
        ax.set_ylabel("Factor Tiempo Tv")
        ax.grid(True)
        st.pyplot(fig)

# ========== NOTAS FINALES ==========
st.sidebar.markdown("""
### 🚀 Cómo usar esta herramienta:
1. Selecciona la pestaña del tipo de carga
2. Ingresa los parámetros requeridos
3. Haz clic en "Calcular"
4. Visualiza resultados y gráficos

### 📌 Para tu exposición:
- Usa pantalla completa (F11)
- Cambia valores durante la demo
- Explica los gráficos generados
""")

st.sidebar.divider()
st.sidebar.caption("Proyecto desarrollado por [Tu Nombre] para ITCR")