import streamlit as st
import math
import matplotlib.pyplot as plt
import numpy as np

# Configuración de la página
st.set_page_config(
    page_title="Herramienta de Suelos ITCR",
    page_icon="🌍",
    layout="wide"
)

# Logo y título
col1, col2 = st.columns([1, 4])
with col1:
    st.image("https://plazareal.co.cr/wp-content/uploads/2018/05/Logo-TEC-Pagina-Interna.png", width=100)
with col2:
    st.title("Herramienta de Análisis de Suelos")
    st.caption("Proyecto Mecánica de Suelos - ITCR")
    st.caption("Armando-Jimena-Luis Esteban-Ricardo-Gabriel")     

# Menú de pestañas
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📐 Circular", 
    "🟦 Rectangular", 
    "• Puntual", 
    "📏 Lineal", 
    "🔺 Trapezoidal", 
    "⚖️ Esfuerzos", 
    "📉 Consolidación"
])

# ========== FUNCIONES COMPARTIDAS ==========
def plot_variation(x, y, x_label, y_label, title):
    fig, ax = plt.subplots()
    ax.plot(x, y, 'b-', linewidth=2)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True)
    st.pyplot(fig)

# ========== PESTAÑA CARGA CIRCULAR ==========
with tab1:
    st.header("Carga Circular")
    col1, col2 = st.columns(2)
    
    with col1:
        R = st.number_input("Radio R (m)", value=5.0, min_value=0.1, key="circ_R")
        z = st.number_input("Profundidad z (m)", value=2.0, min_value=0.1, key="circ_z")
        q = st.number_input("Carga q (kPa)", value=100.0, key="circ_q")
        
        # Selector de posición
        posicion = st.radio("Posición del punto:", ["Centro", "Fuera del centro"], key="circ_pos")
        
        # Solo mostrar entrada de distancia si no está en el centro
        if posicion == "Fuera del centro":
            r = st.number_input("Distancia radial desde el centro (m)", value=1.0, min_value=0.0, key="circ_r")
            I = st.number_input("Factor de influencia I (de tablas)", value=0.5, min_value=0.0, max_value=1.0, step=0.01, key="circ_I")
        
        if st.button("Calcular", key="calc_circular"):
            if posicion == "Centro":
                # Fórmula exacta para centro
                sigma_z = q * (1 - (z**3 / (R**2 + z**2)**1.5))
                formula_usada = "σz = q * [1 - (z³/(R²+z²)^(3/2))]"
            else:
                # Factor de influencia para puntos fuera del centro
                sigma_z = q * I
                formula_usada = f"σz = q * I = {q} * {I}"
            
            st.success(f"""
            **RESULTADOS:**  
            • Fórmula aplicada: `{formula_usada}`  
            • Esfuerzo vertical (σz): `{sigma_z:.2f} kPa`
            """)
    
    with col2:
        if 'calc_circular' in st.session_state:
            # Gráfico de variación con profundidad (solo para centro)
            if posicion == "Centro":
                profundidades = np.linspace(0.1, 3*z, 50)
                esfuerzos = [q * (1 - (zi**3 / (R**2 + zi**2)**1.5)) for zi in profundidades]
                
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.plot(profundidades, esfuerzos, 'b-')
                ax.axvline(x=z, color='r', linestyle='--', label=f'Profundidad calculada ({z}m)')
                ax.set_title("Variación de σz con Profundidad (Centro)")
                ax.set_xlabel("Profundidad z (m)")
                ax.set_ylabel("Esfuerzo σz (kPa)")
                ax.legend()
                ax.grid(True)
                st.pyplot(fig)
            
            # Diagrama esquemático
            fig2 = plt.figure(figsize=(8, 6))
            ax2 = fig2.add_subplot(111)
            
            # Dibujar círculo de carga
            circle = plt.Circle((0, 0), R, color='r', fill=True, alpha=0.3, label='Área cargada')
            ax2.add_patch(circle)
            
            # Marcar punto de cálculo
            if posicion == "Centro":
                ax2.plot(0, -z, 'bo', markersize=10, label='Punto en centro')
            else:
                ax2.plot(r, -z, 'bo', markersize=10, label='Punto fuera del centro')
            
            ax2.set_xlim(-1.5*R, 1.5*R)
            ax2.set_ylim(-3*z, 0.5*R)
            ax2.set_aspect('equal')
            ax2.set_title("Esquema de Carga Circular")
            ax2.set_xlabel("Distancia radial (m)")
            ax2.set_ylabel("Profundidad (m)")
            ax2.legend()
            ax2.grid(True)
            st.pyplot(fig2)

# ========== PESTAÑA CARGA RECTANGULAR ==========
with tab2:
    st.header("Carga Rectangular")
    col1, col2 = st.columns(2)
    
    with col1:
        # Inputs con valores del Excel (B=2, L=6, z=2.5, q=35)
        B = st.number_input("Ancho B (m)", value=2.0, min_value=0.1, key="rect_B")
        L = st.number_input("Largo L (m)", value=6.0, min_value=0.1, key="rect_L")
        z = st.number_input("Profundidad z (m)", value=2.5, min_value=0.1, key="rect_z")
        q = st.number_input("Carga q (kPa)", value=35.0, key="rect_q")
        posicion = st.radio("Posición del punto:", ["Esquina", "Centro"], index=0, key="rect_pos")
        
        if st.button("Calcular", key="calc_rect"):
            # Cálculo exacto como en el Excel
            m = B/z
            n = L/z
            
            if posicion == "Esquina":
                # Fórmula de Giroud (1968) exactamente como en el Excel
                Iz = (1/math.pi)*(math.log(m + math.sqrt(1 + m**2)) + m*math.log((1 + math.sqrt(1 + m**2))/m))
            else:
                # Para centro: Is(centro) = 2*Is(esquina) como en la nota del Excel
                Iz_esquina = (1/math.pi)*(math.log(m + math.sqrt(1 + m**2)) + m*math.log((1 + math.sqrt(1 + m**2))/m))
                Iz = 2 * Iz_esquina
            
            sigma_z = q * Iz
            
            st.success(f"""
            **RESULTADOS (según Excel):**  
            • Relación m (B/z): `{m:.4f}`  
            • Relación n (L/z): `{n:.4f}`  
            • Factor de influencia (Iz): `{Iz:.6f}`  
            • Esfuerzo vertical (σz): `{sigma_z:.2f} kPa`  
            """)
    
    with col2:
        if 'calc_rect' in st.session_state:
            # Gráfico de variación con profundidad
            z_values = np.linspace(0.1, 3*max(B,L), 50)
            iz_values = []
            
            for zi in z_values:
                mi = B/zi
                if posicion == "Esquina":
                    iz = (1/math.pi)*(math.log(mi + math.sqrt(1 + mi**2)) + mi*math.log((1 + math.sqrt(1 + mi**2))/mi))
                else:
                    iz_esq = (1/math.pi)*(math.log(mi + math.sqrt(1 + mi**2)) + mi*math.log((1 + math.sqrt(1 + mi**2))/mi))
                    iz = 2 * iz_esq
                iz_values.append(iz)
            
            fig, ax = plt.subplots(figsize=(8,5))
            ax.plot(z_values, iz_values, 'b-')
            ax.axvline(x=z, color='r', linestyle='--', label=f'Profundidad calculada ({z}m)')
            ax.set_title(f"Variación del Factor de Influencia (Iz)")
            ax.set_xlabel("Profundidad z (m)")
            ax.set_ylabel("Factor Iz")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)
            
            # Diagrama esquemático
            fig2 = plt.figure(figsize=(8,6))
            ax2 = fig2.add_subplot(111)
            rect = plt.Rectangle((0,0), B, L, color='red', alpha=0.3, label='Área cargada')
            ax2.add_patch(rect)
            
            if posicion == "Esquina":
                ax2.plot(0, 0, 'bo', markersize=10, label='Punto (esquina)')
            else:
                ax2.plot(B/2, L/2, 'bo', markersize=10, label='Punto (centro)')
            
            ax2.set_xlim(-1, B+1)
            ax2.set_ylim(-1, L+1)
            ax2.set_aspect('equal')
            ax2.set_title("Esquema de la Cimentación")
            ax2.set_xlabel("Ancho B (m)")
            ax2.set_ylabel("Largo L (m)")
            ax2.legend()
            ax2.grid(True)
            st.pyplot(fig2)
# ========== PESTAÑA CARGA LINEAL ==========
with tab4:
    st.header("Carga Lineal")
    col1, col2 = st.columns(2)
    
    with col1:
        q = st.number_input("Carga q (kN/m)", value=50.0, min_value=0.1, key="lin_q")
        z = st.number_input("Profundidad z (m)", value=2.0, min_value=0.1, key="lin_z")
        x = st.number_input("Distancia x (m)", value=1.0, key="lin_x")
        
        if st.button("Calcular", key="calc_lin"):
            sigma_z = (2 * q * z**3) / (math.pi * (x**2 + z**2)**2)
            
            st.success(f"""
            **RESULTADOS:**  
            • Esfuerzo vertical (σz): `{sigma_z:.2f} kPa`  
            • Relación x/z: `{x/z:.2f}`
            """)
    
    with col2:
        if 'calc_lin' in st.session_state:
            distancias = np.linspace(-3*z, 3*z, 100)
            esfuerzos = [(2 * q * z**3) / (math.pi * (x**2 + z**2)**2) for x in distancias]
            
            fig, ax = plt.subplots()
            ax.plot(distancias, esfuerzos, 'b-', linewidth=2)
            ax.axvline(x=x, color='r', linestyle='--')
            ax.set_title("Distribución de Esfuerzos (Carga Lineal)")
            ax.set_xlabel("Distancia x (m)")
            ax.set_ylabel("Esfuerzo σz (kPa)")
            ax.grid(True)
            st.pyplot(fig)

# ========== PESTAÑA CARGA TRAPEZOIDAL ==========
with tab5:
    st.header("Carga Rectangular (Banqueta)")
    col1, col2 = st.columns(2)
    
    with col1:
        a = st.number_input("Longitud a (m)", value=20.0, min_value=0.1, key="banq_a")
        b = st.number_input("Ancho b (m)", value=5.0, min_value=0.1, key="banq_b")
        z = st.number_input("Profundidad z (m)", value=10.0, min_value=0.1, key="banq_z")
        q = st.number_input("Carga q (kPa)", value=200.0, min_value=0.1, key="banq_q")
        
        if st.button("Calcular", key="calc_banq"):
            # Cálculos según las fórmulas del Excel
            m = a / z
            n = b / z
            Iq = (1/np.pi) * (((m + n)/m) * np.arctan(m/(1 + n**2 + m*n)) + np.arctan(n))
            sigma_z = Iq * q
            
            st.success(f"""
            **RESULTADOS:**  
            • Parámetro m: `{m:.4f}`  
            • Parámetro n: `{n:.4f}`  
            • Factor de influencia Iq: `{Iq:.6f}`  
            • Incremento de tensión vertical (Δσ): `{sigma_z:.2f} kPa`
            """)
    
    with col2:
        if 'calc_banq' in st.session_state:
            # Gráfico de la banqueta
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection='3d')
            
            # Crear coordenadas para la banqueta
            x = np.linspace(0, a, 10)
            y = np.linspace(0, b, 10)
            X, Y = np.meshgrid(x, y)
            Z = np.zeros_like(X)
            
            # Dibujar la banqueta
            ax.plot_surface(X, Y, Z, color='r', alpha=0.5)
            ax.set_title("Geometría de la Banqueta")
            ax.set_xlabel("Longitud a (m)")
            ax.set_ylabel("Ancho b (m)")
            ax.set_zlabel("Profundidad (m)")
            ax.view_init(elev=30, azim=45)
            
            st.pyplot(fig)
            
            # Gráfico de variación de Iq con z
            z_vals = np.linspace(0.1, 2*z, 50)
            m_vals = a / z_vals
            n_vals = b / z_vals
            Iq_vals = (1/np.pi) * (((m_vals + n_vals)/m_vals) * np.arctan(m_vals/(1 + n_vals**2 + m_vals*n_vals)) + np.arctan(n_vals))
            
            fig2, ax2 = plt.subplots(figsize=(8, 4))
            ax2.plot(z_vals, Iq_vals, 'b-')
            ax2.axvline(x=z, color='r', linestyle='--', label=f'z calculado ({z}m)')
            ax2.set_title("Variación del Factor de Influencia con Profundidad")
            ax2.set_xlabel("Profundidad z (m)")
            ax2.set_ylabel("Factor de Influencia Iq")
            ax2.legend()
            ax2.grid(True)
            
            st.pyplot(fig2)
 
# ========== PESTAÑA ESFUERZOS ========== 
with tab6:
    st.header("Cálculo de Esfuerzos por Estratos")
    
    # Configuración inicial
    col1, col2 = st.columns(2)
    with col1:
        nf = st.number_input("Nivel Freático (m)", value=2.0, min_value=0.0, key="nf")
        unidad_gamma = st.selectbox("Unidad peso específico", ["kN/m³", "kg/m³"], key="unidad_gamma")
    
    # Determinar gamma agua según unidad seleccionada
    gamma_agua = 9.81 if unidad_gamma == "kN/m³" else 1000
    unidad_resultado = "kN/m²" if unidad_gamma == "kN/m³" else "kg/m²"
    
    # Lista para almacenar estratos
    if 'estratos' not in st.session_state:
        st.session_state.estratos = [{'espesor': 1.0, 'gamma': 18.0}]
    
    # Interfaz para agregar estratos
    with st.expander("🏗️ Configuración de Estratos"):
        for i, estrato in enumerate(st.session_state.estratos):
            cols = st.columns(2)
            with cols[0]:
                st.session_state.estratos[i]['espesor'] = st.number_input(
                    f"Espesor Estrato {i+1} (m)", 
                    value=estrato['espesor'], 
                    min_value=0.1,
                    key=f"espesor_{i}"
                )
            with cols[1]:
                st.session_state.estratos[i]['gamma'] = st.number_input(
                    f"γ Estrato {i+1} ({unidad_gamma})", 
                    value=estrato['gamma'], 
                    min_value=0.1,
                    key=f"gamma_{i}"
                )
        
        if st.button("➕ Agregar Estrato"):
            st.session_state.estratos.append({'espesor': 1.0, 'gamma': 18.0})
            st.rerun()
    
    # Cálculo y resultados
    if st.button("📊 Calcular Esfuerzos", key="calc_esf"):
        profundidad_acum = 0
        esfuerzo_total = 0
        resultados = []
        
        for estrato in st.session_state.estratos:
            espesor = estrato['espesor']
            gamma = estrato['gamma']
            
            # Cálculo esfuerzo total (en las unidades seleccionadas)
            esfuerzo_total += gamma * espesor
            
            # Cálculo presión de poros (usa gamma_agua según unidad seleccionada)
            if profundidad_acum + espesor <= nf:
                u = 0  # Arriba del nivel freático
            else:
                if profundidad_acum < nf:
                    # Estrato que cruza el nivel freático
                    u = gamma_agua * (profundidad_acum + espesor - nf)
                else:
                    # Completamente debajo del nivel freático
                    u = gamma_agua * espesor + (resultados[-1]['presion_poros'] if resultados else 0)
            
            esfuerzo_efectivo = esfuerzo_total - u
            
            resultados.append({
                'profundidad': f"{profundidad_acum:.2f}-{profundidad_acum + espesor:.2f} m",
                'esfuerzo_total': esfuerzo_total,
                'presion_poros': u,
                'esfuerzo_efectivo': esfuerzo_efectivo
            })
            
            profundidad_acum += espesor
        
        # Mostrar resultados en tabla
        st.markdown(f"**Resultados en {unidad_resultado}:**")
        st.dataframe(
            resultados,
            column_config={
                'profundidad': "Profundidad",
                'esfuerzo_total': st.column_config.NumberColumn(f"Esf. Total ({unidad_resultado})", format="%.2f"),
                'presion_poros': st.column_config.NumberColumn(f"Pres. Poros ({unidad_resultado})", format="%.2f"),
                'esfuerzo_efectivo': st.column_config.NumberColumn(f"Esf. Efectivo ({unidad_resultado})", format="%.2f")
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
        sigma_total = 0
        
        for estrato in st.session_state.estratos:
            gamma = estrato['gamma']
            espesor = estrato['espesor']
            
            # Actualizar esfuerzo total
            sigma_total += gamma * espesor
            current_depth += espesor
            
            # Calcular presión de poros
            if current_depth <= nf:
                presion_poros = 0
            else:
                if current_depth - espesor < nf:
                    # Estrato cruza el NF
                    presion_poros = gamma_agua * (current_depth - nf)
                else:
                    # Todo el estrato bajo el NF
                    presion_poros = gamma_agua * espesor + u[-1]
            
            esfuerzo_efectivo = sigma_total - presion_poros
            
            profundidades.append(current_depth)
            sigmas.append(sigma_total)
            u.append(presion_poros)
            sigma_eff.append(esfuerzo_efectivo)
        
        ax.plot(sigmas, profundidades, 'r-', label=f'Esfuerzo Total ({unidad_resultado})')
        ax.plot(u, profundidades, 'b-', label=f'Presión de Poros ({unidad_resultado})')
        ax.plot(sigma_eff, profundidades, 'g--', label=f'Esfuerzo Efectivo ({unidad_resultado})')
        
        ax.set_ylim(max(profundidades), 0)  # Invertir eje Y
        ax.set_xlabel(f"Esfuerzo ({unidad_resultado})")
        ax.set_ylabel("Profundidad (m)")
        ax.legend()
        ax.grid(True)
        ax.set_title(f"Distribución de Esfuerzos Verticales ({unidad_resultado})")
        st.pyplot(fig)

# ========== PESTAÑA CONSOLIDACIÓN ==========
with tab7:
    st.header("Cálculo de Consolidación Primaria")
    
    col1, col2 = st.columns(2)
    with col1:
        h = st.number_input("Altura de drenaje (m)", value=5.0, min_value=0.0001, key="cons_h")
        cv = st.number_input("Coeficiente consolidación Cv (m²/s)", value=1e-7, format="%e", key="cons_cv")
        u = st.slider("Grado consolidación U (%)", 1, 99, 50, key="cons_u")
    
    if st.button("⏱️ Calcular Tiempo", key="calc_cons"):
        if u < 60:
            tv = (math.pi/4) * (u/100)**2
        else:
            tv = 1.781 - 0.933 * math.log(1 - u/100)
        
        t = (tv * h**2 )/ cv
        
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
        • Factor tiempo (Tv): `{tv:.4f}`  
        • Tiempo requerido: `{tiempo}`  
        • Equivalente: `{t:.2f} segundos`
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
st.sidebar.caption("Proyecto desarrollado por Armando Caldeón (2023172381) para ITCR")
