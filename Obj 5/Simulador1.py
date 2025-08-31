# simulador_pca.py
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Simulador PCA/SQ", layout="wide")

st.title("Dashboard Interactivo: PCA y SQ")
st.markdown(
    "Simula el impacto de las variables sobre PCA y SQ para Hombres y Mujeres."
)

# Tarjeta de presentación
st.markdown("""
---
**MSc Jesús F. Salazar Rojas**  
**Doctorando**  
**Tesis:** *La Propensión Conductual al Ahorro: Un estudio desde los sesgos cognitivos para la toma de decisiones en el ahorro de los hogares*
---
""")

# Selección de grupo
grupo = st.selectbox("Selecciona el grupo", ["Hombres", "Mujeres"])

# Sliders para las variables
st.sidebar.header("Ajusta las variables independientes")

PSE = st.sidebar.slider("PSE", -10.0, 10.0, 0.0, step=0.1)
DH = st.sidebar.slider("DH", -10.0, 10.0, 0.0, step=0.1)
SQ_var = st.sidebar.slider("SQ (como predictor)", -10.0, 10.0, 0.0, step=0.1)
CS = st.sidebar.slider("CS", -10.0, 10.0, 0.0, step=0.1)
AV = st.sidebar.slider("AV", -10.0, 10.0, 0.0, step=0.1)

# Número de simulaciones para Monte Carlo
n_sim = st.sidebar.number_input(
    "Número de simulaciones Monte Carlo", min_value=1, max_value=5000, value=1000, step=100
)

# Definición de ecuaciones


def calcular_pca_sq(grupo, PSE, DH, SQ_var, CS, AV):
    if grupo == "Hombres":
        PCA = 0.3777*PSE + 0.2226*DH - 0.5947*SQ_var + 0.2866*CS
        SQ = 0.8402*AV
    else:
        PCA = 0.3485*PSE - 0.2013*DH - 0.5101*SQ_var + 0.3676*CS
        SQ = 0.6609*AV
    return PCA, SQ


# Calcular resultados base
PCA_base, SQ_base = calcular_pca_sq(grupo, PSE, DH, SQ_var, CS, AV)

st.subheader("Resultados individuales")
st.write(f"PCA: {PCA_base:.3f}")
st.write(f"SQ: {SQ_base:.3f}")

# Monte Carlo
if st.checkbox("Realizar simulación Monte Carlo"):
    st.subheader("Simulación Monte Carlo")
    # Generar variables aleatorias dentro de ±10% de cada input
    PSE_mc = np.random.normal(PSE, abs(PSE)*0.1, n_sim)
    DH_mc = np.random.normal(DH, abs(DH)*0.1, n_sim)
    SQ_mc = np.random.normal(SQ_var, abs(SQ_var)*0.1, n_sim)
    CS_mc = np.random.normal(CS, abs(CS)*0.1, n_sim)
    AV_mc = np.random.normal(AV, abs(AV)*0.1, n_sim)

    PCA_vals, SQ_vals = [], []
    for i in range(n_sim):
        PCA_i, SQ_i = calcular_pca_sq(
            grupo, PSE_mc[i], DH_mc[i], SQ_mc[i], CS_mc[i], AV_mc[i]
        )
        PCA_vals.append(PCA_i)
        SQ_vals.append(SQ_i)

    df_mc = pd.DataFrame({"PCA": PCA_vals, "SQ": SQ_vals})
    st.write("Resultados de Monte Carlo (primeras 10 filas):")
    st.dataframe(df_mc.head(10))

    st.subheader("Estadísticas de Monte Carlo")
    st.write(df_mc.describe())

    st.subheader("Histogramas")
    st.bar_chart(df_mc)

st.markdown("© Dashboard interactivo PCA/SQ")
