# --- Librerías ---
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

# --- Rutas de archivos ---
ruta_items = r"C:\01 academico\001 Doctorado Economia UCAB\d tesis problema ahorro\01 TESIS DEFINITIVA\MODELO\resultados obj5_1\corrida scores sin intermedia\Standardized Indicator Scores ITEMS.xlsx"
ruta_latentes = r"C:\01 academico\001 Doctorado Economia UCAB\d tesis problema ahorro\01 TESIS DEFINITIVA\MODELO\resultados obj5_1\corrida scores sin intermedia\SCORE HM.xlsx"

# --- Lectura de data ---
df_scores = pd.read_excel(ruta_items)
df_latentes = pd.read_excel(ruta_latentes)

# --- Mapear variables sociodemográficas ---
edad_map = {1: '<26', 2: '26-30', 3: '31-34', 4: '36-40',
            5: '41-45', 6: '46-50', 7: '51-55', 8: '56-60', 9: '>60'}
estudios_map = {1: 'Primaria', 2: 'Bachillerato', 3: 'T.S.U.',
                4: 'Universitario', 5: 'Postgrado', 6: 'Doctorado'}
ingreso_map = {1: '3-100$', 2: '101-450$', 3: '451-1800$',
               4: '1801-2500$', 5: '2501-10000$', 6: '>10000$'}

df_scores['Edad'] = df_scores['PCA2'].map(edad_map)
df_scores['Estudios'] = df_scores['PCA4'].map(estudios_map)
df_scores['Ingreso'] = df_scores['PCA5'].map(ingreso_map)

# --- Selección de columnas clave ---
columnas_sesgos = ['AV1', 'AV2', 'AV3', 'AV5', 'DH2', 'DH3',
                   'DH4', 'DH5', 'SQ1', 'SQ2', 'SQ3', 'CS2', 'CS3', 'CS5']
df_sesgos = df_scores[columnas_sesgos + ['GRUPO']]

# --- Streamlit ---
st.title("Simulación PCA con Escenarios Basados en Rumores")

st.sidebar.header("Filtros")
grupo_sel = st.sidebar.multiselect(
    "Seleccione Grupo", options=df_scores['GRUPO'].unique(), default=df_scores['GRUPO'].unique())

df_filtrado = df_scores[df_scores['GRUPO'].isin(grupo_sel)]

# --- Visualización interactiva ---
fig_av = px.box(df_filtrado, x='GRUPO', y=[
                'AV1', 'AV2', 'AV3', 'AV5'], title='Distribución AV por Grupo')
st.plotly_chart(fig_av)

fig_dh = px.box(df_filtrado, x='GRUPO', y=[
                'DH2', 'DH3', 'DH4', 'DH5'], title='Distribución DH por Grupo')
st.plotly_chart(fig_dh)

fig_sq = px.box(df_filtrado, x='GRUPO', y=[
                'SQ1', 'SQ2', 'SQ3'], title='Distribución SQ por Grupo')
st.plotly_chart(fig_sq)

fig_cs = px.box(df_filtrado, x='GRUPO', y=[
                'CS2', 'CS3', 'CS5'], title='Distribución CS por Grupo')
st.plotly_chart(fig_cs)

# --- Monte Carlo: simulación de escenarios ---
st.header("Simulación Monte Carlo PCA")

num_sim = st.slider("Número de iteraciones Monte Carlo",
                    min_value=1000, max_value=10000, value=5000, step=500)

# Definir parámetros base para cada sesgo según promedio
promedios_sesgos = df_sesgos[columnas_sesgos].mean()

sim_results = []

for i in range(num_sim):
    AV_sim = np.random.normal(
        promedios_sesgos[['AV1', 'AV2', 'AV3', 'AV5']].mean(), 0.1)
    DH_sim = np.random.normal(
        promedios_sesgos[['DH2', 'DH3', 'DH4', 'DH5']].mean(), 0.1)
    SQ_sim = np.random.normal(
        promedios_sesgos[['SQ1', 'SQ2', 'SQ3']].mean(), 0.1)
    CS_sim = np.random.normal(
        promedios_sesgos[['CS2', 'CS3', 'CS5']].mean(), 0.1)

    PCA_sim = 0.840*AV_sim - 0.595*SQ_sim + 0.287*CS_sim + 0.223*DH_sim
    sim_results.append(PCA_sim)

sim_results = pd.Series(sim_results)
st.write("Resumen de PCA simulado (Monte Carlo):")
st.write(sim_results.describe())

fig_mc = px.histogram(sim_results, nbins=50,
                      title="Distribución PCA - Monte Carlo")
st.plotly_chart(fig_mc)

# --- Exportar Excel limpio para análisis posterior ---
df_filtrado.to_excel("PCA_clean.xlsx", index=False)
st.success("Archivo PCA_clean.xlsx generado para análisis posterior")
