
"""
Aplicación Dash Interactiva para Tesis Doctoral
Autor: Doctorando MSc. Jesús F. Salazar Rojas
"""

import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# Ruta del archivo Excel
excel_path = r"C:\01 academico\001 Doctorado Economia UCAB\d tesis problema ahorro\01 TESIS DEFINITIVA\MODELO\resultados obj5_1\corrida scores sin intermedia\SCORE HM.xlsx"

# Cargar datos reales
df = pd.read_excel(excel_path, engine="openpyxl")

# Crear aplicación Dash
app = dash.Dash(__name__)
app.title = "Sensibilidad PCA - Doctorando MSc. Jesús F. Salazar Rojas"

app.layout = html.Div([
    html.H3("UNIVERSIDAD CATÓLICA ANDRÉS BELLO", style={'textAlign': 'center'}),
    html.H4("FACULTAD DE CIENCIAS ECONÓMICAS Y SOCIALES - DOCTORADO EN CIENCIAS ECONÓMICAS", style={'textAlign': 'center'}),
    html.H5("LA PROPENSIÓN CONDUCTUAL AL AHORRO: UN ESTUDIO DESDE LOS SESGOS COGNITIVOS", style={'textAlign': 'center'}),
    html.H6("Autor: Doctorando MSc. Jesús F. Salazar Rojas", style={'textAlign': 'center', 'marginBottom': '40px'}),

    html.Label("Seleccionar Grupo:"),
    dcc.Dropdown(id='grupo', options=[
        {'label': 'Hombres (Hah)', 'value': 'Hah'},
        {'label': 'Mujeres (Mah)', 'value': 'Mah'}
    ], value='Hah'),

    html.Label("Tipo de gráfico:"),
    dcc.Dropdown(id='tipo_grafico', options=[
        {'label': 'Histograma', 'value': 'hist'},
        {'label': 'Líneas', 'value': 'line'},
        {'label': 'Circular', 'value': 'pie'},
        {'label': '3D Dispersión', 'value': 'scatter3d'}
    ], value='line'),

    dcc.Graph(id='grafico-pca'),

    html.Label("Mostrar Supuestos del Modelo:"),
    dcc.Checklist(id='mostrar-supuestos', options=[{'label': 'Ver Supuestos', 'value': 'mostrar'}], value=[]),
    html.Div(id='supuestos-output')
])

@app.callback(
    Output('grafico-pca', 'figure'),
    Output('supuestos-output', 'children'),
    Input('grupo', 'value'),
    Input('tipo_grafico', 'value'),
    Input('mostrar-supuestos', 'value')
)
def actualizar_grafico(grupo, tipo_grafico, mostrar):
    df_filtrado = df[df['GRUPO'] == grupo]
    fig = go.Figure()

    if tipo_grafico == 'hist':
        fig.add_trace(go.Histogram(x=df_filtrado['PCA'], name='PCA', marker_color='blue'))
    elif tipo_grafico == 'line':
        fig.add_trace(go.Scatter(x=df_filtrado['Case'], y=df_filtrado['PCA'], mode='lines+markers', name='PCA', line=dict(color='blue')))
    elif tipo_grafico == 'pie':
        fig.add_trace(go.Pie(labels=df_filtrado['GRUPO'], values=df_filtrado['PCA'], name='Distribución PCA'))
    elif tipo_grafico == 'scatter3d':
        fig.add_trace(go.Scatter3d(x=df_filtrado['PSE'], y=df_filtrado['DH'], z=df_filtrado['PCA'],
                                    mode='markers', marker=dict(size=5, color=df_filtrado['PCA'], colorscale='Viridis'),
                                    name='PCA 3D'))

    fig.update_layout(title=f"Visualización PCA - Grupo {grupo}", height=600)

    supuestos = """
    1. Linealidad entre variables latentes.
    2. Independencia de errores.
    3. Normalidad multivariada.
    4. No multicolinealidad severa.
    5. Validez y confiabilidad de los constructos.
    """
    texto = html.Pre(supuestos) if 'mostrar' in mostrar else ""
    return fig, texto

if __name__ == '__main__':
    app.run(debug=True)
