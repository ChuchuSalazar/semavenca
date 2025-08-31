
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

# Crear la app Dash
app = dash.Dash(__name__)
app.title = "Sensibilidad PCA - Tesis Doctoral"

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

    dcc.Graph(id='grafico-pca'),

    html.Label("Mostrar Supuestos del Modelo:"),
    dcc.Checklist(id='mostrar-supuestos', options=[{'label': 'Ver Supuestos', 'value': 'mostrar'}], value=[]),
    html.Div(id='supuestos-output')
])

@app.callback(
    Output('grafico-pca', 'figure'),
    Output('supuestos-output', 'children'),
    Input('grupo', 'value'),
    Input('mostrar-supuestos', 'value')
)
def actualizar_grafico(grupo, mostrar):
    df_filtrado = df[df['GRUPO'] == grupo]
    fig = go.Figure()
    fig.add_trace(go.Box(y=df_filtrado['PCA'], name='PCA', boxpoints='all', jitter=0.5, whiskerwidth=0.2, marker_size=4))
    fig.update_layout(title=f"Distribución PCA - Grupo {grupo}", yaxis_title="PCA", height=500)

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
