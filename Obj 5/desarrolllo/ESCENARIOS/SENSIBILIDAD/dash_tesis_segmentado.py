
"""
Aplicación Dash Interactiva para Tesis Doctoral
Autor: Doctorando MSc. Jesús F. Salazar Rojas
"""

import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import json

# Cargar escalas de segmentación
with open("escalas_segmentacion.json", "r", encoding="utf-8") as f:
    escalas = json.load(f)

# Ecuaciones de los modelos
ecuacion_hombres = "PCA = 0.3777·PSE + 0.2226·DH - 0.5947·SQ + 0.2866·CS"
ecuacion_mujeres = "PCA = 0.3485·PSE - 0.2013·DH - 0.5101·SQ + 0.3676·CS"

# Simulación de sensibilidad PCA
def simulate_sensitivity(grupo='Hah', edad=5, educacion=4, ingreso=3):
    norm_edad = (edad - 5) / 4
    norm_edu = (educacion - 3.5) / 2.5
    norm_ing = (ingreso - 3.5) / 2.5
    if grupo == 'Hah':
        PSE = -0.3557 * norm_edad + 0.2800 * norm_edu + 0.8343 * norm_ing
        DH = 0.7097 * norm_edad + 0.4376 * norm_edu
        SQ = 0.3816 * norm_edad + 0.5930 * norm_edu + 0.3358 * norm_ing
        CS = 0.5733 * norm_edad + 0.4983 * norm_edu + 0.1597 * norm_ing
        PCA = 0.3777 * PSE + 0.2226 * DH - 0.5947 * SQ + 0.2866 * CS
    else:
        PSE = -0.5168 * norm_edad - 0.0001 * norm_edu + 0.8496 * norm_ing
        DH = 0.3290 * norm_edad + 0.0660 * norm_edu + 0.8397 * norm_ing
        SQ = 0.5458 * norm_edad + 0.4646 * norm_edu + 0.2946 * norm_ing
        CS = 0.5452 * norm_edad + 0.5117 * norm_edu + 0.2631 * norm_ing
        PCA = 0.3485 * PSE - 0.2013 * DH - 0.5101 * SQ + 0.3676 * CS
    return PSE, DH, SQ, CS, PCA

# Crear la app Dash
app = dash.Dash(__name__)
app.title = "Sensibilidad PCA - Tesis Doctoral"

app.layout = html.Div([
    html.H3("UNIVERSIDAD CATÓLICA ANDRÉS BELLO", style={'textAlign': 'center'}),
    html.H4("FACULTAD DE CIENCIAS ECONÓMICAS Y SOCIALES - DOCTORADO EN CIENCIAS ECONÓMICAS", style={'textAlign': 'center'}),
    html.H5("LA PROPENSIÓN CONDUCTUAL AL AHORRO: UN ESTUDIO DESDE LOS SESGOS COGNITIVOS", style={'textAlign': 'center', 'marginBottom': '30px'}),
    html.H6("Autor: Doctorando MSc. Jesús F. Salazar Rojas", style={'textAlign': 'center', 'marginBottom': '40px'}),

    html.Div([
        html.Label("Grupo:"),
        dcc.Dropdown(id='grupo', options=[
            {'label': 'Hombres (Hah)', 'value': 'Hah'},
            {'label': 'Mujeres (Mah)', 'value': 'Mah'}
        ], value='Hah')
    ], style={'width': '30%', 'display': 'inline-block'}),

    html.Div([
        html.Label("Rango de Edad (PCA2):"),
        dcc.Dropdown(id='edad', options=[{'label': v, 'value': int(k)} for k, v in escalas['PCA2'].items()], value=5)
    ], style={'width': '30%', 'display': 'inline-block', 'marginLeft': '20px'}),

    html.Div([
        html.Label("Nivel Educativo (PCA4):"),
        dcc.Dropdown(id='educacion', options=[{'label': v, 'value': int(k)} for k, v in escalas['PCA4'].items()], value=4)
    ], style={'width': '30%', 'display': 'inline-block', 'marginLeft': '20px'}),

    html.Div([
        html.Label("Ingreso Mensual (PCA5):"),
        dcc.Dropdown(id='ingreso', options=[{'label': v, 'value': int(k)} for k, v in escalas['PCA5'].items()], value=3)
    ], style={'width': '30%', 'display': 'inline-block', 'marginTop': '20px'}),

    dcc.Graph(id='sensibilidad-graph'),
    html.Div(id='ecuacion-modelo', style={'marginTop': '20px', 'fontWeight': 'bold'})
])

@app.callback(
    Output('sensibilidad-graph', 'figure'),
    Output('ecuacion-modelo', 'children'),
    Input('grupo', 'value'),
    Input('edad', 'value'),
    Input('educacion', 'value'),
    Input('ingreso', 'value')
)
def update_graph(grupo, edad, educacion, ingreso):
    PSE, DH, SQ, CS, PCA = simulate_sensitivity(grupo, edad, educacion, ingreso)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=['PSE', 'DH', 'SQ', 'CS', 'PCA'], y=[PSE, DH, SQ, CS, PCA], marker_color=['green','orange','red','purple','blue']))
    fig.update_layout(title=f"Sensibilidad PCA según Segmentación - Grupo {grupo}", yaxis_title="Valor", height=500)
    ecuacion = f"Ecuación del modelo ({'Hombres' if grupo == 'Hah' else 'Mujeres'}): " + (ecuacion_hombres if grupo == 'Hah' else ecuacion_mujeres)
    return fig, ecuacion

if __name__ == '__main__':
    app.run(debug=True)
