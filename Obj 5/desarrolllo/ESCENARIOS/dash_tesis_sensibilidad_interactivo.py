
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# Ecuaciones de los modelos
ecuacion_hombres = "PCA = 0.3777·PSE + 0.2226·DH - 0.5947·SQ + 0.2866·CS"
ecuacion_mujeres = "PCA = 0.3485·PSE - 0.2013·DH - 0.5101·SQ + 0.3676·CS"

# Simulación de sensibilidad PCA
def simulate_sensitivity(variable='edad', grupo='Hah', value_range=(20, 65), n_steps=30):
    values = np.linspace(value_range[0], value_range[1], n_steps)
    results = []
    for v in values:
        norm = (v - 35) / 15
        if grupo == 'Hah':
            PSE = 0.2 + 0.1 * norm
            DH = 0.3 + 0.15 * norm
            SQ = -0.1 + 0.05 * norm
            CS = 0.1 + 0.12 * norm
            PCA = 0.3777 * PSE + 0.2226 * DH - 0.5947 * SQ + 0.2866 * CS
        else:
            PSE = -0.1 + 0.08 * norm
            DH = -0.4 + 0.1 * norm
            SQ = 0.2 + 0.04 * norm
            CS = -0.3 + 0.1 * norm
            PCA = 0.3485 * PSE - 0.2013 * DH - 0.5101 * SQ + 0.3676 * CS
        results.append({'Valor': v, 'PCA': PCA, 'PSE': PSE, 'DH': DH, 'SQ': SQ, 'CS': CS})
    return pd.DataFrame(results)

# Crear la app Dash
app = dash.Dash(__name__)
app.title = "Sensibilidad PCA - Tesis Doctoral"

app.layout = html.Div([
    html.H3("UNIVERSIDAD CATÓLICA ANDRÉS BELLO", style={'textAlign': 'center'}),
    html.H4("FACULTAD DE CIENCIAS ECONÓMICAS Y SOCIALES - DOCTORADO EN CIENCIAS ECONÓMICAS", style={'textAlign': 'center'}),
    html.H5("LA PROPENSIÓN CONDUCTUAL AL AHORRO: UN ESTUDIO DESDE LOS SESGOS COGNITIVOS", style={'textAlign': 'center', 'marginBottom': '30px'}),

    html.Div([
        html.Label("Grupo:"),
        dcc.Dropdown(id='grupo', options=[
            {'label': 'Hombres (Hah)', 'value': 'Hah'},
            {'label': 'Mujeres (Mah)', 'value': 'Mah'}
        ], value='Hah')
    ], style={'width': '30%', 'display': 'inline-block'}),

    html.Div([
        html.Label("Variable:"),
        dcc.Dropdown(id='variable', options=[
            {'label': 'Edad', 'value': 'edad'},
            {'label': 'Ingreso', 'value': 'ingreso'},
            {'label': 'Educación', 'value': 'educacion'}
        ], value='edad')
    ], style={'width': '30%', 'display': 'inline-block', 'marginLeft': '20px'}),

    html.Div([
        html.Label("Rango de Valores:"),
        dcc.RangeSlider(id='rango', min=18, max=70, step=1, value=[30, 60],
                        marks={i: str(i) for i in range(18, 71, 6)})
    ], style={'marginTop': '20px'}),

    dcc.Graph(id='sensitivity-graph'),
    html.Div(id='ecuacion-modelo', style={'marginTop': '20px', 'fontWeight': 'bold'})
])

@app.callback(
    Output('sensitivity-graph', 'figure'),
    Output('ecuacion-modelo', 'children'),
    Input('grupo', 'value'),
    Input('variable', 'value'),
    Input('rango', 'value')
)
def update_graph(grupo, variable, rango):
    df = simulate_sensitivity(variable=variable, grupo=grupo, value_range=tuple(rango))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Valor'], y=df['PCA'], mode='lines+markers', name='PCA', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df['Valor'], y=df['PSE'], mode='lines', name='PSE', line=dict(color='green')))
    fig.add_trace(go.Scatter(x=df['Valor'], y=df['DH'], mode='lines', name='DH', line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=df['Valor'], y=df['SQ'], mode='lines', name='SQ', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=df['Valor'], y=df['CS'], mode='lines', name='CS', line=dict(color='purple')))
    fig.update_layout(title=f"Sensibilidad de PCA ante cambios en {variable.title()} - Grupo {grupo}",
                      xaxis_title=variable.title(), yaxis_title="Valor", height=600)
    ecuacion = f"Ecuación del modelo ({'Hombres' if grupo == 'Hah' else 'Mujeres'}): " + (ecuacion_hombres if grupo == 'Hah' else ecuacion_mujeres)
    return fig, ecuacion

if __name__ == '__main__':
    app.run(debug=True)
