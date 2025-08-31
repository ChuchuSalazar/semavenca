
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# Ecuaciones de los modelos
ecuacion_hombres = "PCA = 0.3777·PSE + 0.2226·DH - 0.5947·SQ + 0.2866·CS"
ecuacion_mujeres = "PCA = 0.3485·PSE - 0.2013·DH - 0.5101·SQ + 0.3676·CS"

# Simulación Montecarlo para efectos de rumores sobre CS y DH
def simulate_rumor_effects(n_simulations=1000, rumor_type='negativo', grupo='Hah'):
    np.random.seed(42)
    base_CS = 0.5
    base_DH = 0.3
    if rumor_type == 'negativo':
        cs_shift = np.random.normal(-0.2, 0.05, n_simulations)
        dh_shift = np.random.normal(0.15, 0.05, n_simulations)
    elif rumor_type == 'positivo':
        cs_shift = np.random.normal(0.2, 0.05, n_simulations)
        dh_shift = np.random.normal(-0.15, 0.05, n_simulations)
    else:
        cs_shift = np.zeros(n_simulations)
        dh_shift = np.zeros(n_simulations)
    simulated_CS = base_CS + cs_shift
    simulated_DH = base_DH + dh_shift
    if grupo == 'Hah':
        PCA = 0.3777 * 0.5 + 0.2226 * simulated_DH + (-0.5947) * 0.2 + 0.2866 * simulated_CS
    else:
        PCA = 0.3485 * 0.5 + (-0.2013) * simulated_DH + (-0.5101) * 0.2 + 0.3676 * simulated_CS
    return pd.DataFrame({'CS': simulated_CS, 'DH': simulated_DH, 'PCA': PCA})

# Crear la app Dash
app = dash.Dash(__name__)
app.title = "Sensibilidad PCA - Tesis Doctoral"

app.layout = html.Div([
    html.H3("UNIVERSIDAD CATÓLICA ANDRÉS BELLO", style={'textAlign': 'center'}),
    html.H4("FACULTAD DE CIENCIAS ECONÓMICAS Y SOCIALES - DOCTORADO EN CIENCIAS ECONÓMICAS", style={'textAlign': 'center'}),
    html.H5("LA PROPENSIÓN CONDUCTUAL AL AHORRO: UN ESTUDIO DESDE LOS SESGOS COGNITIVOS", style={'textAlign': 'center', 'marginBottom': '30px'}),

    html.Div([
        html.Label("Seleccionar Grupo:"),
        dcc.Dropdown(id='grupo', options=[
            {'label': 'Hombres (Hah)', 'value': 'Hah'},
            {'label': 'Mujeres (Mah)', 'value': 'Mah'}
        ], value='Hah')
    ], style={'width': '30%', 'display': 'inline-block'}),

    html.Div([
        html.Label("Seleccionar Variable:"),
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

    html.Div([
        html.Button("Simular Rumor Negativo", id='btn-negativo', n_clicks=0),
        html.Button("Simular Rumor Positivo", id='btn-positivo', n_clicks=0)
    ], style={'margin': '20px'}),

    html.Div(id='ecuacion-modelo', style={'marginBottom': '20px', 'fontWeight': 'bold'}),
    dcc.Graph(id='pca-graph'),
    html.Div(id='summary-output')
])

@app.callback(
    Output('pca-graph', 'figure'),
    Output('summary-output', 'children'),
    Output('ecuacion-modelo', 'children'),
    Input('btn-negativo', 'n_clicks'),
    Input('btn-positivo', 'n_clicks'),
    Input('grupo', 'value'),
    Input('variable', 'value'),
    Input('rango', 'value')
)
def update_graph(n_neg, n_pos, grupo, variable, rango):
    ctx = dash.callback_context
    if not ctx.triggered:
        rumor_type = 'none'
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        rumor_type = 'negativo' if button_id == 'btn-negativo' else 'positivo'

    df = simulate_rumor_effects(rumor_type=rumor_type, grupo=grupo)
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=df['PCA'], nbinsx=50, name='Distribución PCA', marker_color='indigo'))
    fig.update_layout(title=f"Distribución PCA bajo rumor {rumor_type} - Grupo {grupo}", xaxis_title="PCA", yaxis_title="Frecuencia")

    summary = f"Promedio PCA: {df['PCA'].mean():.4f} | Desviación estándar: {df['PCA'].std():.4f}"
    ecuacion = f"Ecuación del modelo ({'Hombres' if grupo == 'Hah' else 'Mujeres'}): " + (ecuacion_hombres if grupo == 'Hah' else ecuacion_mujeres)
    return fig, summary, ecuacion

if __name__ == '__main__':
    app.run(debug=True)
