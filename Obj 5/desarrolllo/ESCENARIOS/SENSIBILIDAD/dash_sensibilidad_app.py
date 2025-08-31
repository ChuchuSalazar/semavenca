
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# Simulación Montecarlo para efectos de rumores sobre CS y DH


def simulate_rumor_effects(n_simulations=1000, rumor_type='negativo'):
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
    return pd.DataFrame({
        'CS': simulated_CS,
        'DH': simulated_DH,
        'PCA': 0.3777 * 0.5 + 0.2226 * simulated_DH + (-0.5947) * 0.2 + 0.2866 * simulated_CS
    })


# Crear la app Dash
app = dash.Dash(__name__)
app.layout = html.Div([
    html.H2("Simulación de Rumores y PCA"),
    html.Div([
        html.Button("Simular Rumor Negativo", id='btn-negativo', n_clicks=0),
        html.Button("Simular Rumor Positivo", id='btn-positivo', n_clicks=0)
    ], style={'margin': '10px'}),
    dcc.Graph(id='pca-graph'),
    html.Div(id='summary-output')
])


@app.callback(
    Output('pca-graph', 'figure'),
    Output('summary-output', 'children'),
    Input('btn-negativo', 'n_clicks'),
    Input('btn-positivo', 'n_clicks')
)
def update_graph(n_neg, n_pos):
    ctx = dash.callback_context
    if not ctx.triggered:
        rumor_type = 'none'
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        rumor_type = 'negativo' if button_id == 'btn-negativo' else 'positivo'

    df = simulate_rumor_effects(rumor_type=rumor_type)
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=df['PCA'], nbinsx=50, name='Distribución PCA'))
    fig.update_layout(
        title=f"Distribución PCA bajo rumor {rumor_type}", xaxis_title="PCA", yaxis_title="Frecuencia")

    summary = f"Promedio PCA: {df['PCA'].mean():.4f} | Desviación estándar: {df['PCA'].std():.4f}"
    return fig, summary


if __name__ == '__main__':
    app.run(debug=True)
