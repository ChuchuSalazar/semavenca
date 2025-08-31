import pandas as pd
import numpy as np
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output

# Parámetros base
II = 2482400      # Inversión inicial
IM = 415872       # Ingreso mensual base
MESES = 48        # Horizonte en meses


def calcular_escenario(incremento_pct, tasa_descuento):
    """
    Calcula flujo acumulado neto (IA - II), VAN y mes de recuperación.
    """
    flujo_acumulado = []
    ia = 0
    van = 0

    for mes in range(1, MESES + 1):
        if mes <= 3:  # sin ingresos los 3 primeros meses
            ingreso = 0
        else:
            ingreso = IM * (1 + incremento_pct)

        ia += ingreso
        flujo_neto = ia - II
        flujo_acumulado.append(flujo_neto)

        # VAN: sumamos flujos descontados
        van += ingreso / ((1 + tasa_descuento) ** mes)

    # restamos la inversión inicial
    van -= II

    # mes de recuperación (cuando flujo acumulado cruza cero)
    mes_recuperacion = next(
        (i + 1 for i, f in enumerate(flujo_acumulado) if f >= 0), None)

    df = pd.DataFrame({
        "Mes": range(1, MESES + 1),
        "Flujo Neto (IA - II)": np.round(flujo_acumulado, 2)
    })

    return df, van, mes_recuperacion


# --- DASH APP ---
app = Dash(__name__)

app.layout = html.Div([
    html.H2("Escenario de Sensibilidad Financiera",
            style={"textAlign": "center"}),

    html.Div([
        html.Label("Variación % de Ingresos Mensuales:"),
        dcc.Slider(
            id="incremento-slider",
            min=-0.5, max=0.5, step=0.05, value=0.0,
            marks={i/10: f"{int(i*100)}%" for i in range(-5, 6)}
        ),
        html.Br(),

        html.Label("Tasa de Descuento (% anual equivalente a la TIR):"),
        dcc.Slider(
            id="tasa-slider",
            min=0.0, max=0.5, step=0.01, value=0.1,
            marks={i/10: f"{int(i*100)}%" for i in range(0, 6)}
        ),
    ], style={"margin": "20px"}),

    html.Div(id="indicadores", style={
        "textAlign": "center",
        "margin": "20px",
        "fontSize": "18px"
    }),

    dcc.Graph(id="grafico"),
])


@app.callback(
    [Output("grafico", "figure"),
     Output("indicadores", "children")],
    [Input("incremento-slider", "value"),
     Input("tasa-slider", "value")]
)
def actualizar(incremento, tasa_descuento):
    df, van, mes_recuperacion = calcular_escenario(incremento, tasa_descuento)

    # --- Gráfico ---
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["Mes"],
        y=df["Flujo Neto (IA - II)"],
        mode="lines+markers",
        name="Flujo Neto (IA - II)",
        line=dict(color="blue", width=3)
    ))

    fig.update_layout(
        title="Recuperación de la Inversión (IA - II)",
        xaxis_title="Mes",
        yaxis_title="Monto Acumulado",
        plot_bgcolor="white"
    )

    # --- Indicadores ---
    indicadores = [
        html.Div(f"VAN (a tasa {round(tasa_descuento*100, 2)}%): {van:,.2f}"),
        html.Div(
            f"Mes de recuperación: {mes_recuperacion if mes_recuperacion else 'No se recupera en el horizonte'}")
    ]

    return fig, indicadores


if __name__ == "__main__":
    app.run(debug=True)
