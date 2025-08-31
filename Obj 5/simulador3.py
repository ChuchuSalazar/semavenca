from dash.dependencies import Input, Output
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import dash
from dash import dcc, html, dash_table
import plotly.express as px
import plotly.graph_objects as go
import os

# -----------------------------
# RUTAS
# -----------------------------
ruta_data_original = r"C:\01 academico\001 Doctorado Economia UCAB\d tesis problema ahorro\01 TESIS DEFINITIVA\MODELO\resultados obj5\descripiva por grupo de ahorradores.xlsx"
ruta_data_agregada = r"C:\01 academico\001 Doctorado Economia UCAB\d tesis problema ahorro\01 TESIS DEFINITIVA\MODELO\resultados obj5\DATA_CONSOLIDADA promedio H M .xlsx"
ruta_resultados = r"C:\01 academico\001 Doctorado Economia UCAB\d tesis problema ahorro\01 TESIS DEFINITIVA\MODELO\resultados obj5"

# -----------------------------
# LEER DATOS POR HOJAS
# -----------------------------
df_hombres = pd.read_excel(ruta_data_original, sheet_name='hombres')
df_mujeres = pd.read_excel(ruta_data_original, sheet_name='mujeres')

df_hombres_ag = pd.read_excel(ruta_data_agregada, sheet_name='hombres')
df_mujeres_ag = pd.read_excel(ruta_data_agregada, sheet_name='mujeres')

# -----------------------------
# COEFICIENTES PLS-SEM
# -----------------------------
coefs_hombres = {'PSE': 0.3777, 'SQ': -0.5947,
                 'DH': 0.2226, 'CS': 0.2866, 'AV': 0.8402}
coefs_mujeres = {'PSE': 0.3485, 'SQ': -0.5101,
                 'DH': -0.2013, 'CS': 0.3676, 'AV': 0.6609}

# -----------------------------
# FUNCIONES
# -----------------------------


def calcular_PCA_SQ(df, coefs):
    df = df.copy()
    df['SQ_pred'] = coefs['AV'] * df['AV']
    df['PCA_pred'] = (coefs['PSE']*df['PSE'] +
                      coefs['SQ']*df['SQ_pred'] +
                      coefs['DH']*df['DH'] +
                      coefs['CS']*df['CS'])
    return df


def plspredict(df, y_col, X_cols, k=5):
    """Validación cruzada tipo PLSpredict"""
    df = df.copy()
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    y_true_all, y_pred_all = [], []

    for train_idx, test_idx in kf.split(df):
        X_train, X_test = df.iloc[train_idx][X_cols], df.iloc[test_idx][X_cols]
        y_train, y_test = df.iloc[train_idx][y_col], df.iloc[test_idx][y_col]

        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)

    rmse = np.sqrt(mean_squared_error(y_true_all, y_pred_all))
    mae = mean_absolute_error(y_true_all, y_pred_all)
    return y_true_all, y_pred_all, rmse, mae


def monte_carlo(df, coefs, n_sim=1000):
    errors = []
    for _ in range(n_sim):
        df_sim = df.copy()
        for col in ['PSE', 'DH', 'CS', 'AV']:
            df_sim[col] = df_sim[col] * \
                (1 + np.random.normal(0, 0.05, len(df)))
        df_sim = calcular_PCA_SQ(df_sim, coefs)
        err = df_sim['PCA_pred'] - df_sim['PCA_pred'].mean()
        errors.append(err)
    errors = np.array(errors).flatten()
    return errors


# -----------------------------
# CALCULOS
# -----------------------------
df_hombres = calcular_PCA_SQ(df_hombres, coefs_hombres)
df_mujeres = calcular_PCA_SQ(df_mujeres, coefs_mujeres)

# Validación Cruzada PLSpredict
X_cols = ['PSE', 'DH', 'CS', 'AV']
y_col = 'PCA_pred'

y_true_h, y_pred_h, rmse_h, mae_h = plspredict(df_hombres, y_col, X_cols)
y_true_m, y_pred_m, rmse_m, mae_m = plspredict(df_mujeres, y_col, X_cols)

# Monte Carlo
errors_h = monte_carlo(df_hombres, coefs_hombres, n_sim=1000)
errors_m = monte_carlo(df_mujeres, coefs_mujeres, n_sim=1000)

# -----------------------------
# DASHBOARD
# -----------------------------
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Dashboard PCA y SQ - Tesis Doctoral",
            style={'textAlign': 'center'}),

    html.H2("Resumen Estadístico Avanzado"),
    dash_table.DataTable(
        columns=[{"name": i, "id": i}
                 for i in ['Grupo', 'RMSE', 'MAE', 'Media_pred', 'Std_pred']],
        data=[
            {'Grupo': 'Hombres', 'RMSE': rmse_h, 'MAE': mae_h,
             'Media_pred': np.mean(y_pred_h), 'Std_pred': np.std(y_pred_h)},
            {'Grupo': 'Mujeres', 'RMSE': rmse_m, 'MAE': mae_m,
             'Media_pred': np.mean(y_pred_m), 'Std_pred': np.std(y_pred_m)}
        ]
    ),

    html.H2("Histogramas de Errores Monte Carlo"),
    dcc.Graph(figure=px.histogram(errors_h, nbins=50, opacity=0.7,
              title='Hombres', color_discrete_sequence=['blue'])),
    dcc.Graph(figure=px.histogram(errors_m, nbins=50, opacity=0.7,
              title='Mujeres', color_discrete_sequence=['red'])),

    html.H2("Comparativa Predicho vs Observado"),
    dcc.Graph(
        figure=go.Figure(
            data=[
                go.Scatter(x=y_true_h, y=y_pred_h,
                           mode='markers', name='Hombres'),
                go.Scatter(x=y_true_m, y=y_pred_m,
                           mode='markers', name='Mujeres')
            ],
            layout=go.Layout(
                title="Predicho vs Observado",
                xaxis_title="Observado PCA",
                yaxis_title="Predicho PCA",
                showlegend=True
            )
        )
    ),

    html.Button("Exportar Resultados a Excel", id='btn_export', n_clicks=0)
])

# Callback para exportar


@app.callback(
    Output('btn_export', 'children'),
    Input('btn_export', 'n_clicks')
)
def export_excel(n_clicks):
    if n_clicks > 0:
        df_hombres.to_excel(os.path.join(
            ruta_resultados, 'PCA_Hombres_resultados.xlsx'), index=False)
        df_mujeres.to_excel(os.path.join(
            ruta_resultados, 'PCA_Mujeres_resultados.xlsx'), index=False)
        return "Exportado Correctamente"
    return "Exportar Resultados a Excel"


# -----------------------------
# EJECUCION
# -----------------------------
if __name__ == '__main__':
    app.run_server(debug=True)
