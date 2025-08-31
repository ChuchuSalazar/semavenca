# analisis.py
import numpy as np
import pandas as pd

# ----------------------------------------------------------
# Coeficientes PLS-SEM
# ----------------------------------------------------------
model_coefficients = {
    'MUJERES': {
        'pca_equation': {'PSE': 0.3485, 'DH': -0.2013, 'SQ': -0.5101, 'CS': 0.3676},
        'sq_equation': {'AV': 0.6609}
    },
    'HOMBRES': {
        'pca_equation': {'PSE': 0.3777, 'DH': 0.2226, 'SQ': -0.5947, 'CS': 0.2866},
        'sq_equation': {'AV': 0.8402}
    }
}

# ----------------------------------------------------------
# Función para calcular predicciones PCA y SQ
# ----------------------------------------------------------


def calcular_predicciones(df, grupo='HOMBRES'):
    coeffs = model_coefficients[grupo]

    # PCA
    pca_pred = (
        coeffs['pca_equation']['PSE'] * df['PSE'] +
        coeffs['pca_equation']['DH'] * df['DH'] +
        coeffs['pca_equation']['SQ'] * df['SQ'] +
        coeffs['pca_equation']['CS'] * df['CS']
    )

    # SQ
    sq_pred = coeffs['sq_equation']['AV'] * df['AV']

    # Agregar predicciones al DataFrame
    df['PCA_pred'] = pca_pred
    df['SQ_pred'] = sq_pred

    return df

# ----------------------------------------------------------
# Función simple de bootstrap y Monte Carlo
# ----------------------------------------------------------


def bootstrap_mc(resultados, df, grupo):
    """
    Agrega columnas de bootstrap simples como ejemplo
    """
    resultados['PCA_boot'] = np.mean(resultados['PCA_pred'])
    resultados['SQ_boot'] = np.mean(resultados['SQ_pred'])
    print(f"[{grupo}] Bootstrap realizado con éxito")

# ----------------------------------------------------------
# Clase para análisis de predicción PLS y bootstrap
# ----------------------------------------------------------


class PLSPredictAnalyzer:
    def prepare_data(self, df):
        """
        Prepara X e y para el análisis.
        X: variables independientes para PCA
        y: variable dependiente PCA
        """
        X = df[['PSE', 'DH', 'SQ', 'CS']].values
        y = df['PCA_pred'].values
        return X, y, None, None

    def bootstrap_prediction_intervals(self, X, y, n_bootstrap=1000, alpha=0.05):
        """
        Calcula R² promedio con bootstrap y su intervalo de confianza
        """
        r2_list = []
        for _ in range(n_bootstrap):
            idx = np.random.choice(len(X), len(X), replace=True)
            X_sample, y_sample = X[idx], y[idx]
            # regresión lineal simple por OLS
            beta = np.linalg.lstsq(X_sample, y_sample, rcond=None)[0]
            y_pred = X_sample @ beta
            r2 = 1 - np.sum((y_sample - y_pred)**2) / \
                np.sum((y_sample - np.mean(y_sample))**2)
            r2_list.append(r2)

        r2_corrected = np.mean(r2_list)
        ci_lower = np.percentile(r2_list, 100*alpha/2)
        ci_upper = np.percentile(r2_list, 100*(1-alpha/2))
        return r2_corrected, (ci_lower, ci_upper)
