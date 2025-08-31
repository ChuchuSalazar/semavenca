# simulado6_corregido.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.stats import pearsonr
import os
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class PLSPredictAnalyzer:
    def __init__(self, output_path="C:/01 academico/001 Doctorado Economia UCAB/d tesis problema ahorro/01 TESIS DEFINITIVA/MODELO/resultados obj5/"):
        self.output_path = output_path
        self.results = {}
        self.figures = []

    def load_data(self, file_path):
        try:
            data = pd.read_excel(file_path)
            print(f"Datos cargados exitosamente: {data.shape}")
            return data
        except Exception as e:
            print(f"Error al cargar datos: {e}")
            return None

    def load_descriptive_stats(self, file_path):
        try:
            desc_stats = pd.read_excel(file_path)
            return desc_stats
        except Exception as e:
            print(f"Error al cargar estadísticas descriptivas: {e}")
            return None

    def prepare_data(self, data):
        predictors = ['PROM_AV', 'PROM_DH', 'PROM_SQ', 'PROM_CS']
        target = 'PPCA'
        X = data[predictors].copy()
        y = data[target].copy()
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X_clean = X[mask]
        y_clean = y[mask]
        print(f"Datos finales para análisis: {X_clean.shape[0]} observaciones")
        return X_clean, y_clean, predictors, target

    # Mantener todas tus funciones de análisis: cross_validation_analysis, cvpat_test,
    # monte_carlo_sensitivity, sensitivity_analysis, bootstrap_prediction_intervals
    # como estaban originalmente (por brevedad no se repiten aquí)

    # --- Funciones de visualización ---
    def create_visualizations(self, X, y):
        """Crear visualizaciones avanzadas"""
        self.plot_observed_vs_predicted(X, y)
        if 'monte_carlo' in self.results:
            self.plot_monte_carlo_distribution()
        self.plot_residual_analysis(X, y)
        if 'cross_validation' in self.results:
            self.plot_model_comparison()
        if 'sensitivity' in self.results:
            self.plot_sensitivity_analysis()

    def plot_observed_vs_predicted(self, X, y):
        """Gráfico de dispersión observado vs predicho con línea 1:1 y leyenda mayores/menores 0"""
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = LinearRegression()
        model.fit(X_scaled, y)
        y_pred = model.predict(X_scaled)

        plt.figure(figsize=(10, 8))
        plt.scatter(y_pred, y, alpha=0.6, s=50, color='steelblue',
                    edgecolors='darkblue', linewidth=0.5)
        min_val = min(y.min(), y_pred.min())
        max_val = max(y.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--',
                 linewidth=2, label='Predicción Perfecta (1:1)')
        z = np.polyfit(y_pred, y, 1)
        p = np.poly1d(z)
        plt.plot(y_pred, p(y_pred), 'orange', linewidth=2,
                 label=f'Regresión (y = {z[0]:.3f}x + {z[1]:.3f})')

        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))

        # Conteo mayores y menores de cero
        mayores_cero = (y_pred > 0).sum()
        menores_cero = (y_pred <= 0).sum()
        plt.legend([f'Mayores 0: {mayores_cero}', f'Menores 0: {menores_cero}',
                   'Predicción Perfecta (1:1)', f'Regresión (y = {z[0]:.3f}x + {z[1]:.3f})'])

        plt.xlabel('Valores Predichos')
        plt.ylabel('Valores Observados')
        plt.title(f'Observado vs Predicho\nR² = {r2:.4f}, RMSE = {rmse:.4f}')
        plt.grid(True, alpha=0.3)

        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        plt.savefig(os.path.join(self.output_path,
                    'observed_vs_predicted.png'), dpi=300, bbox_inches='tight')
        plt.show()

    # Mantener plot_monte_carlo_distribution, plot_residual_analysis, plot_model_comparison,
    # plot_sensitivity_analysis como en tu código original

    def run_complete_analysis(self, data_files, desc_files=None):
        """Ejecutar análisis completo para ambos grupos"""
        base_path = self.output_path  # Guardar path original
        for group, file_path in data_files.items():
            print(f"\n{'='*80}\nANÁLISIS PARA GRUPO: {group.upper()}\n{'='*80}")

            data = self.load_data(file_path)
            if data is None:
                continue

            if desc_files and group in desc_files:
                desc_stats = self.load_descriptive_stats(desc_files[group])
                if desc_stats is not None:
                    print(f"Estadísticas descriptivas cargadas para {group}")

            X, y, predictors, target = self.prepare_data(data)

            # Crear path específico para el grupo
            self.output_path = os.path.join(base_path, f"{group.upper()}_")
            if not os.path.exists(self.output_path):
                os.makedirs(self.output_path)

            try:
                self.cross_validation_analysis(X, y)
                self.monte_carlo_sensitivity(X, y)
                self.bootstrap_prediction_intervals(X, y)
                self.create_visualizations(X, y)
                self.generate_comprehensive_report()
            except Exception as e:
                print(f"Error en análisis de {group}: {e}")
                continue

        self.output_path = base_path  # Restaurar path original

# Función main y clases complementarias GroupComparison, validate_sample_size y check_data_quality
# se mantienen iguales al original, pero la gestión de rutas de salida está corregida
# FUNCIÓN PRINCIPAL PARA EJECUTAR ANÁLISIS


def main():
    """
    Función principal que ejecuta el análisis completo
    """
    # Rutas de archivos
    data_files = {
        'HOMBRES': "C:/01 academico/001 Doctorado Economia UCAB/d tesis problema ahorro/01 TESIS DEFINITIVA/MODELO/resultados obj5/DATA_CONSOLIDADA HOMBRES promedio H M .xlsx",
        'MUJERES': "C:/01 academico/001 Doctorado Economia UCAB/d tesis problema ahorro/01 TESIS DEFINITIVA/MODELO/resultados obj5/DATA_CONSOLIDADA MUJERES promedio H M .xlsx"
    }

    desc_files = {
        'HOMBRES': "C:/01 academico/001 Doctorado Economia UCAB/d tesis problema ahorro/01 TESIS DEFINITIVA/MODELO/resultados obj5/descripiva HOMBRES ahorradores.xlsx",
        'MUJERES': "C:/01 academico/001 Doctorado Economia UCAB/d tesis problema ahorro/01 TESIS DEFINITIVA/MODELO/resultados obj5/descripiva MUJERES ahorradores.xlsx"
    }

    # Crear analizador
    analyzer = PLSPredictAnalyzer()

    print("INICIANDO ANÁLISIS PLS-SEM PREDICTIVO ROBUSTO")
    print("=" * 80)
    print("Implementando mejores prácticas metodológicas:")
    print("• Validación cruzada 10-fold con 20 repeticiones")
    print("• Análisis Monte Carlo con 5,000 simulaciones")
    print("• Bootstrap optimismo-corregido")
    print("• CVPAT para significancia estadística")
    print("• Análisis comprehensivo de residuos")
    print("• Visualizaciones avanzadas")
    print("=" * 80)

    # Ejecutar análisis completo
    analyzer.run_complete_analysis(data_files, desc_files)

    print("\n" + "="*80)
    print("ANÁLISIS COMPLETADO EXITOSAMENTE")
    print("="*80)
    print("Archivos generados:")
    print("• Gráficos: observed_vs_predicted.png, monte_carlo_distribution.png, etc.")
    print("• Resultados Excel: validacion_cruzada_resultados.xlsx, bootstrap_resultados.xlsx, etc.")
    print("• Reportes comprehensivos en consola")


# ANÁLISIS COMPLEMENTARIO: COMPARACIÓN ENTRE GRUPOS
class GroupComparison:
    """Clase para comparar resultados entre grupos (Hombres vs Mujeres)"""

    def __init__(self, output_path):
        self.output_path = output_path

    def compare_predictive_capacity(self, results_h, results_m):
        """Comparar capacidad predictiva entre grupos"""
        print("\n" + "="*60)
        print("COMPARACIÓN ENTRE GRUPOS")
        print("="*60)

        if 'cross_validation' in results_h and 'cross_validation' in results_m:
            h_r2 = results_h['cross_validation']['PLS_SEM']['R2_mean']
            m_r2 = results_m['cross_validation']['PLS_SEM']['R2_mean']

            h_rmse = results_h['cross_validation']['PLS_SEM']['RMSE_mean']
            m_rmse = results_m['cross_validation']['PLS_SEM']['RMSE_mean']

            print(f"Capacidad Predictiva (R²):")
            print(f"  Hombres: {h_r2:.4f}")
            print(f"  Mujeres: {m_r2:.4f}")
            print(f"  Diferencia: {abs(h_r2 - m_r2):.4f}")

            print(f"\nError de Predicción (RMSE):")
            print(f"  Hombres: {h_rmse:.4f}")
            print(f"  Mujeres: {m_rmse:.4f}")
            print(f"  Diferencia: {abs(h_rmse - m_rmse):.4f}")

            # Interpretación
            if abs(h_r2 - m_r2) < 0.05:
                print("\n✓ Capacidad predictiva similar entre grupos")
            else:
                better_group = "Hombres" if h_r2 > m_r2 else "Mujeres"
                print(f"\n⚠ {better_group} muestran mayor capacidad predictiva")


# VALIDACIONES ADICIONALES
def validate_sample_size(X, y, min_obs_per_param=10):
    """Validar que el tamaño de muestra sea adecuado"""
    n_params = X.shape[1] + 1  # predictores + intercepto
    min_required = n_params * min_obs_per_param

    print(f"\nValidación de Tamaño de Muestra:")
    print(f"Observaciones disponibles: {len(X)}")
    print(f"Parámetros del modelo: {n_params}")
    print(f"Mínimo requerido: {min_required}")

    if len(X) >= min_required:
        print("✓ Tamaño de muestra adecuado")
        return True
    else:
        print("⚠ Tamaño de muestra puede ser insuficiente")
        return False


def check_data_quality(X, y):
    """Verificar calidad de los datos"""
    print(f"\nVerificación de Calidad de Datos:")

    # Valores faltantes
    missing_X = X.isnull().sum().sum()
    missing_y = y.isnull().sum()
    print(f"Valores faltantes en predictores: {missing_X}")
    print(f"Valores faltantes en variable objetivo: {missing_y}")

    # Valores extremos
    for col in X.columns:
        q1 = X[col].quantile(0.25)
        q3 = X[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = ((X[col] < lower_bound) | (X[col] > upper_bound)).sum()
        if outliers > 0:
            print(
                f"Valores atípicos en {col}: {outliers} ({outliers/len(X)*100:.1f}%)")

    # Multicolinealidad
    correlation_matrix = X.corr()
    high_corr = (correlation_matrix.abs() > 0.8) & (correlation_matrix != 1.0)
    if high_corr.any().any():
        print("⚠ Posible multicolinealidad detectada (correlaciones > 0.8)")
    else:
        print("✓ No se detecta multicolinealidad severa")


if __name__ == "__main__":
    main()
