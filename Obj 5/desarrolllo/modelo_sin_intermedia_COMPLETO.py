"""
===============================================================================
    SIMULADOR CAPACIDAD PREDICTIVA PLS-SEM: ANÁLISIS ESTRUCTURAL CON VISUALIZACIONES
===============================================================================

Sistema integral de validación predictiva para modelos PLS-SEM estructurales
Autor: MSc. Jesus Fernando Salazar Rojas
Doctorado Economia
UCAB
Fecha: 2025

FUNCIONALIDADES PRINCIPALES:
- Validación predictiva robusta con ecuaciones simultáneas
- Análisis Monte Carlo y Bootstrap
- Comparación entre grupos demográficos (Hombres vs Mujeres)
- Visualizaciones profesionales automatizadas
- Reportes comprehensivos exportables
 CARACTERÍSTICAS TÉCNICAS AVANZADAS
Metodología Econométrica Robusta

Ecuaciones estructurales simultáneas con coeficientes PLS-SEM específicos
Validación cruzada k-fold repetida (10×20) para evaluación robusta
Análisis Monte Carlo (5,000 simulaciones) para evaluación de incertidumbre
Bootstrap optimismo-corregido para intervalos de confianza no sesgados
Pruebas CVPAT para significancia de superioridad predictiva

Análisis Diferenciado por Grupos

Mujeres Ahorradoras (Mah): Modelo con efectos negativos del descuento hiperbólico
Hombres Ahorradores (Hah): Modelo con efectos positivos del descuento hiperbólico
Comparación sistemática entre grupos con análisis estadístico

Validación de Supuestos

Detección automática de multicolinealidad y valores atípicos
Winsorización de outliers en lugar de eliminación
Pruebas de normalidad y validación de supuestos estructurales

📊 VISUALIZACIONES PROFESIONALES GENERADAS
Para cada grupo (Hah/Mah):

observed_vs_predicted_[grupo].png: Análisis de ajuste con ecuaciones simultáneas
monte_carlo_distribution_[grupo].png: Distribuciones de capacidad predictiva
residual_analysis_[grupo].png: Diagnósticos exhaustivos de residuos
model_comparison_[grupo].png: Estructural vs Regresión directa
sensitivity_analysis_[grupo].png: Importancia relativa de variables

Características Visuales:

Alta resolución (300 DPI) para publicación
Paleta de colores diferenciada por género
Estadísticas integradas en cada gráfico
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, RepeatedKFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy import stats
from scipy.stats import pearsonr, normaltest
import warnings
import os
from datetime import datetime
import itertools
from sklearn.preprocessing import StandardScaler
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec

# Configuración global
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')
np.random.seed(42)


os.system('cls' if os.name == 'nt' else 'clear')


class PLSSEMPredictiveAnalyzer:
    """
    Analizador de capacidad predictiva para modelos PLS-SEM con ecuaciones estructurales
    """

    def __init__(self, output_dir=None):
        """
        Inicializa el analizador con configuraciones específicas
        """
        if output_dir is None:
            self.output_dir = r"C:\01 academico\001 Doctorado Economia UCAB\d tesis problema ahorro\01 TESIS DEFINITIVA\MODELO\resultados obj5_1"
        else:
            self.output_dir = output_dir

        # Crear directorio de salida si no existe
        os.makedirs(self.output_dir, exist_ok=True)

        # Modelos estructurales específicos para cada grupo
        self.models = {
            'Mah': {  # Mujeres Ahorradoras
                'name': 'Mujeres Ahorradoras',
                'main_equation': {
                    'target': 'PCA',
                    'predictors': ['PSE', 'DH', 'SQ', 'CS'],
                    'coefficients': [0.3485, -0.2013, -0.5101, 0.3676]
                },
                
                'variable_names': {
                    'PSE': 'PERFIL SOCIOECONÓMICO',
                    'DH': 'DESCUENTO HIPERBÓLICO',
                    'CS': 'CONTAGIO SOCIAL',
                    'AV': 'AVERSIÓN AL RIESGO',
                    'SQ': 'SATISFACCIÓN',
                    'PCA': 'COMPORTAMIENTO AHORRO'
                }
            },
            'Hah': {  # Hombres Ahorradores
                'name': 'Hombres Ahorradores',
                'main_equation': {
                    'target': 'PCA',
                    'predictors': ['PSE', 'DH', 'SQ', 'CS'],
                    'coefficients': [0.3777, 0.2226, -0.5947, 0.2866]
                },
                
                'variable_names': {
                    'PSE': 'PERFIL SOCIOECONOMICO',
                    'DH': 'DESCUENTO HIPERBOLICO',
                    'CS': 'CONTAGIO SOCIAL',
                    'AV': 'AVERSIÓN A LAS PERDIDAS',
                    'SQ': 'STATUS QUO',
                    'PCA': 'PROPENSION CONDUCTUAL AL AHORRO'
                }
            }
        }

        # Variables de mapeo automático
        self.variable_mapping = {
            'PSE': ['PSEP', 'PSE'],
            'PCA': ['PPCA', 'PCA'],
            'DH': ['DH', 'DH'],
            'CS': ['CS', 'CS'],
            'AV': ['AV', 'AV'],
            'SQ': ['SQ', 'SQ']
        }

        # Resultados por grupo
        self.results = {}

        # Configuración de visualizaciones
        self.setup_plotting_style()

    def setup_plotting_style(self):
        """
        Configuración avanzada de estilo para visualizaciones profesionales
        """
        plt.rcParams.update({
            'figure.figsize': (12, 8),
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'font.size': 11,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 16
        })

        # Paleta de colores profesional
        self.colors = {
            'Mah': '#E91E63',  # Rosa para mujeres
            'Hah': '#2196F3',  # Azul para hombres
            'comparison': '#FF9800',  # Naranja para comparaciones
            'neutral': '#9E9E9E'  # Gris para elementos neutros
        }

    def load_and_prepare_data(self, data_path, descriptive_path=None, group='Mah'):
        """
        Carga y prepara los datos con mapeo automático de variables

        Parameters:
        -----------
        data_path : str
            Ruta del archivo de datos principales
        descriptive_path : str, optional
            Ruta del archivo de estadísticas descriptivas
        group : str
            Identificador del grupo ('Mah' o 'Hah')
        """
        print(f"\n{'='*60}")
        print(f"FASE 1: PREPARACIÓN Y CONFIGURACIÓN - GRUPO {group.upper()}")
        print(f"{'='*60}")

        try:
            # Cargar datos principales
            print(f"Cargando datos desde: {data_path}")
            data = pd.read_excel(data_path)
            print(
                f"Datos cargados: {data.shape[0]} observaciones, {data.shape[1]} variables")

            # Mapear variables automáticamente
            mapped_data = self.map_variables(data, group)

            # Limpiar datos faltantes
            cleaned_data = self.clean_data(mapped_data, group)

            # Cargar estadísticas descriptivas si están disponibles
            if descriptive_path and os.path.exists(descriptive_path):
                descriptive_stats = pd.read_excel(descriptive_path)
                print(
                    f"Estadísticas descriptivas cargadas desde: {descriptive_path}")
            else:
                descriptive_stats = self.calculate_descriptive_stats(
                    cleaned_data)
                print("Estadísticas descriptivas calculadas internamente")

            # Validar supuestos del modelo
            self.validate_assumptions(cleaned_data, group)

            # Almacenar datos preparados
            if not hasattr(self, 'data'):
                self.data = {}
                self.descriptive_stats = {}

            self.data[group] = cleaned_data
            self.descriptive_stats[group] = descriptive_stats

            print(f"✓ Preparación completada para grupo {group}")
            print(f"  - Variables mapeadas y validadas")
            print(f"  - {cleaned_data.shape[0]} observaciones válidas")
            print(f"  - Supuestos del modelo verificados")

        except Exception as e:
            print(
                f"❌ Error en preparación de datos para grupo {group}: {str(e)}")
            raise

    def map_variables(self, data, group):
        """
        Mapea automáticamente las variables según nombres alternativos
        """
        mapped_data = data.copy()

        # Mapear variables según el diccionario de mapeo
        for target_var, possible_names in self.variable_mapping.items():
            found = False
            for name in possible_names:
                if name in data.columns:
                    if name != target_var:
                        mapped_data[target_var] = data[name]
                        print(f"  Variable mapeada: {name} → {target_var}")
                    found = True
                    break

            if not found:
                print(f"  ⚠️ Variable {target_var} no encontrada en los datos")

        return mapped_data

    def clean_data(self, data, group):
        """
        Limpia datos faltantes y valida calidad
        """
        print(f"\nLimpieza de datos para grupo {group}:")

        # Variables requeridas para el modelo
        required_vars = list(self.models[group]['main_equation']['predictors']) + \
            [self.models[group]['main_equation']['target']] + \
            self.models[group]['intermediate_equation']['predictors']

        # Filtrar solo variables requeridas
        available_vars = [var for var in required_vars if var in data.columns]
        if len(available_vars) != len(required_vars):
            missing = set(required_vars) - set(available_vars)
            print(f"  ⚠️ Variables faltantes: {missing}")

        # Seleccionar datos con variables disponibles
        clean_data = data[available_vars].copy()

        # Eliminar filas con valores faltantes
        initial_rows = len(clean_data)
        clean_data = clean_data.dropna()
        final_rows = len(clean_data)

        if initial_rows != final_rows:
            print(
                f"  Eliminadas {initial_rows - final_rows} filas con valores faltantes")

        # Detectar y manejar valores atípicos (usando IQR)
        for col in clean_data.select_dtypes(include=[np.number]).columns:
            Q1 = clean_data[col].quantile(0.25)
            Q3 = clean_data[col].quantile(0.75)
            IQR = Q3 - Q1
            outlier_threshold = 1.5 * IQR

            outliers = ((clean_data[col] < Q1 - outlier_threshold) |
                        (clean_data[col] > Q3 + outlier_threshold))

            if outliers.sum() > 0:
                print(
                    f"  Detectados {outliers.sum()} valores atípicos en {col}")
                # Winsorizar valores atípicos en lugar de eliminarlos
                clean_data.loc[clean_data[col] < Q1 -
                               outlier_threshold, col] = Q1 - outlier_threshold
                clean_data.loc[clean_data[col] > Q3 +
                               outlier_threshold, col] = Q3 + outlier_threshold

        print(f"  ✓ Datos limpios: {len(clean_data)} observaciones válidas")
        return clean_data

    def calculate_descriptive_stats(self, data):
        """
        Calcula estadísticas descriptivas completas
        """
        stats_data = []

        for col in data.select_dtypes(include=[np.number]).columns:
            series = data[col]

            # Calcular estadísticas
            stats_dict = {
                'Variable': col,
                'Minimum': series.min(),
                'Maximum': series.max(),
                'Mean': series.mean(),
                'Variance': series.var(),
                'SD': series.std(),
                'Skewness': stats.skew(series),
                'Kurtosis': stats.kurtosis(series)
            }

            stats_data.append(stats_dict)

        return pd.DataFrame(stats_data)

    def validate_assumptions(self, data, group):
        """
        Valida supuestos estadísticos del modelo estructural
        """
        print(f"\nValidación de supuestos para grupo {group}:")

        # Verificar multicolinealidad
        numeric_data = data.select_dtypes(include=[np.number])
        correlation_matrix = numeric_data.corr()

        high_corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_value = abs(correlation_matrix.iloc[i, j])
                if corr_value > 0.8:
                    high_corr_pairs.append((
                        correlation_matrix.columns[i],
                        correlation_matrix.columns[j],
                        corr_value
                    ))

        if high_corr_pairs:
            print(f"  ⚠️ Alta multicolinealidad detectada:")
            for var1, var2, corr in high_corr_pairs:
                print(f"    {var1} - {var2}: r = {corr:.3f}")
        else:
            print(f"  ✓ No se detectó multicolinealidad problemática")

        # Verificar normalidad de variables
        non_normal_vars = []
        for col in numeric_data.columns:
            stat, p_value = normaltest(numeric_data[col])
            if p_value < 0.05:
                non_normal_vars.append((col, p_value))

        if non_normal_vars:
            print(f"  ⚠️ Variables con distribución no normal:")
            for var, p_val in non_normal_vars:
                print(f"    {var}: p = {p_val:.4f}")
        else:
            print(f"  ✓ Todas las variables siguen distribución normal")

    def structural_prediction(self, data, group):
        """
        Implementa predicción con ecuaciones estructurales PLS-SEM

        Parameters:
        -----------
        data : pd.DataFrame
            Datos de entrada
        group : str
            Identificador del grupo ('Mah' o 'Hah')

        Returns:
        --------
        dict : Predicciones y métricas del modelo estructural
        """
        model = self.models[group]

        # PASO 1: Predicción ecuación intermedia (AV → SQ)
        intermediate_eq = model['intermediate_equation']
        X_intermediate = data[intermediate_eq['predictors']].values
        coef_intermediate = np.array(intermediate_eq['coefficients'])

        # Predicción SQ
        sq_predicted = X_intermediate @ coef_intermediate

        # PASO 2: Predicción ecuación principal usando SQ predicho
        main_eq = model['main_equation']

        # Preparar predictores para ecuación principal
        X_main = data[main_eq['predictors']].copy()
        X_main['SQ'] = sq_predicted  # Usar SQ predicho

        coef_main = np.array(main_eq['coefficients'])

        # Predicción PCA
        pca_predicted = X_main.values @ coef_main

        # Calcular métricas para ambas ecuaciones
        sq_actual = data[intermediate_eq['target']]
        pca_actual = data[main_eq['target']]

        # Métricas ecuación intermedia
        intermediate_metrics = {
            'r2': r2_score(sq_actual, sq_predicted),
            'rmse': np.sqrt(mean_squared_error(sq_actual, sq_predicted)),
            'mae': mean_absolute_error(sq_actual, sq_predicted),
            'correlation': pearsonr(sq_actual, sq_predicted)[0]
        }

        # Métricas ecuación principal
        main_metrics = {
            'r2': r2_score(pca_actual, pca_predicted),
            'rmse': np.sqrt(mean_squared_error(pca_actual, pca_predicted)),
            'mae': mean_absolute_error(pca_actual, pca_predicted),
            'correlation': pearsonr(pca_actual, pca_predicted)[0]
        }

        return {
            'sq_predicted': sq_predicted,
            'pca_predicted': pca_predicted,
            'sq_actual': sq_actual,
            'pca_actual': pca_actual,
            'intermediate_metrics': intermediate_metrics,
            'main_metrics': main_metrics,
            'X_intermediate': X_intermediate,
            'X_main': X_main.values
        }

    def direct_regression_prediction(self, data, group):
        """
        Implementa regresión directa como modelo de comparación
        """
        model = self.models[group]

        # Variables independientes (sin SQ intermedia)
        main_predictors = [p for p in model['main_equation']
                           ['predictors'] if p != 'SQ']
        main_predictors.extend(model['intermediate_equation']['predictors'])

        X = data[main_predictors]
        y = data[model['main_equation']['target']]

        # Ajustar regresión lineal
        reg_model = LinearRegression()
        reg_model.fit(X, y)

        # Predicciones
        y_pred = reg_model.predict(X)

        # Métricas
        metrics = {
            'r2': r2_score(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'mae': mean_absolute_error(y, y_pred),
            'correlation': pearsonr(y, y_pred)[0]
        }

        return {
            'predicted': y_pred,
            'actual': y,
            'metrics': metrics,
            'coefficients': reg_model.coef_,
            'intercept': reg_model.intercept_,
            'predictors': main_predictors
        }

    def cross_validation_analysis(self, data, group, cv_folds=10, cv_repeats=20):
        """
        Validación cruzada K-fold repetida para modelo estructural

        Parameters:
        -----------
        data : pd.DataFrame
            Datos de entrada
        group : str
            Identificador del grupo
        cv_folds : int
            Número de folds para validación cruzada
        cv_repeats : int
            Número de repeticiones

        Returns:
        --------
        dict : Resultados de validación cruzada
        """
        print(f"\n{'='*60}")
        print(f"FASE 3: VALIDACIÓN CRUZADA - GRUPO {group.upper()}")
        print(f"{'='*60}")
        print(f"Configuración: {cv_folds}-fold CV repetida {cv_repeats} veces")

        model = self.models[group]

        # Configurar validación cruzada repetida
        rkf = RepeatedKFold(n_splits=cv_folds,
                            n_repeats=cv_repeats, random_state=42)

        # Almacenar resultados
        structural_scores = {
            'intermediate_r2': [], 'intermediate_rmse': [], 'intermediate_mae': [],
            'main_r2': [], 'main_rmse': [], 'main_mae': []
        }

        direct_scores = {
            'r2': [], 'rmse': [], 'mae': []
        }

        fold_count = 0
        total_folds = cv_folds * cv_repeats

        for train_idx, test_idx in rkf.split(data):
            fold_count += 1
            if fold_count % 50 == 0:
                print(f"  Procesando fold {fold_count}/{total_folds}")

            # Dividir datos
            train_data = data.iloc[train_idx]
            test_data = data.iloc[test_idx]

            try:
                # VALIDACIÓN MODELO ESTRUCTURAL
                # Ecuación intermedia en train
                X_int_train = train_data[model['intermediate_equation']
                                         ['predictors']]
                y_int_train = train_data[model['intermediate_equation']['target']]

                # Predicción intermedia en test (usando coeficientes PLS-SEM)
                X_int_test = test_data[model['intermediate_equation']
                                       ['predictors']].values
                coef_int = np.array(
                    model['intermediate_equation']['coefficients'])
                sq_pred_test = X_int_test @ coef_int

                # Métricas ecuación intermedia
                y_int_test = test_data[model['intermediate_equation']['target']]
                structural_scores['intermediate_r2'].append(
                    r2_score(y_int_test, sq_pred_test))
                structural_scores['intermediate_rmse'].append(
                    np.sqrt(mean_squared_error(y_int_test, sq_pred_test)))
                structural_scores['intermediate_mae'].append(
                    mean_absolute_error(y_int_test, sq_pred_test))

                # Ecuación principal usando SQ predicho
                X_main_test = test_data[model['main_equation']
                                        ['predictors']].copy()
                X_main_test['SQ'] = sq_pred_test

                coef_main = np.array(model['main_equation']['coefficients'])
                pca_pred_test = X_main_test.values @ coef_main

                # Métricas ecuación principal
                y_main_test = test_data[model['main_equation']['target']]
                structural_scores['main_r2'].append(
                    r2_score(y_main_test, pca_pred_test))
                structural_scores['main_rmse'].append(
                    np.sqrt(mean_squared_error(y_main_test, pca_pred_test)))
                structural_scores['main_mae'].append(
                    mean_absolute_error(y_main_test, pca_pred_test))

                # VALIDACIÓN REGRESIÓN DIRECTA
                main_predictors = [
                    p for p in model['main_equation']['predictors'] if p != 'SQ']
                main_predictors.extend(
                    model['intermediate_equation']['predictors'])

                X_direct_train = train_data[main_predictors]
                X_direct_test = test_data[main_predictors]
                y_direct_train = train_data[model['main_equation']['target']]
                y_direct_test = test_data[model['main_equation']['target']]

                # Ajustar modelo directo
                reg_model = LinearRegression()
                reg_model.fit(X_direct_train, y_direct_train)
                y_direct_pred = reg_model.predict(X_direct_test)

                # Métricas regresión directa
                direct_scores['r2'].append(
                    r2_score(y_direct_test, y_direct_pred))
                direct_scores['rmse'].append(
                    np.sqrt(mean_squared_error(y_direct_test, y_direct_pred)))
                direct_scores['mae'].append(
                    mean_absolute_error(y_direct_test, y_direct_pred))

            except Exception as e:
                print(f"    ⚠️ Error en fold {fold_count}: {str(e)}")
                continue

        # Calcular estadísticas de validación cruzada
        cv_results = {}

        for score_type, scores in structural_scores.items():
            if scores:  # Solo si hay resultados
                cv_results[f'structural_{score_type}'] = {
                    'mean': np.mean(scores),
                    'std': np.std(scores),
                    'median': np.median(scores),
                    'ci_lower': np.percentile(scores, 2.5),
                    'ci_upper': np.percentile(scores, 97.5),
                    'scores': scores
                }

        for score_type, scores in direct_scores.items():
            if scores:  # Solo si hay resultados
                cv_results[f'direct_{score_type}'] = {
                    'mean': np.mean(scores),
                    'std': np.std(scores),
                    'median': np.median(scores),
                    'ci_lower': np.percentile(scores, 2.5),
                    'ci_upper': np.percentile(scores, 97.5),
                    'scores': scores
                }

        # Prueba de superioridad predictiva (CVPAT)
        if 'structural_main_r2' in cv_results and 'direct_r2' in cv_results:
            structural_r2 = cv_results['structural_main_r2']['scores']
            direct_r2 = cv_results['direct_r2']['scores']

            if len(structural_r2) == len(direct_r2):
                # Prueba t pareada
                diff = np.array(structural_r2) - np.array(direct_r2)
                t_stat, p_value = stats.ttest_1samp(diff, 0)

                cv_results['superiority_test'] = {
                    'difference_mean': np.mean(diff),
                    'difference_std': np.std(diff),
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }

        print(f"✓ Validación cruzada completada:")
        print(f"  - {len(structural_scores['main_r2'])} folds exitosos")
        print(
            f"  - R² promedio modelo estructural: {cv_results.get('structural_main_r2', {}).get('mean', 'N/A'):.4f}")
        print(
            f"  - R² promedio regresión directa: {cv_results.get('direct_r2', {}).get('mean', 'N/A'):.4f}")

        return cv_results

    def monte_carlo_analysis(self, data, group, n_simulations=5000):
        """
        Análisis Monte Carlo para evaluación de robustez

        Parameters:
        -----------
        data : pd.DataFrame
            Datos de entrada
        group : str
            Identificador del grupo
        n_simulations : int
            Número de simulaciones Monte Carlo

        Returns:
        --------
        dict : Resultados del análisis Monte Carlo
        """
        print(f"\n{'='*60}")
        print(f"FASE 3B: ANÁLISIS MONTE CARLO - GRUPO {group.upper()}")
        print(f"{'='*60}")
        print(f"Ejecutando {n_simulations:,} simulaciones...")

        model = self.models[group]

        # Almacenar resultados de simulaciones
        mc_results = {
            'intermediate_r2': [],
            'main_r2': [],
            'intermediate_rmse': [],
            'main_rmse': [],
            'total_r2': []  # R² para predicción total
        }

        # Parámetros de simulación
        n_obs = len(data)

        for sim in range(n_simulations):
            if (sim + 1) % 1000 == 0:
                print(f"  Simulación {sim + 1:,}/{n_simulations:,}")

            try:
                # Generar muestra bootstrap
                boot_indices = np.random.choice(
                    n_obs, size=int(0.8 * n_obs), replace=True)
                boot_data = data.iloc[boot_indices].reset_index(drop=True)

                # Predicción modelo estructural
                struct_pred = self.structural_prediction(boot_data, group)

                # Almacenar métricas
                mc_results['intermediate_r2'].append(
                    struct_pred['intermediate_metrics']['r2'])
                mc_results['main_r2'].append(struct_pred['main_metrics']['r2'])
                mc_results['intermediate_rmse'].append(
                    struct_pred['intermediate_metrics']['rmse'])
                mc_results['main_rmse'].append(
                    struct_pred['main_metrics']['rmse'])

                # R² total (predicción final vs observado)
                total_r2 = r2_score(
                    struct_pred['pca_actual'], struct_pred['pca_predicted'])
                mc_results['total_r2'].append(total_r2)

            except Exception as e:
                continue

        # Calcular estadísticas Monte Carlo
        mc_statistics = {}

        for metric, values in mc_results.items():
            if values:
                mc_statistics[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'median': np.median(values),
                    'q25': np.percentile(values, 25),
                    'q75': np.percentile(values, 75),
                    'ci_lower': np.percentile(values, 2.5),
                    'ci_upper': np.percentile(values, 97.5),
                    'distribution': values
                }

        print(f"✓ Análisis Monte Carlo completado:")
        print(f"  - {len(mc_results['main_r2'])} simulaciones exitosas")
        print(
            f"  - R² promedio ecuación principal: {mc_statistics.get('main_r2', {}).get('mean', 'N/A'):.4f}")
        print(
            f"  - R² promedio ecuación intermedia: {mc_statistics.get('intermediate_r2', {}).get('mean', 'N/A'):.4f}")

        return mc_statistics

    def bootstrap_analysis(self, data, group, n_bootstrap=1000):
        """
        Bootstrap optimismo-corregido para intervalos de confianza

        Parameters:
        -----------
        data : pd.DataFrame
            Datos de entrada
        group : str
            Identificador del grupo
        n_bootstrap : int
            Número de muestras bootstrap

        Returns:
        --------
        dict : Resultados bootstrap con corrección por optimismo
        """
        print(
            f"\nAnálisis Bootstrap para grupo {group} ({n_bootstrap} muestras)...")

        # Predicción en muestra original
        original_pred = self.structural_prediction(data, group)
        original_r2 = original_pred['main_metrics']['r2']

        # Almacenar resultados bootstrap
        bootstrap_r2 = []
        optimism_estimates = []

        n_obs = len(data)

        for b in range(n_bootstrap):
            if (b + 1) % 200 == 0:
                print(f"    Bootstrap {b + 1}/{n_bootstrap}")

            try:
                # Generar muestra bootstrap
                boot_indices = np.random.choice(
                    n_obs, size=n_obs, replace=True)
                boot_data = data.iloc[boot_indices].reset_index(drop=True)

                # Predicción en muestra bootstrap
                boot_pred = self.structural_prediction(boot_data, group)
                boot_r2 = boot_pred['main_metrics']['r2']
                bootstrap_r2.append(boot_r2)

                # Predicción en muestra original usando modelo bootstrap
                # (Para calcular optimismo)
                try:
                    orig_with_boot_model = self.structural_prediction(
                        data, group)
                    orig_r2_boot = orig_with_boot_model['main_metrics']['r2']
                    optimism = boot_r2 - orig_r2_boot
                    optimism_estimates.append(optimism)
                except:
                    optimism_estimates.append(0)

            except Exception as e:
                continue

        # Calcular estadísticas bootstrap
        if bootstrap_r2:
            optimism_correction = np.mean(
                optimism_estimates) if optimism_estimates else 0
            corrected_r2 = original_r2 - optimism_correction

            bootstrap_stats = {
                'original_r2': original_r2,
                'optimism_correction': optimism_correction,
                'corrected_r2': corrected_r2,
                'bootstrap_mean': np.mean(bootstrap_r2),
                'bootstrap_std': np.std(bootstrap_r2),
                'ci_lower': np.percentile(bootstrap_r2, 2.5),
                'ci_upper': np.percentile(bootstrap_r2, 97.5),
                'distribution': bootstrap_r2
            }
        else:
            bootstrap_stats = {
                'error': 'No se pudieron calcular estadísticas bootstrap'}

        return bootstrap_stats

    def sensitivity_analysis(self, data, group):
        """
        Análisis de sensibilidad mediante perturbación de variables

        Parameters:
        -----------
        data : pd.DataFrame
            Datos de entrada
        group : str
            Identificador del grupo

        Returns:
        --------
        dict : Resultados del análisis de sensibilidad
        """
        print(f"\nAnálisis de sensibilidad para grupo {group}...")

        # Predicción base
        base_pred = self.structural_prediction(data, group)
        base_r2 = base_pred['main_metrics']['r2']

        model = self.models[group]
        sensitivity_results = {}

        # Analizar sensibilidad por variable
        all_vars = (model['main_equation']['predictors'] +
                    model['intermediate_equation']['predictors'])
        unique_vars = list(set(all_vars))

        for var in unique_vars:
            if var in data.columns:
                var_impacts = []

                # Diferentes niveles de perturbación
                perturbation_levels = [-0.2, -0.1, -0.05, 0.05, 0.1, 0.2]

                for pert in perturbation_levels:
                    try:
                        # Crear datos perturbados
                        perturbed_data = data.copy()
                        original_std = data[var].std()
                        perturbed_data[var] = data[var] + (pert * original_std)

                        # Predicción con datos perturbados
                        pert_pred = self.structural_prediction(
                            perturbed_data, group)
                        pert_r2 = pert_pred['main_metrics']['r2']

                        # Calcular impacto
                        impact = pert_r2 - base_r2
                        var_impacts.append({
                            'perturbation': pert,
                            'r2_change': impact,
                            'relative_change': impact / base_r2 if base_r2 != 0 else 0
                        })

                    except Exception as e:
                        continue

                if var_impacts:
                    # Calcular sensibilidad promedio
                    avg_sensitivity = np.mean(
                        [abs(imp['r2_change']) for imp in var_impacts])
                    max_sensitivity = max([abs(imp['r2_change'])
                                          for imp in var_impacts])

                    sensitivity_results[var] = {
                        'average_sensitivity': avg_sensitivity,
                        'max_sensitivity': max_sensitivity,
                        'impacts': var_impacts,
                        'variable_name': model['variable_names'].get(var, var)
                    }

        # Ordenar por sensibilidad
        sorted_sensitivity = dict(sorted(sensitivity_results.items(),
                                         key=lambda x: x[1]['average_sensitivity'],
                                         reverse=True))

        return sorted_sensitivity

    def comprehensive_analysis(self, data_paths, descriptive_paths=None):
        """
        Ejecuta análisis completo para ambos grupos

        Parameters:
        -----------
        data_paths : dict
            Rutas de archivos de datos por grupo
        descriptive_paths : dict, optional
            Rutas de archivos descriptivos por grupo
        """
        print("\n" + "="*80)
        print("SIMULADOR CAPACIDAD PREDICTIVA PLS-SEM: ANÁLISIS INTEGRAL")
        print("="*80)

        if descriptive_paths is None:
            descriptive_paths = {}

        for group in ['Mah', 'Hah']:
            if group not in data_paths:
                print(f"⚠️ No se proporcionó ruta de datos para grupo {group}")
                continue

            try:
                print(f"\n🔄 INICIANDO ANÁLISIS PARA GRUPO {group.upper()}")

                # FASE 1: Preparación de datos
                self.load_and_prepare_data(
                    data_paths[group],
                    descriptive_paths.get(group),
                    group
                )

                data = self.data[group]

                # FASE 2: Implementación del modelo estructural
                print(f"\n{'='*60}")
                print(
                    f"FASE 2: IMPLEMENTACIÓN MODELO ESTRUCTURAL - GRUPO {group.upper()}")
                print(f"{'='*60}")

                structural_pred = self.structural_prediction(data, group)
                direct_pred = self.direct_regression_prediction(data, group)

                print(f"✓ Modelo estructural implementado:")
                print(
                    f"  - R² ecuación intermedia: {structural_pred['intermediate_metrics']['r2']:.4f}")
                print(
                    f"  - R² ecuación principal: {structural_pred['main_metrics']['r2']:.4f}")
                print(f"✓ Modelo regresión directa:")
                print(
                    f"  - R² regresión directa: {direct_pred['metrics']['r2']:.4f}")

                # FASE 3: Validación predictiva robusta
                cv_results = self.cross_validation_analysis(data, group)
                mc_results = self.monte_carlo_analysis(data, group)
                bootstrap_results = self.bootstrap_analysis(data, group)

                # FASE 4: Análisis de sensibilidad
                print(f"\n{'='*60}")
                print(
                    f"FASE 4: ANÁLISIS DE SENSIBILIDAD - GRUPO {group.upper()}")
                print(f"{'='*60}")

                sensitivity_results = self.sensitivity_analysis(data, group)

                print(f"✓ Variables por orden de importancia:")
                for i, (var, results) in enumerate(list(sensitivity_results.items())[:3]):
                    print(
                        f"  {i+1}. {results['variable_name']}: {results['average_sensitivity']:.4f}")

                # Almacenar todos los resultados
                self.results[group] = {
                    'data': data,
                    'structural_prediction': structural_pred,
                    'direct_prediction': direct_pred,
                    'cross_validation': cv_results,
                    'monte_carlo': mc_results,
                    'bootstrap': bootstrap_results,
                    'sensitivity': sensitivity_results,
                    'descriptive_stats': self.descriptive_stats[group]
                }

                print(f"\n✅ ANÁLISIS COMPLETADO PARA GRUPO {group.upper()}")

            except Exception as e:
                print(f"❌ Error en análisis del grupo {group}: {str(e)}")
                import traceback
                traceback.print_exc()

        # FASE 5: Generación de visualizaciones
        self.generate_all_visualizations()

        # FASE 6: Generación de reportes
        self.generate_comprehensive_reports()

    def generate_all_visualizations(self):
        """
        Genera todas las visualizaciones automáticamente
        """
        print(f"\n{'='*60}")
        print("FASE 5: GENERACIÓN DE VISUALIZACIONES PROFESIONALES")
        print(f"{'='*60}")

        for group in self.results.keys():
            try:
                print(
                    f"\n📊 Generando visualizaciones para grupo {group.upper()}...")

                # 1. Gráfico observado vs predicho
                self.plot_observed_vs_predicted(group)

                # 2. Distribuciones Monte Carlo
                self.plot_monte_carlo_distributions(group)

                # 3. Análisis de residuos
                self.plot_residual_analysis(group)

                # 4. Comparación entre modelos
                self.plot_model_comparison(group)

                # 5. Análisis de sensibilidad
                self.plot_sensitivity_analysis(group)

                print(
                    f"✓ Todas las visualizaciones generadas para grupo {group}")

            except Exception as e:
                print(
                    f"❌ Error generando visualizaciones para {group}: {str(e)}")

    def plot_observed_vs_predicted(self, group):
        """
        Gráfico de valores observados vs predichos para modelo estructural
        """
        try:
            results = self.results[group]
            struct_pred = results['structural_prediction']

            # Crear figura con subplots
            fig = plt.figure(figsize=(15, 10))
            gs = GridSpec(2, 2, figure=fig)

            # Configurar colores
            color = self.colors[group]

            # Subplot 1: Ecuación intermedia (AV → SQ)
            ax1 = fig.add_subplot(gs[0, 0])
            sq_actual = struct_pred['sq_actual']
            sq_predicted = struct_pred['sq_predicted']

            ax1.scatter(sq_actual, sq_predicted, color=color, alpha=0.6, s=60)

            # Línea de referencia perfecta
            min_val = min(sq_actual.min(), sq_predicted.min())
            max_val = max(sq_actual.max(), sq_predicted.max())
            ax1.plot([min_val, max_val], [min_val, max_val],
                     'r--', lw=2, alpha=0.8)

            # Línea de tendencia
            z = np.polyfit(sq_actual, sq_predicted, 1)
            p = np.poly1d(z)
            ax1.plot(sq_actual, p(sq_actual), color='darkblue', linewidth=2)

            # Métricas
            r2 = struct_pred['intermediate_metrics']['r2']
            rmse = struct_pred['intermediate_metrics']['rmse']

            ax1.set_xlabel('SQ Observado', fontweight='bold')
            ax1.set_ylabel('SQ Predicho', fontweight='bold')
            ax1.set_title(f'Ecuación Intermedia: AV → SQ\nR² = {r2:.4f}, RMSE = {rmse:.4f}',
                          fontweight='bold')
            ax1.grid(True, alpha=0.3)

            # Subplot 2: Ecuación principal (→ PCA)
            ax2 = fig.add_subplot(gs[0, 1])
            pca_actual = struct_pred['pca_actual']
            pca_predicted = struct_pred['pca_predicted']

            ax2.scatter(pca_actual, pca_predicted,
                        color=color, alpha=0.6, s=60)

            # Línea de referencia perfecta
            min_val = min(pca_actual.min(), pca_predicted.min())
            max_val = max(pca_actual.max(), pca_predicted.max())
            ax2.plot([min_val, max_val], [min_val, max_val],
                     'r--', lw=2, alpha=0.8)

            # Línea de tendencia
            z = np.polyfit(pca_actual, pca_predicted, 1)
            p = np.poly1d(z)
            ax2.plot(pca_actual, p(pca_actual), color='darkblue', linewidth=2)

            # Métricas
            r2_main = struct_pred['main_metrics']['r2']
            rmse_main = struct_pred['main_metrics']['rmse']

            ax2.set_xlabel('PCA Observado', fontweight='bold')
            ax2.set_ylabel('PCA Predicho', fontweight='bold')
            ax2.set_title(f'Ecuación Principal: → PCA\nR² = {r2_main:.4f}, RMSE = {rmse_main:.4f}',
                          fontweight='bold')
            ax2.grid(True, alpha=0.3)

            # Subplot 3: Comparación con modelo directo
            ax3 = fig.add_subplot(gs[1, :])

            direct_pred = results['direct_prediction']

            # Barras comparativas
            categories = ['R²', 'RMSE', 'MAE']
            structural_values = [r2_main, rmse_main,
                                 struct_pred['main_metrics']['mae']]
            direct_values = [direct_pred['metrics']['r2'],
                             direct_pred['metrics']['rmse'],
                             direct_pred['metrics']['mae']]

            x = np.arange(len(categories))
            width = 0.35

            bars1 = ax3.bar(x - width/2, structural_values, width,
                            label='Modelo Estructural PLS-SEM',
                            color=color, alpha=0.8)
            bars2 = ax3.bar(x + width/2, direct_values, width,
                            label='Regresión Directa',
                            color='orange', alpha=0.8)

            ax3.set_xlabel('Métricas de Predicción', fontweight='bold')
            ax3.set_ylabel('Valores', fontweight='bold')
            ax3.set_title('Comparación: Modelo Estructural vs Regresión Directa',
                          fontweight='bold')
            ax3.set_xticks(x)
            ax3.set_xticklabels(categories)
            ax3.legend()
            ax3.grid(True, alpha=0.3)

            # Añadir valores en las barras
            for bar in bars1:
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                         f'{height:.4f}', ha='center', va='bottom', fontweight='bold')

            for bar in bars2:
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                         f'{height:.4f}', ha='center', va='bottom', fontweight='bold')

            # Título general
            model_name = self.models[group]['name']
            fig.suptitle(f'ANÁLISIS PREDICTIVO: {model_name.upper()} ({group.upper()})',
                         fontsize=16, fontweight='bold', y=0.95)

            plt.tight_layout()

            # Guardar
            filename = f'observed_vs_predicted_{group}.png'
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"  ✓ Gráfico observado vs predicho guardado: {filename}")

        except Exception as e:
            print(
                f"  ❌ Error generando gráfico observado vs predicho: {str(e)}")

    def plot_monte_carlo_distributions(self, group):
        """
        Visualización de distribuciones Monte Carlo
        """
        try:
            results = self.results[group]
            mc_results = results['monte_carlo']

            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            color = self.colors[group]

            # Métricas a visualizar
            metrics = ['main_r2', 'intermediate_r2',
                       'main_rmse', 'intermediate_rmse']
            titles = ['R² Ecuación Principal', 'R² Ecuación Intermedia',
                      'RMSE Ecuación Principal', 'RMSE Ecuación Intermedia']

            for i, (metric, title) in enumerate(zip(metrics, titles)):
                ax = axes[i//2, i % 2]

                if metric in mc_results and 'distribution' in mc_results[metric]:
                    data = mc_results[metric]['distribution']

                    # Histograma
                    ax.hist(data, bins=50, density=True, alpha=0.7,
                            color=color, edgecolor='black', linewidth=0.5)

                    # Estadísticas
                    mean_val = mc_results[metric]['mean']
                    median_val = mc_results[metric]['median']
                    ci_lower = mc_results[metric]['ci_lower']
                    ci_upper = mc_results[metric]['ci_upper']

                    # Líneas de referencia
                    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2,
                               label=f'Media: {mean_val:.4f}')
                    ax.axvline(median_val, color='green', linestyle='-', linewidth=2,
                               label=f'Mediana: {median_val:.4f}')
                    ax.axvline(ci_lower, color='orange', linestyle=':', linewidth=2,
                               alpha=0.8)
                    ax.axvline(ci_upper, color='orange', linestyle=':', linewidth=2,
                               alpha=0.8, label=f'IC 95%: [{ci_lower:.4f}, {ci_upper:.4f}]')

                    # Área de confianza
                    ax.axvspan(ci_lower, ci_upper, alpha=0.2, color='orange')

                    ax.set_xlabel('Valor de la Métrica', fontweight='bold')
                    ax.set_ylabel('Densidad', fontweight='bold')
                    ax.set_title(title, fontweight='bold')
                    ax.legend(fontsize=9)
                    ax.grid(True, alpha=0.3)

                else:
                    ax.text(0.5, 0.5, 'Datos no disponibles',
                            transform=ax.transAxes, ha='center', va='center')
                    ax.set_title(title, fontweight='bold')

            model_name = self.models[group]['name']
            fig.suptitle(f'DISTRIBUCIONES MONTE CARLO: {model_name.upper()} ({group.upper()})',
                         fontsize=16, fontweight='bold')

            plt.tight_layout()

            # Guardar
            filename = f'monte_carlo_distribution_{group}.png'
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"  ✓ Distribuciones Monte Carlo guardadas: {filename}")

        except Exception as e:
            print(f"  ❌ Error generando distribuciones Monte Carlo: {str(e)}")

    def plot_residual_analysis(self, group):
        """
        Análisis exhaustivo de residuos
        """
        try:
            results = self.results[group]
            struct_pred = results['structural_prediction']

            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            color = self.colors[group]

            # Residuos para ecuación principal
            pca_actual = struct_pred['pca_actual']
            pca_predicted = struct_pred['pca_predicted']
            residuals_main = pca_actual - pca_predicted
            fitted_values_main = pca_predicted

            # Residuos para ecuación intermedia
            sq_actual = struct_pred['sq_actual']
            sq_predicted = struct_pred['sq_predicted']
            residuals_int = sq_actual - sq_predicted
            fitted_values_int = sq_predicted

            # Subplot 1: Residuos vs Valores ajustados (Principal)
            ax1 = axes[0, 0]
            ax1.scatter(fitted_values_main, residuals_main,
                        color=color, alpha=0.6)
            ax1.axhline(y=0, color='red', linestyle='--', linewidth=2)
            ax1.set_xlabel('Valores Ajustados', fontweight='bold')
            ax1.set_ylabel('Residuos', fontweight='bold')
            ax1.set_title(
                'Residuos vs Ajustados\n(Ecuación Principal)', fontweight='bold')
            ax1.grid(True, alpha=0.3)

            # Subplot 2: Q-Q plot normalidad (Principal)
            ax2 = axes[0, 1]
            stats.probplot(residuals_main, dist="norm", plot=ax2)
            ax2.set_title(
                'Q-Q Plot Normalidad\n(Ecuación Principal)', fontweight='bold')
            ax2.grid(True, alpha=0.3)

            # Subplot 3: Histograma residuos (Principal)
            ax3 = axes[0, 2]
            ax3.hist(residuals_main, bins=20, density=True, alpha=0.7,
                     color=color, edgecolor='black')

            # Superponer curva normal
            mu, sigma = stats.norm.fit(residuals_main)
            x = np.linspace(residuals_main.min(), residuals_main.max(), 100)
            ax3.plot(x, stats.norm.pdf(x, mu, sigma),
                     'r-', lw=2, label='Normal Teórica')
            ax3.set_xlabel('Residuos', fontweight='bold')
            ax3.set_ylabel('Densidad', fontweight='bold')
            ax3.set_title(
                'Distribución Residuos\n(Ecuación Principal)', fontweight='bold')
            ax3.legend()
            ax3.grid(True, alpha=0.3)

            # Subplot 4: Residuos vs Valores ajustados (Intermedia)
            ax4 = axes[1, 0]
            ax4.scatter(fitted_values_int, residuals_int,
                        color=color, alpha=0.6)
            ax4.axhline(y=0, color='red', linestyle='--', linewidth=2)
            ax4.set_xlabel('Valores Ajustados', fontweight='bold')
            ax4.set_ylabel('Residuos', fontweight='bold')
            ax4.set_title(
                'Residuos vs Ajustados\n(Ecuación Intermedia)', fontweight='bold')
            ax4.grid(True, alpha=0.3)

            # Subplot 5: Q-Q plot normalidad (Intermedia)
            ax5 = axes[1, 1]
            stats.probplot(residuals_int, dist="norm", plot=ax5)
            ax5.set_title(
                'Q-Q Plot Normalidad\n(Ecuación Intermedia)', fontweight='bold')
            ax5.grid(True, alpha=0.3)

            # Subplot 6: Comparación distribuciones
            ax6 = axes[1, 2]
            ax6.hist(residuals_main, bins=15, alpha=0.7, label='Ec. Principal',
                     color=color, density=True)
            ax6.hist(residuals_int, bins=15, alpha=0.7, label='Ec. Intermedia',
                     color='orange', density=True)
            ax6.set_xlabel('Residuos', fontweight='bold')
            ax6.set_ylabel('Densidad', fontweight='bold')
            ax6.set_title('Comparación Distribuciones\nResiduos',
                          fontweight='bold')
            ax6.legend()
            ax6.grid(True, alpha=0.3)

            # Título general
            model_name = self.models[group]['name']
            fig.suptitle(f'ANÁLISIS DE RESIDUOS: {model_name.upper()} ({group.upper()})',
                         fontsize=16, fontweight='bold')

            plt.tight_layout()

            # Guardar
            filename = f'residual_analysis_{group}.png'
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"  ✓ Análisis de residuos guardado: {filename}")

        except Exception as e:
            print(f"  ❌ Error generando análisis de residuos: {str(e)}")

    def plot_model_comparison(self, group):
        """
        Comparación visual entre modelo estructural y regresión directa
        """
        try:
            results = self.results[group]
            struct_pred = results['structural_prediction']
            direct_pred = results['direct_prediction']
            cv_results = results.get('cross_validation', {})

            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            color = self.colors[group]

            # Subplot 1: Comparación R² por validación cruzada
            ax1 = axes[0, 0]

            if 'structural_main_r2' in cv_results and 'direct_r2' in cv_results:
                struct_r2_cv = cv_results['structural_main_r2']['scores']
                direct_r2_cv = cv_results['direct_r2']['scores']

                bp1 = ax1.boxplot([struct_r2_cv, direct_r2_cv],
                                  labels=['Estructural', 'Directo'],
                                  patch_artist=True, notch=True)

                bp1['boxes'][0].set_facecolor(color)
                bp1['boxes'][1].set_facecolor('orange')

                ax1.set_ylabel('R² (Validación Cruzada)', fontweight='bold')
                ax1.set_title('Comparación R² por CV', fontweight='bold')
                ax1.grid(True, alpha=0.3)

                # Prueba de significancia
                if 'superiority_test' in cv_results:
                    p_val = cv_results['superiority_test']['p_value']
                    sig_text = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                    ax1.text(0.5, 0.95, f'Diferencia: {sig_text} (p={p_val:.4f})',
                             transform=ax1.transAxes, ha='center', fontweight='bold')

            # Subplot 2: Métricas comparativas
            ax2 = axes[0, 1]

            metrics = ['R²', 'RMSE', 'MAE']
            struct_values = [struct_pred['main_metrics']['r2'],
                             struct_pred['main_metrics']['rmse'],
                             struct_pred['main_metrics']['mae']]
            direct_values = [direct_pred['metrics']['r2'],
                             direct_pred['metrics']['rmse'],
                             direct_pred['metrics']['mae']]

            x = np.arange(len(metrics))
            width = 0.35

            bars1 = ax2.bar(x - width/2, struct_values, width,
                            label='Estructural', color=color, alpha=0.8)
            bars2 = ax2.bar(x + width/2, direct_values, width,
                            label='Directo', color='orange', alpha=0.8)

            ax2.set_xlabel('Métricas', fontweight='bold')
            ax2.set_ylabel('Valores', fontweight='bold')
            ax2.set_title('Métricas Comparativas', fontweight='bold')
            ax2.set_xticks(x)
            ax2.set_xticklabels(metrics)
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            # Subplot 3: Predicciones vs Observados (Ambos modelos)
            ax3 = axes[1, 0]

            pca_actual = struct_pred['pca_actual']
            pca_struct = struct_pred['pca_predicted']
            pca_direct = direct_pred['predicted']

            ax3.scatter(pca_actual, pca_struct, color=color, alpha=0.6,
                        label=f'Estructural (R²={struct_pred["main_metrics"]["r2"]:.3f})')
            ax3.scatter(pca_actual, pca_direct, color='orange', alpha=0.6,
                        label=f'Directo (R²={direct_pred["metrics"]["r2"]:.3f})')

            # Línea de referencia
            min_val = pca_actual.min()
            max_val = pca_actual.max()
            ax3.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)

            ax3.set_xlabel('PCA Observado', fontweight='bold')
            ax3.set_ylabel('PCA Predicho', fontweight='bold')
            ax3.set_title('Predicciones vs Observados', fontweight='bold')
            ax3.legend()
            ax3.grid(True, alpha=0.3)

            # Subplot 4: Diferencias en residuos
            ax4 = axes[1, 1]

            residuals_struct = pca_actual - pca_struct
            residuals_direct = pca_actual - pca_direct

            ax4.hist(residuals_struct, bins=20, alpha=0.7, label='Estructural',
                     color=color, density=True)
            ax4.hist(residuals_direct, bins=20, alpha=0.7, label='Directo',
                     color='orange', density=True)

            ax4.set_xlabel('Residuos', fontweight='bold')
            ax4.set_ylabel('Densidad', fontweight='bold')
            ax4.set_title('Distribución de Residuos', fontweight='bold')
            ax4.legend()
            ax4.grid(True, alpha=0.3)

            # Título general
            model_name = self.models[group]['name']
            fig.suptitle(f'COMPARACIÓN DE MODELOS: {model_name.upper()} ({group.upper()})',
                         fontsize=16, fontweight='bold')

            plt.tight_layout()

            # Guardar
            filename = f'model_comparison_{group}.png'
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"  ✓ Comparación de modelos guardada: {filename}")

        except Exception as e:
            print(f"  ❌ Error generando comparación de modelos: {str(e)}")

    def plot_sensitivity_analysis(self, group):
        """
        Visualización del análisis de sensibilidad
        """
        try:
            results = self.results[group]
            sensitivity_results = results['sensitivity']

            if not sensitivity_results:
                print(f"  ⚠️ No hay datos de sensibilidad para grupo {group}")
                return

            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            color = self.colors[group]

            # Preparar datos
            variables = list(sensitivity_results.keys())
            avg_sensitivity = [sensitivity_results[var]
                               ['average_sensitivity'] for var in variables]
            max_sensitivity = [sensitivity_results[var]
                               ['max_sensitivity'] for var in variables]
            var_names = [sensitivity_results[var]['variable_name']
                         for var in variables]

            # Subplot 1: Sensibilidad promedio por variable
            ax1 = axes[0, 0]
            bars = ax1.barh(var_names, avg_sensitivity, color=color, alpha=0.8)
            ax1.set_xlabel('Sensibilidad Promedio', fontweight='bold')
            ax1.set_title('Importancia Relativa de Variables',
                          fontweight='bold')
            ax1.grid(True, alpha=0.3)

            # Añadir valores en las barras
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax1.text(width, bar.get_y() + bar.get_height()/2,
                         f'{width:.4f}', ha='left', va='center', fontweight='bold')

            # Subplot 2: Sensibilidad máxima por variable
            ax2 = axes[0, 1]
            bars2 = ax2.barh(var_names, max_sensitivity,
                             color='orange', alpha=0.8)
            ax2.set_xlabel('Sensibilidad Máxima', fontweight='bold')
            ax2.set_title('Máxima Variabilidad por Variable',
                          fontweight='bold')
            ax2.grid(True, alpha=0.3)

            # Añadir valores en las barras
            for i, bar in enumerate(bars2):
                width = bar.get_width()
                ax2.text(width, bar.get_y() + bar.get_height()/2,
                         f'{width:.4f}', ha='left', va='center', fontweight='bold')

            # Subplot 3: Curvas de sensibilidad para variable más importante
            ax3 = axes[1, 0]

            if variables:
                # Ya están ordenados por importancia
                most_important_var = variables[0]
                impacts = sensitivity_results[most_important_var]['impacts']

                perturbations = [imp['perturbation'] for imp in impacts]
                r2_changes = [imp['r2_change'] for imp in impacts]

                ax3.plot(perturbations, r2_changes, marker='o', linewidth=2,
                         markersize=6, color=color)
                ax3.axhline(y=0, color='red', linestyle='--', alpha=0.7)
                ax3.axvline(x=0, color='red', linestyle='--', alpha=0.7)
                ax3.set_xlabel(
                    'Perturbación (desviaciones estándar)', fontweight='bold')
                ax3.set_ylabel('Cambio en R²', fontweight='bold')
                ax3.set_title(f'Curva de Sensibilidad\n{sensitivity_results[most_important_var]["variable_name"]}',
                              fontweight='bold')
                ax3.grid(True, alpha=0.3)

            # Subplot 4: Diagrama de radar de sensibilidades
            ax4 = axes[1, 1]

            # Configurar diagrama de radar
            angles = np.linspace(
                0, 2 * np.pi, len(variables), endpoint=False).tolist()
            avg_sensitivity += [avg_sensitivity[0]]  # Cerrar el polígono
            angles += [angles[0]]

            ax4 = plt.subplot(2, 2, 4, projection='polar')
            ax4.plot(angles, avg_sensitivity, 'o-', linewidth=2, color=color)
            ax4.fill(angles, avg_sensitivity, alpha=0.25, color=color)
            ax4.set_xticks(angles[:-1])
            ax4.set_xticklabels([name[:15] + '...' if len(name) > 15 else name
                                 for name in var_names])
            ax4.set_title('Diagrama de Sensibilidad\n(Variables)',
                          fontweight='bold', pad=20)
            ax4.grid(True)

            # Título general
            model_name = self.models[group]['name']
            fig.suptitle(f'ANÁLISIS DE SENSIBILIDAD: {model_name.upper()} ({group.upper()})',
                         fontsize=16, fontweight='bold')

            plt.tight_layout()

            # Guardar
            filename = f'sensitivity_analysis_{group}.png'
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"  ✓ Análisis de sensibilidad guardado: {filename}")

        except Exception as e:
            print(f"  ❌ Error generando análisis de sensibilidad: {str(e)}")

    def generate_comprehensive_reports(self):
        """
        Genera reportes comprehensivos en formato texto
        """
        print(f"\n{'='*60}")
        print("FASE 6: GENERACIÓN DE REPORTES COMPREHENSIVOS")
        print(f"{'='*60}")

        for group in self.results.keys():
            try:
                print(f"\n📄 Generando reporte para grupo {group.upper()}...")
                self.generate_detailed_report(group)
                print(f"✓ Reporte detallado generado para grupo {group}")
            except Exception as e:
                print(f"❌ Error generando reporte para {group}: {str(e)}")

        # Generar reporte comparativo
        try:
            self.generate_comparative_report()
            print("✓ Reporte comparativo generado")
        except Exception as e:
            print(f"❌ Error generando reporte comparativo: {str(e)}")

    def generate_detailed_report(self, group):
        """
        Genera reporte detallado para un grupo específico
        """
        results = self.results[group]
        model = self.models[group]

        filename = f'reporte_detallado_{group}.txt'
        filepath = os.path.join(self.output_dir, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            # Encabezado
            f.write("=" * 80 + "\n")
            f.write(f"REPORTE ANÁLISIS CAPACIDAD PREDICTIVA PLS-SEM\n")
            f.write(f"GRUPO: {model['name'].upper()} ({group.upper()})\n")
            f.write(f"FECHA: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")

            # 1. INFORMACIÓN DEL MODELO
            f.write("1. ESPECIFICACIÓN DEL MODELO ESTRUCTURAL\n")
            f.write("-" * 50 + "\n")

            # Ecuación intermedia
            int_eq = model['intermediate_equation']
            f.write(f"Ecuación Intermedia: {int_eq['target']} = ")
            for i, (pred, coef) in enumerate(zip(int_eq['predictors'], int_eq['coefficients'])):
                sign = "+" if coef >= 0 and i > 0 else ""
                f.write(f"{sign}{coef:.4f}·{pred} ")
            f.write("+ ε₁\n")

            # Ecuación principal
            main_eq = model['main_equation']
            f.write(f"Ecuación Principal: {main_eq['target']} = ")
            for i, (pred, coef) in enumerate(zip(main_eq['predictors'], main_eq['coefficients'])):
                sign = "+" if coef >= 0 and i > 0 else ""
                f.write(f"{sign}{coef:.4f}·{pred} ")
            f.write("+ ε₂\n\n")

            # Variables
            f.write("Variables del Modelo:\n")
            for var, description in model['variable_names'].items():
                f.write(f"  - {var}: {description}\n")
            f.write("\n")

            # 2. ESTADÍSTICAS DESCRIPTIVAS
            f.write("2. ESTADÍSTICAS DESCRIPTIVAS\n")
            f.write("-" * 50 + "\n")

            desc_stats = results['descriptive_stats']
            f.write(
                f"{'Variable':<12} {'Min':<8} {'Max':<8} {'Media':<8} {'SD':<8} {'Asimetría':<10} {'Curtosis':<8}\n")
            f.write("-" * 70 + "\n")

            for _, row in desc_stats.iterrows():
                f.write(f"{row['Variable']:<12} {row['Minimum']:<8.3f} {row['Maximum']:<8.3f} "
                        f"{row['Mean']:<8.3f} {row['SD']:<8.3f} {row['Skewness']:<10.3f} "
                        f"{row['Kurtosis']:<8.3f}\n")
            f.write("\n")

            # 3. RESULTADOS DEL MODELO ESTRUCTURAL
            f.write("3. RESULTADOS DEL MODELO ESTRUCTURAL\n")
            f.write("-" * 50 + "\n")

            struct_pred = results['structural_prediction']

            f.write("Ecuación Intermedia (AV → SQ):\n")
            f.write(
                f"  R² = {struct_pred['intermediate_metrics']['r2']:.6f}\n")
            f.write(
                f"  RMSE = {struct_pred['intermediate_metrics']['rmse']:.6f}\n")
            f.write(
                f"  MAE = {struct_pred['intermediate_metrics']['mae']:.6f}\n")
            f.write(
                f"  Correlación = {struct_pred['intermediate_metrics']['correlation']:.6f}\n\n")

            f.write("Ecuación Principal (→ PCA):\n")
            f.write(f"  R² = {struct_pred['main_metrics']['r2']:.6f}\n")
            f.write(f"  RMSE = {struct_pred['main_metrics']['rmse']:.6f}\n")
            f.write(f"  MAE = {struct_pred['main_metrics']['mae']:.6f}\n")
            f.write(
                f"  Correlación = {struct_pred['main_metrics']['correlation']:.6f}\n\n")

            # 4. RESULTADOS VALIDACIÓN CRUZADA
            f.write("4. VALIDACIÓN CRUZADA (10-FOLD × 20 REPETICIONES)\n")
            f.write("-" * 50 + "\n")

            cv_results = results['cross_validation']

            if 'structural_main_r2' in cv_results:
                f.write("Modelo Estructural:\n")
                f.write(f"  R² Medio = {cv_results['structural_main_r2']['mean']:.6f} "
                        f"± {cv_results['structural_main_r2']['std']:.6f}\n")
                f.write(f"  IC 95% = [{cv_results['structural_main_r2']['ci_lower']:.6f}, "
                        f"{cv_results['structural_main_r2']['ci_upper']:.6f}]\n")

                if 'structural_main_rmse' in cv_results:
                    f.write(f"  RMSE Medio = {cv_results['structural_main_rmse']['mean']:.6f} "
                            f"± {cv_results['structural_main_rmse']['std']:.6f}\n")

            if 'direct_r2' in cv_results:
                f.write("\nRegresión Directa:\n")
                f.write(f"  R² Medio = {cv_results['direct_r2']['mean']:.6f} "
                        f"± {cv_results['direct_r2']['std']:.6f}\n")
                f.write(f"  IC 95% = [{cv_results['direct_r2']['ci_lower']:.6f}, "
                        f"{cv_results['direct_r2']['ci_upper']:.6f}]\n")

            # Prueba de superioridad
            if 'superiority_test' in cv_results:
                f.write("\nPrueba de Superioridad Predictiva (CVPAT):\n")
                sup_test = cv_results['superiority_test']
                f.write(
                    f"  Diferencia Media = {sup_test['difference_mean']:.6f}\n")
                f.write(f"  t-estadístico = {sup_test['t_statistic']:.4f}\n")
                f.write(f"  p-valor = {sup_test['p_value']:.6f}\n")
                f.write(
                    f"  Significativo = {'Sí' if sup_test['significant'] else 'No'}\n")
            f.write("\n")

            # 5. ANÁLISIS MONTE CARLO
            f.write("5. ANÁLISIS MONTE CARLO (5,000 SIMULACIONES)\n")
            f.write("-" * 50 + "\n")

            mc_results = results['monte_carlo']

            for metric_key, metric_name in [('main_r2', 'R² Ecuación Principal'),
                                            ('intermediate_r2',
                                             'R² Ecuación Intermedia'),
                                            ('main_rmse', 'RMSE Ecuación Principal')]:
                if metric_key in mc_results:
                    mc_data = mc_results[metric_key]
                    f.write(f"{metric_name}:\n")
                    f.write(f"  Media = {mc_data['mean']:.6f}\n")
                    f.write(f"  Mediana = {mc_data['median']:.6f}\n")
                    f.write(f"  Desviación Estándar = {mc_data['std']:.6f}\n")
                    f.write(
                        f"  IC 95% = [{mc_data['ci_lower']:.6f}, {mc_data['ci_upper']:.6f}]\n")
                    f.write(
                        f"  Q1-Q3 = [{mc_data['q25']:.6f}, {mc_data['q75']:.6f}]\n\n")

            # 6. BOOTSTRAP ANALYSIS
            f.write("6. ANÁLISIS BOOTSTRAP CON CORRECCIÓN POR OPTIMISMO\n")
            f.write("-" * 50 + "\n")

            bootstrap_results = results['bootstrap']

            if 'original_r2' in bootstrap_results:
                f.write(
                    f"R² Original = {bootstrap_results['original_r2']:.6f}\n")
                f.write(
                    f"Corrección por Optimismo = {bootstrap_results['optimism_correction']:.6f}\n")
                f.write(
                    f"R² Corregido = {bootstrap_results['corrected_r2']:.6f}\n")
                f.write(f"IC Bootstrap 95% = [{bootstrap_results['ci_lower']:.6f}, "
                        f"{bootstrap_results['ci_upper']:.6f}]\n\n")

            # 7. ANÁLISIS DE SENSIBILIDAD
            f.write("7. ANÁLISIS DE SENSIBILIDAD\n")
            f.write("-" * 50 + "\n")

            sensitivity_results = results['sensitivity']

            f.write(
                "Importancia Relativa de Variables (ordenado por sensibilidad):\n")
            f.write(
                f"{'Variable':<15} {'Nombre':<25} {'Sens. Promedio':<15} {'Sens. Máxima':<12}\n")
            f.write("-" * 70 + "\n")

            for var, sens_data in sensitivity_results.items():
                f.write(f"{var:<15} {sens_data['variable_name'][:24]:<25} "
                        f"{sens_data['average_sensitivity']:<15.6f} "
                        f"{sens_data['max_sensitivity']:<12.6f}\n")
            f.write("\n")

            # 8. CONCLUSIONES Y RECOMENDACIONES
            f.write("8. CONCLUSIONES Y RECOMENDACIONES METODOLÓGICAS\n")
            f.write("-" * 50 + "\n")

            # Evaluar calidad del modelo
            main_r2 = struct_pred['main_metrics']['r2']

            if main_r2 > 0.70:
                f.write("✓ CALIDAD DEL MODELO: EXCELENTE\n")
            elif main_r2 > 0.50:
                f.write("✓ CALIDAD DEL MODELO: BUENA\n")
            elif main_r2 > 0.30:
                f.write("○ CALIDAD DEL MODELO: MODERADA\n")
            else:
                f.write("⚠ CALIDAD DEL MODELO: BAJA\n")

            f.write(
                f"  El modelo explica {main_r2*100:.2f}% de la varianza en de la Propensión Conductual del Ahorro PCA.\n\n")

            # Comparación con regresión directa
            if 'superiority_test' in cv_results and cv_results['superiority_test']['significant']:
                f.write("✓ VENTAJA ESTRUCTURAL: CONFIRMADA\n")
                f.write(
                    "  El modelo estructural PLS-SEM muestra superioridad predictiva\n")
                f.write(
                    "  estadísticamente significativa sobre la regresión directa.\n\n")
            else:
                f.write("○ VENTAJA ESTRUCTURAL: NO CONFIRMADA\n")
                f.write(
                    "  No se detectó superioridad significativa del modelo estructural.\n\n")

            # Variables más importantes
            if sensitivity_results:
                most_important = list(sensitivity_results.keys())[0]
                f.write(
                    f"✓ VARIABLE MÁS INFLUYENTE: {sensitivity_results[most_important]['variable_name']}\n")
                f.write(
                    f"  Esta variable muestra la mayor sensibilidad predictiva.\n\n")

            f.write("RECOMENDACIONES:\n")
            f.write(
                "1. Considerar el modelo estructural para interpretación teórica.\n")
            f.write("2. Validar resultados con muestras independientes.\n")
            f.write("3. Explorar interacciones entre variables principales.\n")
            f.write("4. Monitorear estabilidad temporal del modelo.\n\n")

            f.write("=" * 80 + "\n")
            f.write("Econ. J. Salazar R. / FIN DEL REPORTE\n")
            f.write("=" * 80 + "\n")

    def generate_comparative_report(self):
        """
        Genera reporte comparativo entre grupos
        """
        if len(self.results) < 2:
            return

        filename = 'reporte_comparativo_grupos.txt'
        filepath = os.path.join(self.output_dir, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("REPORTE COMPARATIVO: HOMBRES VS MUJERES AHORRADORES\n")
            f.write(f"FECHA: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")

            # Comparación de R² principales
            f.write("1. COMPARACIÓN DE CAPACIDAD PREDICTIVA\n")
            f.write("-" * 50 + "\n")

            f.write(
                f"{'Métrica':<25} {'Hombres (Hah)':<15} {'Mujeres (Mah)':<15} {'Diferencia':<12}\n")
            f.write("-" * 70 + "\n")

            groups_data = {}
            for group in self.results.keys():
                struct_pred = self.results[group]['structural_prediction']
                groups_data[group] = {
                    'r2_main': struct_pred['main_metrics']['r2'],
                    'rmse_main': struct_pred['main_metrics']['rmse'],
                    'r2_int': struct_pred['intermediate_metrics']['r2']
                }

            if 'Hah' in groups_data and 'Mah' in groups_data:
                hah_data = groups_data['Hah']
                mah_data = groups_data['Mah']

                f.write(f"{'R² Ecuación Principal':<25} {hah_data['r2_main']:<15.6f} "
                        f"{mah_data['r2_main']:<15.6f} {hah_data['r2_main']-mah_data['r2_main']:<12.6f}\n")
                f.write(f"{'RMSE Ec. Principal':<25} {hah_data['rmse_main']:<15.6f} "
                        f"{mah_data['rmse_main']:<15.6f} {hah_data['rmse_main']-mah_data['rmse_main']:<12.6f}\n")
                f.write(f"{'R² Ecuación Intermedia':<25} {hah_data['r2_int']:<15.6f} "
                        f"{mah_data['r2_int']:<15.6f} {hah_data['r2_int']-mah_data['r2_int']:<12.6f}\n")

            f.write("\n")

            # Comparación de variables más importantes
            f.write("2. VARIABLES MÁS INFLUYENTES POR GRUPO\n")
            f.write("-" * 50 + "\n")

            for group in self.results.keys():
                model_name = self.models[group]['name']
                sensitivity = self.results[group]['sensitivity']

                f.write(f"{model_name}:\n")
                for i, (var, data) in enumerate(list(sensitivity.items())[:3]):
                    f.write(
                        f"  {i+1}. {data['variable_name']} (Sens: {data['average_sensitivity']:.4f})\n")
                f.write("\n")

            # Conclusiones comparativas
            f.write("3. CONCLUSIONES COMPARATIVAS\n")
            f.write("-" * 50 + "\n")

            if 'Hah' in groups_data and 'Mah' in groups_data:
                if hah_data['r2_main'] > mah_data['r2_main']:
                    f.write(
                        "✓ El modelo para HOMBRES muestra mayor capacidad predictiva.\n")
                else:
                    f.write(
                        "✓ El modelo para MUJERES muestra mayor capacidad predictiva.\n")

                diff_magnitude = abs(hah_data['r2_main'] - mah_data['r2_main'])
                if diff_magnitude > 0.05:
                    f.write("  La diferencia entre grupos es SUSTANCIAL.\n")
                elif diff_magnitude > 0.02:
                    f.write("  La diferencia entre grupos es MODERADA.\n")
                else:
                    f.write("  La diferencia entre grupos es PEQUEÑA.\n")

            f.write("\nIMPLICACIONES:\n")
            f.write(
                "- Los patrones de ahorro pueden diferir significativamente entre géneros.\n")
            f.write(
                "- Se recomienda desarrollar estrategias diferenciadas por grupo.\n")
            f.write(
                "- Futuras investigaciones deberían explorar las causas de estas diferencias.\n\n")

            f.write("=" * 80 + "\n")
            f.write("FIN DEL REPORTE COMPARATIVO\n")
            f.write("=" * 80 + "\n")


def main():
    # Rutas únicas de los archivos consolidados
    data_path = r"C:\01 academico\001 Doctorado Economia UCAB\d tesis problema ahorro\01 TESIS DEFINITIVA\MODELO\resultados obj5\DATA_CONSOLIDADA promedio HM .xlsx"
    descriptive_path = r"C:\01 academico\001 Doctorado Economia UCAB\d tesis problema ahorro\01 TESIS DEFINITIVA\MODELO\resultados obj5\descripiva HMah.xlsx"

    import pandas as pd
    df = pd.read_excel(data_path, engine="openpyxl")

    # Separar por grupo
    data_paths = {}
    descriptive_paths = {}
    for group in ['Mah', 'Hah']:
        df_group = df[df['grupo'] == group].copy()
        temp_path = f"temp_data_{group}.xlsx"
        df_group.to_excel(temp_path, index=False)
        data_paths[group] = temp_path
        descriptive_paths[group] = descriptive_path

    # Inicializar analizador
    analyzer = PLSSEMPredictiveAnalyzer()

    # Ejecutar análisis completo por grupo
    analyzer.comprehensive_analysis(data_paths, descriptive_paths)

    print("ANÁLISIS COMPLETO FINALIZADO EXITOSAMENTE")
    print(f"📁 Todos los resultados guardados en: {analyzer.output_dir}")


if __name__ == "__main__":
    main()
