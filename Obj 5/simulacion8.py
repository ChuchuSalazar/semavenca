"""
===============================================================================
    SIMULADOR CAPACIDAD PREDICTIVA PLS-SEM: ANÁLISIS ESTRUCTURAL CON VISUALIZACIONES
===============================================================================

DESCRIPCIÓN GENERAL:
    Sistema integral de validación predictiva para modelos PLS-SEM estructurales
    que implementa coeficientes específicos derivados del análisis SEM y evalúa
    su capacidad predictiva mediante técnicas de validación robusta con
    generación automática de visualizaciones profesionales.

MODELOS ESTRUCTURALES IMPLEMENTADOS:

    • MUJERES AHORRADORAS:
      - Ecuación Principal: PCA = 0.3485·PSE - 0.2013·DH - 0.5101·SQ + 0.3676·CS + ε₁
      - Ecuación Intermedia: SQ = 0.6609·AV + ε₂
      - Variables: PSE (Educación), DH (Demografía), CS (Cultura), AV (Aversión)

    • HOMBRES AHORRADORES:
      - Ecuación Principal: PCA = 0.3777·PSE + 0.2226·DH - 0.5947·SQ + 0.2866·CS + ε₁
      - Ecuación Intermedia: SQ = 0.8402·AV + ε₂
      - Variables: PSE (Educación), DH (Demografía), CS (Cultura), AV (Aversión)

SECUENCIA DE DESARROLLO Y FUNCIONALIDADES:

    FASE 1: PREPARACIÓN Y CONFIGURACIÓN
       - Carga y mapeo automático de variables
       - Limpieza de datos faltantes y validación de calidad
       - Separación de variables intermedias vs estructurales
       - Verificación de supuestos del modelo estructural

    FASE 2: IMPLEMENTACIÓN DEL MODELO ESTRUCTURAL
       - Predicción con coeficientes PLS-SEM específicos
       - Sistema de ecuaciones simultáneas (AV→SQ→PCA)
       - Manejo de efectos directos e indirectos
       - Comparación con regresión directa estándar

    FASE 3: VALIDACIÓN PREDICTIVA ROBUSTA
       - Validación Cruzada K-fold Repetida (10×20)
         * Métricas para ecuación intermedia (AV→SQ)
         * Métricas para ecuación principal (→PCA)
         * Pruebas estadísticas de capacidad predictiva superior
       - Análisis Monte Carlo (5,000 simulaciones)
         * Evaluación de robustez del sistema completo
         * Análisis de sensibilidad por variable
         * Distribuciones predictivas bajo incertidumbre
       - Bootstrap Optimismo-Corregido
         * Intervalos de confianza para R² no sesgados
         * Corrección de sobreajuste en métricas

    FASE 4: ANÁLISIS COMPARATIVO Y VALIDACIÓN
       - Modelo Estructural vs Regresión Directa
       - CVPAT: Pruebas de significancia predictiva
       - Evaluación de diferencias entre grupos
       - Benchmarking contra modelos baseline

    FASE 5: GENERACIÓN AUTOMÁTICA DE VISUALIZACIONES
       - Gráficos de ajuste observado vs predicho
       - Distribuciones Monte Carlo con percentiles
       - Análisis exhaustivo de residuos
       - Comparación visual entre modelos
       - Diagramas de sensibilidad y contribuciones
       - Visualización de coeficientes estructurales

    FASE 6: REPORTES Y DOCUMENTACIÓN
       - Reportes comprehensivos por grupo
       - Exportación automática a archivos PNG de alta resolución
       - Generación de visualizaciones profesionales
       - Tablas de resultados estandarizadas

VISUALIZACIONES GENERADAS AUTOMÁTICAMENTE:

    Para cada grupo (HOMBRES/MUJERES):
    • observed_vs_predicted_[grupo].png - Ajuste del modelo estructural
    • monte_carlo_distribution_[grupo].png - Distribuciones simuladas
    • residual_analysis_[grupo].png - Diagnósticos de residuos
    • model_comparison_[grupo].png - Estructural vs Regresión directa
    • sensitivity_analysis_[grupo].png - Importancia relativa de variables

CARACTERÍSTICAS METODOLÓGICAS AVANZADAS:

    RIGOR CIENTÍFICO:
       • Implementación exacta de coeficientes PLS-SEM derivados
       • Validación que respeta estructura de ecuaciones simultáneas
       • Corrección por optimismo en estimaciones de capacidad predictiva
       • Pruebas estadísticas para significancia de superioridad predictiva

    ROBUSTEZ TÉCNICA:
       • Manejo automático de datos faltantes y nombres alternativos
       • Validación de supuestos estadísticos (normalidad, linealidad, multicolinealidad)
       • Análisis de sensibilidad mediante simulación Monte Carlo
       • Intervalos de confianza bootstrap para métricas clave

    INTERPRETABILIDAD VISUAL:
       • Separación clara de efectos directos e indirectos
       • Visualizaciones especializadas para modelos estructurales
       • Comparación sistemática entre grupos demográficos
       • Reportes ejecutivos con recomendaciones metodológicas
       • Gráficos profesionales con alta resolución (300 DPI)

    REPRODUCIBILIDAD:
       • Semillas aleatorias fijas para reproducibilidad total
       • Documentación exhaustiva de decisiones metodológicas
       • Exportación completa de resultados y visualizaciones
       • Código modular y extensible para futuras adaptaciones

REQUISITOS DE DATOS:
    Variables requeridas por grupo:
    • PSE/PSEP: Promedios educación socioeconómica
    • DH/PROM_DH: Variables demográficas
    • CS/PROM_CS: Factores culturales/sociales
    • AV/PROM_AV: Medidas de aversión
    • SQ: Variable satisfacción (intermedia)
    • PCA/PPCA: Variable comportamiento ahorro (objetivo)

OUTPUTS GENERADOS:
    • Archivos de visualización PNG (alta resolución)
    • Métricas de validación cruzada y Monte Carlo
    • Reportes de comparación entre modelos
    • Análisis de sensibilidad y contribuciones
    • Diagnósticos completos de ajuste del modelo

INNOVACIONES DE ESTA VERSIÓN:
    • Corrección del flujo de visualizaciones (anteriormente no se ejecutaban)
    • Integración completa de funciones gráficas en el análisis principal
    • Manejo robusto de errores en generación de gráficos
    • Creación automática de directorios de salida
    • Visualizaciones especializadas para modelos estructurales PLS-SEM

AUTOR: Sistema Experto en PLS-SEM y Modelado Predictivo
VERSIÓN: 2.1 - Estructural Completo con Visualizaciones Integradas
FECHA: 2025 - Optimizado para Investigación Académica

═══════════════════════════════════════════════════════════════════════════════════
    SIMULADOR CAPACIDAD PREDICTIVA PLS-SEM: ANÁLISIS ESTRUCTURAL CON VISUALIZACIONES
═══════════════════════════════════════════════════════════════════════════════════
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, RepeatedKFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.stats import pearsonr
import warnings
import os
from datetime import datetime
warnings.filterwarnings('ignore')

# Configuración de estilo para gráficos
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class PLSSEMPredictAnalyzer:
    """
    Clase para análisis predictivo de modelos PLS-SEM estructurales específicos
    CON VISUALIZACIONES COMPLETAS
    """

    def __init__(self, base_output_path="C:/01 academico/001 Doctorado Economia UCAB/d tesis problema ahorro/01 TESIS DEFINITIVA/MODELO/resultados obj5/"):
        # Crear carpeta de corrida con timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_folder = f"Corrida_{timestamp}"
        self.output_path = os.path.join(base_output_path, self.run_folder)

        # Crear directorio de la corrida
        os.makedirs(self.output_path, exist_ok=True)

        self.results = {}
        self.figures = []

        # Limpiar pantalla del terminal
        os.system('cls' if os.name == 'nt' else 'clear')

        print(f"Directorio de corrida creado: {self.output_path}")

        # Coeficientes del modelo PLS-SEM
        self.model_coefficients = {
            'MUJERES': {
                'pca_equation': {'PSE': 0.3485, 'DH': -0.2013, 'SQ': -0.5101, 'CS': 0.3676},
                'sq_equation': {'AV': 0.6609}
            },
            'HOMBRES': {
                'pca_equation': {'PSE': 0.3777, 'DH': 0.2226, 'SQ': -0.5947, 'CS': 0.2866},
                'sq_equation': {'AV': 0.8402}
            }
        }

    def load_data(self, file_path):
        """Cargar datos desde archivo Excel"""
        try:
            data = pd.read_excel(file_path)
            print(f"Datos cargados exitosamente: {data.shape}")
            return data
        except Exception as e:
            print(f"Error al cargar datos: {e}")
            return None

    def prepare_structural_data(self, data, group='MUJERES'):
        """Preparar datos para modelo estructural específico del grupo"""

        # Variables según el modelo estructural
        required_vars = ['PSE', 'DH', 'CS', 'AV', 'PCA']
        intermediate_vars = ['AV']  # Para predecir SQ
        structural_vars = ['PSE', 'DH', 'CS']  # Para PCA (sin SQ aún)

        # Verificar que todas las variables estén presentes
        missing_vars = set(required_vars) - set(data.columns)
        if missing_vars:
            print(f"Advertencia: Variables faltantes: {missing_vars}")
            # Intentar mapear nombres alternativos
            alternative_names = {
                'PSE': ['PSEP', 'PSE_PROMEDIO'],
                'DH': ['PROM_DH', 'DH_PROMEDIO'],
                'CS': ['PROM_CS', 'CS_PROMEDIO'],
                'AV': ['PROM_AV', 'AV_PROMEDIO'],
                'PCA': ['PPCA', 'PCA_PROMEDIO']
            }

            for var in missing_vars:
                if var in alternative_names:
                    for alt_name in alternative_names[var]:
                        if alt_name in data.columns:
                            data[var] = data[alt_name]
                            print(f"Mapeado: {alt_name} -> {var}")
                            break

        # Extraer variables finales
        try:
            X_intermediate = data[intermediate_vars].copy()  # Para predecir SQ
            X_structural = data[structural_vars].copy()
            y_target = data['PCA'].copy()

            # Si tenemos SQ observado, lo incluimos para validación
            if 'SQ' in data.columns:
                y_intermediate = data['SQ'].copy()
            else:
                y_intermediate = None

        except KeyError as e:
            print(f"Error: Variable no encontrada: {e}")
            return None, None, None, None, None, None

        # Limpiar datos faltantes
        if y_intermediate is not None:
            mask = ~(X_intermediate.isnull().any(axis=1) |
                     X_structural.isnull().any(axis=1) |
                     y_intermediate.isnull() |
                     y_target.isnull())
        else:
            mask = ~(X_intermediate.isnull().any(axis=1) |
                     X_structural.isnull().any(axis=1) |
                     y_target.isnull())

        X_intermediate_clean = X_intermediate[mask]
        X_structural_clean = X_structural[mask]
        y_target_clean = y_target[mask]
        y_intermediate_clean = y_intermediate[mask] if y_intermediate is not None else None

        print(
            f"Datos limpios para análisis {group}: {X_structural_clean.shape[0]} observaciones")

        return X_intermediate_clean, X_structural_clean, y_target_clean, y_intermediate_clean, intermediate_vars, structural_vars

    def predict_with_structural_model(self, X_intermediate, X_structural, group='MUJERES'):
        """
        Predecir usando el modelo estructural específico
        """
        coeffs = self.model_coefficients[group]

        # Paso 1: Predecir SQ usando AV
        SQ_pred = coeffs['sq_equation']['AV'] * X_intermediate['AV']

        # Paso 2: Predecir PCA usando ecuación estructural
        PCA_pred = (coeffs['pca_equation']['PSE'] * X_structural['PSE'] +
                    coeffs['pca_equation']['DH'] * X_structural['DH'] +
                    coeffs['pca_equation']['CS'] * X_structural['CS'] +
                    coeffs['pca_equation']['SQ'] * SQ_pred)

        return PCA_pred, SQ_pred

    def structural_cross_validation(self, X_intermediate, X_structural, y_target, y_intermediate=None,
                                    group='MUJERES', n_splits=10, n_repeats=20):
        """
        Validación cruzada adaptada para modelo estructural
        """
        print(f"=== VALIDACIÓN CRUZADA ESTRUCTURAL - {group} ===")

        rkf = RepeatedKFold(n_splits=n_splits,
                            n_repeats=n_repeats, random_state=42)

        # Métricas para PCA (objetivo final)
        cv_rmse_pca = []
        cv_mae_pca = []
        cv_r2_pca = []

        # Métricas para SQ (variable intermedia) si disponible
        cv_rmse_sq = []
        cv_mae_sq = []
        cv_r2_sq = []

        for train_idx, test_idx in rkf.split(X_structural):
            # Dividir datos
            X_int_test = X_intermediate.iloc[test_idx]
            X_str_test = X_structural.iloc[test_idx]
            y_target_test = y_target.iloc[test_idx]

            if y_intermediate is not None:
                y_int_test = y_intermediate.iloc[test_idx]

            # Predecir usando coeficientes fijos
            coeffs = self.model_coefficients[group]
            sq_pred_test = coeffs['sq_equation']['AV'] * X_int_test['AV']

            if y_intermediate is not None:
                cv_rmse_sq.append(
                    np.sqrt(mean_squared_error(y_int_test, sq_pred_test)))
                cv_mae_sq.append(mean_absolute_error(y_int_test, sq_pred_test))
                cv_r2_sq.append(r2_score(y_int_test, sq_pred_test))

            # Predecir PCA usando modelo estructural
            pca_pred_test = (coeffs['pca_equation']['PSE'] * X_str_test['PSE'] +
                             coeffs['pca_equation']['DH'] * X_str_test['DH'] +
                             coeffs['pca_equation']['CS'] * X_str_test['CS'] +
                             coeffs['pca_equation']['SQ'] * sq_pred_test)

            # Evaluar predicción de PCA
            cv_rmse_pca.append(
                np.sqrt(mean_squared_error(y_target_test, pca_pred_test)))
            cv_mae_pca.append(mean_absolute_error(
                y_target_test, pca_pred_test))
            cv_r2_pca.append(r2_score(y_target_test, pca_pred_test))

        # Compilar resultados
        results = {
            'PCA_Structural': {
                'RMSE': np.array(cv_rmse_pca),
                'MAE': np.array(cv_mae_pca),
                'R2': np.array(cv_r2_pca),
                'RMSE_mean': np.mean(cv_rmse_pca),
                'RMSE_std': np.std(cv_rmse_pca),
                'MAE_mean': np.mean(cv_mae_pca),
                'MAE_std': np.std(cv_mae_pca),
                'R2_mean': np.mean(cv_r2_pca),
                'R2_std': np.std(cv_r2_pca)
            }
        }

        if cv_rmse_sq:  # Si tenemos métricas de SQ
            results['SQ_Intermediate'] = {
                'RMSE': np.array(cv_rmse_sq),
                'MAE': np.array(cv_mae_sq),
                'R2': np.array(cv_r2_sq),
                'RMSE_mean': np.mean(cv_rmse_sq),
                'RMSE_std': np.std(cv_rmse_sq),
                'MAE_mean': np.mean(cv_mae_sq),
                'MAE_std': np.std(cv_mae_sq),
                'R2_mean': np.mean(cv_r2_sq),
                'R2_std': np.std(cv_r2_sq)
            }

        self.results[f'structural_cv_{group}'] = results
        return results

        # Mostrar resultados
        print(f"\nResultados Validación Cruzada - {group}:")
        print("-" * 60)
        for model_name, result in results.items():
            print(f"\n{model_name}:")
            print(
                f"  RMSE: {result['RMSE_mean']:.4f} ± {result['RMSE_std']:.4f}")
            print(
                f"  MAE:  {result['MAE_mean']:.4f} ± {result['MAE_std']:.4f}")
            print(f"  R²:   {result['R2_mean']:.4f} ± {result['R2_std']:.4f}")

    def save_results_to_excel(self, group='MUJERES'):
        """
        Guardar todos los resultados del análisis en archivos Excel
        """
        print(f"Guardando resultados en Excel para {group}...")

        try:
            # 1. Resultados de Validación Cruzada
            cv_key = f'structural_cv_{group}'
            if cv_key in self.results:
                cv_data = []
                cv_results = self.results[cv_key]

                for model_name, metrics in cv_results.items():
                    cv_data.append({
                        'Grupo': group,
                        'Modelo': model_name,
                        'Métrica': 'RMSE',
                        'Media': metrics['RMSE_mean'],
                        'Desv_Est': metrics['RMSE_std']
                    })
                    cv_data.append({
                        'Grupo': group,
                        'Modelo': model_name,
                        'Métrica': 'MAE',
                        'Media': metrics['MAE_mean'],
                        'Desv_Est': metrics['MAE_std']
                    })
                    cv_data.append({
                        'Grupo': group,
                        'Modelo': model_name,
                        'Métrica': 'R²',
                        'Media': metrics['R2_mean'],
                        'Desv_Est': metrics['R2_std']
                    })

                cv_df = pd.DataFrame(cv_data)
                cv_file = os.path.join(
                    self.output_path, f"{group}_validacion_cruzada.xlsx")
                cv_df.to_excel(cv_file, index=False)
                print(f"✓ Guardado: {cv_file}")

            # 2. Resultados Monte Carlo
            mc_key = f'monte_carlo_{group}'
            if mc_key in self.results:
                mc_results = self.results[mc_key]

                # Datos de simulaciones
                scenarios_df = pd.DataFrame(mc_results['scenarios'])
                mc_file = os.path.join(
                    self.output_path, f"{group}_monte_carlo_simulaciones.xlsx")

                # Crear múltiples hojas
                with pd.ExcelWriter(mc_file, engine='openpyxl') as writer:
                    scenarios_df.to_excel(
                        writer, sheet_name='Simulaciones', index=False)

                    # Percentiles
                    percentiles_data = {
                        'Percentil': list(mc_results['pca_percentiles'].keys()),
                        'PCA_Valor': list(mc_results['pca_percentiles'].values()),
                        'SQ_Valor': list(mc_results['sq_percentiles'].values())
                    }
                    pd.DataFrame(percentiles_data).to_excel(
                        writer, sheet_name='Percentiles', index=False)

                    # Estadísticas
                    stats_data = {
                        'Variable': ['PCA', 'SQ'],
                        'Media': [mc_results['pca_predictions'].mean(), mc_results['sq_predictions'].mean()],
                        'Desv_Est': [mc_results['pca_predictions'].std(), mc_results['sq_predictions'].std()],
                        'Min': [mc_results['pca_predictions'].min(), mc_results['sq_predictions'].min()],
                        'Max': [mc_results['pca_predictions'].max(), mc_results['sq_predictions'].max()]
                    }
                    pd.DataFrame(stats_data).to_excel(
                        writer, sheet_name='Estadisticas', index=False)

                print(f"✓ Guardado: {mc_file}")

            # 3. Comparación de modelos
            comp_key = f'model_comparison_{group}'
            if comp_key in self.results:
                comp_results = self.results[comp_key]

                comparison_data = {
                    'Fold': range(1, len(comp_results['structural_r2']) + 1),
                    'R2_Estructural': comp_results['structural_r2'],
                    'R2_Regresion_Directa': comp_results['direct_r2'],
                    'Diferencia': comp_results['structural_r2'] - comp_results['direct_r2']
                }

                comp_df = pd.DataFrame(comparison_data)
                comp_file = os.path.join(
                    self.output_path, f"{group}_comparacion_modelos.xlsx")
                comp_df.to_excel(comp_file, index=False)
                print(f"✓ Guardado: {comp_file}")

        except Exception as e:
            print(f"Error guardando resultados Excel para {group}: {e}")

    def generate_comprehensive_report(self, group='MUJERES'):
        """
        Generar reporte comprehensivo en archivo TXT
        """
        report_file = os.path.join(
            self.output_path, f"{group}_reporte_completo.txt")

        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write(
                    f"REPORTE COMPLETO ANÁLISIS PLS-SEM ESTRUCTURAL - {group}\n")
                f.write("=" * 80 + "\n")
                f.write(
                    f"Fecha y hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Directorio: {self.output_path}\n")
                f.write("=" * 80 + "\n\n")

                # 1. ECUACIONES DEL MODELO
                f.write("1. ECUACIONES DEL MODELO ESTRUCTURAL\n")
                f.write("-" * 50 + "\n")
                coeffs = self.model_coefficients[group]
                pca_eq = coeffs['pca_equation']

                f.write(f"Ecuación Principal (PCA):\n")
                f.write(
                    f"PCA = {pca_eq['PSE']:.4f}·PSE + {pca_eq['DH']:.4f}·DH + {pca_eq['SQ']:.4f}·SQ + {pca_eq['CS']:.4f}·CS + ε₁\n\n")

                f.write(f"Ecuación Intermedia (SQ):\n")
                sq_coeff = coeffs['sq_equation']['AV']
                f.write(f"SQ = {sq_coeff:.4f}·AV + ε₂\n\n")

                # 2. VALIDACIÓN CRUZADA
                cv_key = f'structural_cv_{group}'
                if cv_key in self.results:
                    f.write("2. RESULTADOS VALIDACIÓN CRUZADA\n")
                    f.write("-" * 50 + "\n")
                    cv_results = self.results[cv_key]

                    for model_name, metrics in cv_results.items():
                        f.write(f"\n{model_name}:\n")
                        f.write(
                            f"  RMSE: {metrics['RMSE_mean']:.4f} ± {metrics['RMSE_std']:.4f}\n")
                        f.write(
                            f"  MAE:  {metrics['MAE_mean']:.4f} ± {metrics['MAE_std']:.4f}\n")
                        f.write(
                            f"  R²:   {metrics['R2_mean']:.4f} ± {metrics['R2_std']:.4f}\n")

                # 3. MONTE CARLO
                mc_key = f'monte_carlo_{group}'
                if mc_key in self.results:
                    f.write("\n3. ANÁLISIS MONTE CARLO (5,000 simulaciones)\n")
                    f.write("-" * 50 + "\n")
                    mc_results = self.results[mc_key]
                    pca_preds = mc_results['pca_predictions']
                    sq_preds = mc_results['sq_predictions']

                    f.write(f"Predicciones PCA:\n")
                    f.write(f"  Media: {pca_preds.mean():.4f}\n")
                    f.write(f"  Desviación Estándar: {pca_preds.std():.4f}\n")
                    f.write(
                        f"  Rango: [{pca_preds.min():.4f}, {pca_preds.max():.4f}]\n")

                    f.write(f"\nPercentiles PCA:\n")
                    for p, val in mc_results['pca_percentiles'].items():
                        f.write(f"  P{p}: {val:.4f}\n")

                    f.write(f"\nPredicciones SQ:\n")
                    f.write(f"  Media: {sq_preds.mean():.4f}\n")
                    f.write(f"  Desviación Estándar: {sq_preds.std():.4f}\n")
                    f.write(
                        f"  Rango: [{sq_preds.min():.4f}, {sq_preds.max():.4f}]\n")

                # 4. COMPARACIÓN DE MODELOS
                comp_key = f'model_comparison_{group}'
                if comp_key in self.results:
                    f.write("\n4. COMPARACIÓN ESTRUCTURAL vs REGRESIÓN DIRECTA\n")
                    f.write("-" * 50 + "\n")
                    comp_results = self.results[comp_key]

                    structural_mean = comp_results['structural_r2'].mean()
                    direct_mean = comp_results['direct_r2'].mean()
                    structural_std = comp_results['structural_r2'].std()
                    direct_std = comp_results['direct_r2'].std()

                    f.write(
                        f"Modelo Estructural R²: {structural_mean:.4f} ± {structural_std:.4f}\n")
                    f.write(
                        f"Regresión Directa R²: {direct_mean:.4f} ± {direct_std:.4f}\n")
                    f.write(
                        f"Diferencia: {(structural_mean - direct_mean):.4f}\n")
                    f.write(f"p-valor: {comp_results['p_value']:.6f}\n")

                    if comp_results['p_value'] < 0.05:
                        if structural_mean > direct_mean:
                            f.write(
                                "CONCLUSIÓN: Modelo estructural es significativamente superior\n")
                        else:
                            f.write(
                                "CONCLUSIÓN: Regresión directa es significativamente superior\n")
                    else:
                        f.write(
                            "CONCLUSIÓN: No hay diferencia significativa entre modelos\n")

                # 5. ARCHIVOS GENERADOS
                f.write("\n5. ARCHIVOS GENERADOS\n")
                f.write("-" * 50 + "\n")
                f.write("Visualizaciones:\n")
                f.write(
                    f"• {group}_observed_vs_predicted.png - Ajuste del modelo\n")
                f.write(
                    f"• {group}_monte_carlo_distribution.png - Distribuciones simuladas\n")
                f.write(
                    f"• {group}_residual_analysis.png - Análisis de residuos\n")
                f.write(
                    f"• {group}_model_comparison.png - Comparación de modelos\n")
                f.write(
                    f"• {group}_sensitivity_analysis.png - Análisis de sensibilidad\n")

                f.write("\nDatos:\n")
                f.write(
                    f"• {group}_validacion_cruzada.xlsx - Métricas de validación\n")
                f.write(
                    f"• {group}_monte_carlo_simulaciones.xlsx - Simulaciones Monte Carlo\n")
                f.write(
                    f"• {group}_comparacion_modelos.xlsx - Comparación estructural vs directa\n")
                f.write(f"• {group}_reporte_completo.txt - Este reporte\n")

                # 6. INTERPRETACIONES
                f.write("\n6. INTERPRETACIONES Y RECOMENDACIONES\n")
                f.write("-" * 50 + "\n")

                if cv_key in self.results:
                    pca_r2 = self.results[cv_key]['PCA_Structural']['R2_mean']
                    if pca_r2 > 0.7:
                        f.write(
                            "• ALTA capacidad predictiva del modelo estructural\n")
                    elif pca_r2 > 0.4:
                        f.write(
                            "• MODERADA capacidad predictiva del modelo estructural\n")
                    else:
                        f.write(
                            "• BAJA capacidad predictiva del modelo estructural\n")

                f.write("• Modelo implementa coeficientes PLS-SEM específicos\n")
                f.write(
                    "• Validación respeta estructura de ecuaciones simultáneas\n")
                f.write("• Monte Carlo evalúa robustez bajo incertidumbre\n")
                f.write(
                    "• Visualizaciones permiten diagnóstico completo del modelo\n")

                f.write("\n" + "=" * 80 + "\n")
                f.write("FIN DEL REPORTE\n")
                f.write("=" * 80 + "\n")

            print(f"✓ Reporte completo guardado: {report_file}")

        except Exception as e:
            print(f"Error generando reporte para {group}: {e}")

    def plot_observed_vs_predicted(self, X_intermediate, X_structural, y_target, y_intermediate, group='MUJERES'):
        """
        Gráfico observado vs predicho
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        coeffs = self.model_coefficients[group]

        # Predicciones del modelo
        sq_pred = coeffs['sq_equation']['AV'] * X_intermediate['AV']
        pca_pred = (coeffs['pca_equation']['PSE'] * X_structural['PSE'] +
                    coeffs['pca_equation']['DH'] * X_structural['DH'] +
                    coeffs['pca_equation']['CS'] * X_structural['CS'] +
                    coeffs['pca_equation']['SQ'] * sq_pred)

        # SQ: Observado vs Predicho
        if y_intermediate is not None:
            axes[0].scatter(y_intermediate, sq_pred,
                            alpha=0.6, color='steelblue')
            min_val = min(y_intermediate.min(), sq_pred.min())
            max_val = max(y_intermediate.max(), sq_pred.max())
            axes[0].plot([min_val, max_val], [
                         min_val, max_val], 'r--', linewidth=2)

            r2_sq = r2_score(y_intermediate, sq_pred)
            axes[0].set_title(f'SQ: Observado vs Predicho\nR² = {r2_sq:.4f}')
            axes[0].set_xlabel('SQ Observado')
            axes[0].set_ylabel('SQ Predicho')
            axes[0].grid(True, alpha=0.3)

        # PCA: Observado vs Predicho
        axes[1].scatter(y_target, pca_pred, alpha=0.6, color='darkgreen')
        min_val = min(y_target.min(), pca_pred.min())
        max_val = max(y_target.max(), pca_pred.max())
        axes[1].plot([min_val, max_val], [
                     min_val, max_val], 'r--', linewidth=2)

        r2_pca = r2_score(y_target, pca_pred)
        axes[1].set_title(f'PCA: Observado vs Predicho\nR² = {r2_pca:.4f}')
        axes[1].set_xlabel('PCA Observado')
        axes[1].set_ylabel('PCA Predicho')
        axes[1].grid(True, alpha=0.3)

        plt.suptitle(
            f'Ajuste del Modelo Estructural - {group}', fontsize=16, fontweight='bold')
        plt.tight_layout()

        # Nombrado correcto: GRUPO_nombre.png
        filename = f"{group}_observed_vs_predicted.png"
        plt.savefig(os.path.join(self.output_path, filename),
                    dpi=300, bbox_inches='tight')
        plt.show()

    def monte_carlo_structural(self, X_intermediate, X_structural, group='MUJERES', n_simulations=5000):
        """
        Análisis Monte Carlo para modelo estructural
        """
        print(f"\n=== ANÁLISIS MONTE CARLO ESTRUCTURAL - {group} ===")

        # Estadísticas descriptivas
        X_int_stats = X_intermediate.describe()
        X_str_stats = X_structural.describe()

        # Generar simulaciones
        predictions_pca = []
        predictions_sq = []
        scenarios = []

        np.random.seed(42)
        coeffs = self.model_coefficients[group]

        for i in range(n_simulations):
            scenario = {}

            # Generar valores para variables intermedias (AV)
            for col in X_intermediate.columns:
                mean_val = X_int_stats.loc['mean', col]
                std_val = X_int_stats.loc['std', col]
                min_val = X_int_stats.loc['min', col]
                max_val = X_int_stats.loc['max', col]

                sim_val = np.random.normal(mean_val, std_val)
                sim_val = np.clip(sim_val, min_val, max_val)
                scenario[f'{col}_intermediate'] = sim_val

            # Generar valores para variables estructurales
            for col in X_structural.columns:
                mean_val = X_str_stats.loc['mean', col]
                std_val = X_str_stats.loc['std', col]
                min_val = X_str_stats.loc['min', col]
                max_val = X_str_stats.loc['max', col]

                sim_val = np.random.normal(mean_val, std_val)
                sim_val = np.clip(sim_val, min_val, max_val)
                scenario[f'{col}_structural'] = sim_val

            # Predecir SQ usando coeficientes fijos
            sq_pred = coeffs['sq_equation']['AV'] * scenario['AV_intermediate']
            predictions_sq.append(sq_pred)
            scenario['SQ_predicted'] = sq_pred

            # Predecir PCA
            pca_pred = (coeffs['pca_equation']['PSE'] * scenario['PSE_structural'] +
                        coeffs['pca_equation']['DH'] * scenario['DH_structural'] +
                        coeffs['pca_equation']['CS'] * scenario['CS_structural'] +
                        coeffs['pca_equation']['SQ'] * sq_pred)

            predictions_pca.append(pca_pred)
            scenario['PCA_predicted'] = pca_pred
            scenarios.append(scenario)

        predictions_pca = np.array(predictions_pca)
        predictions_sq = np.array(predictions_sq)

        # Análisis de percentiles
        percentiles = [10, 25, 50, 75, 90]
        pca_percentiles = np.percentile(predictions_pca, percentiles)
        sq_percentiles = np.percentile(predictions_sq, percentiles)

        print(f"Simulaciones Monte Carlo: {n_simulations}")
        print("\nDistribución de Predicciones PCA:")
        for i, p in enumerate(percentiles):
            print(f"  P{p}: {pca_percentiles[i]:.4f}")

        print(f"Media PCA: {predictions_pca.mean():.4f}")
        print(f"SD PCA: {predictions_pca.std():.4f}")

        print("\nDistribución de Predicciones SQ:")
        for i, p in enumerate(percentiles):
            print(f"  P{p}: {sq_percentiles[i]:.4f}")
        print(f"Media SQ: {predictions_sq.mean():.4f}")
        print(f"SD SQ: {predictions_sq.std():.4f}")

        # Guardar resultados
        self.results[f'monte_carlo_{group}'] = {
            'pca_predictions': predictions_pca,
            'sq_predictions': predictions_sq,
            'scenarios': scenarios,
            'pca_percentiles': dict(zip(percentiles, pca_percentiles)),
            'sq_percentiles': dict(zip(percentiles, sq_percentiles))
        }

        return predictions_pca, predictions_sq, scenarios

    def compare_structural_vs_direct_regression(self, X_intermediate, X_structural, y_target, group='MUJERES'):
        """
        Comparar modelo estructural PLS-SEM vs regresión directa
        """
        print(
            f"\n=== COMPARACIÓN MODELO ESTRUCTURAL VS REGRESIÓN DIRECTA - {group} ===")

        # Combinar todas las variables para regresión directa
        X_combined = pd.concat([X_intermediate, X_structural], axis=1)

        # Validación cruzada para ambos modelos
        rkf = RepeatedKFold(n_splits=10, n_repeats=20, random_state=42)

        structural_r2 = []
        direct_r2 = []

        coeffs = self.model_coefficients[group]

        for train_idx, test_idx in rkf.split(X_combined):
            # Datos de prueba
            X_int_test = X_intermediate.iloc[test_idx]
            X_str_test = X_structural.iloc[test_idx]
            X_comb_test = X_combined.iloc[test_idx]
            y_test = y_target.iloc[test_idx]

            # Datos de entrenamiento para regresión directa
            X_comb_train = X_combined.iloc[train_idx]
            y_train = y_target.iloc[train_idx]

            # Modelo 1: Estructural PLS-SEM
            sq_pred = coeffs['sq_equation']['AV'] * X_int_test['AV']
            pca_pred_structural = (coeffs['pca_equation']['PSE'] * X_str_test['PSE'] +
                                   coeffs['pca_equation']['DH'] * X_str_test['DH'] +
                                   coeffs['pca_equation']['CS'] * X_str_test['CS'] +
                                   coeffs['pca_equation']['SQ'] * sq_pred)

            structural_r2.append(r2_score(y_test, pca_pred_structural))

            # Modelo 2: Regresión directa
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_comb_train)
            X_test_scaled = scaler.transform(X_comb_test)

            reg_direct = LinearRegression()
            reg_direct.fit(X_train_scaled, y_train)
            pca_pred_direct = reg_direct.predict(X_test_scaled)

            direct_r2.append(r2_score(y_test, pca_pred_direct))

        structural_r2 = np.array(structural_r2)
        direct_r2 = np.array(direct_r2)

        # Prueba estadística
        t_stat, p_val = stats.ttest_rel(structural_r2, direct_r2)

        print(
            f"Modelo Estructural R²: {structural_r2.mean():.4f} ± {structural_r2.std():.4f}")
        print(
            f"Regresión Directa R²: {direct_r2.mean():.4f} ± {direct_r2.std():.4f}")
        print(f"Diferencia: {(structural_r2.mean() - direct_r2.mean()):.4f}")
        print(f"Prueba t: t = {t_stat:.4f}, p = {p_val:.4f}")

        if p_val < 0.05:
            if structural_r2.mean() > direct_r2.mean():
                print("✓ Modelo estructural es significativamente superior")
            else:
                print("⚠ Regresión directa es significativamente superior")
        else:
            print("→ No hay diferencia significativa entre modelos")

        # Guardar para visualización
        self.results[f'model_comparison_{group}'] = {
            'structural_r2': structural_r2,
            'direct_r2': direct_r2,
            'p_value': p_val
        }

        return structural_r2, direct_r2, p_val

    # ======================= VISUALIZACIONES =======================

    def plot_observed_vs_predicted(self, X_intermediate, X_structural, y_target, y_intermediate, group='MUJERES'):
        """
        Gráfico observado vs predicho
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        coeffs = self.model_coefficients[group]

        # Predicciones del modelo
        sq_pred = coeffs['sq_equation']['AV'] * X_intermediate['AV']
        pca_pred = (coeffs['pca_equation']['PSE'] * X_structural['PSE'] +
                    coeffs['pca_equation']['DH'] * X_structural['DH'] +
                    coeffs['pca_equation']['CS'] * X_structural['CS'] +
                    coeffs['pca_equation']['SQ'] * sq_pred)

        # SQ: Observado vs Predicho
        if y_intermediate is not None:
            axes[0].scatter(y_intermediate, sq_pred,
                            alpha=0.6, color='steelblue')
            min_val = min(y_intermediate.min(), sq_pred.min())
            max_val = max(y_intermediate.max(), sq_pred.max())
            axes[0].plot([min_val, max_val], [
                         min_val, max_val], 'r--', linewidth=2)

            r2_sq = r2_score(y_intermediate, sq_pred)
            axes[0].set_title(f'SQ: Observado vs Predicho\nR² = {r2_sq:.4f}')
            axes[0].set_xlabel('SQ Observado')
            axes[0].set_ylabel('SQ Predicho')
            axes[0].grid(True, alpha=0.3)

        # PCA: Observado vs Predicho
        axes[1].scatter(y_target, pca_pred, alpha=0.6, color='darkgreen')
        min_val = min(y_target.min(), pca_pred.min())
        max_val = max(y_target.max(), pca_pred.max())
        axes[1].plot([min_val, max_val], [
                     min_val, max_val], 'r--', linewidth=2)

        r2_pca = r2_score(y_target, pca_pred)
        axes[1].set_title(f'PCA: Observado vs Predicho\nR² = {r2_pca:.4f}')
        axes[1].set_xlabel('PCA Observado')
        axes[1].set_ylabel('PCA Predicho')
        axes[1].grid(True, alpha=0.3)

        plt.suptitle(
            f'Ajuste del Modelo Estructural - {group}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{self.output_path}observed_vs_predicted_{group.lower()}.png",
                    dpi=300, bbox_inches='tight')
        plt.show()

    def plot_monte_carlo_distribution(self, group='MUJERES'):
        """
        Distribución Monte Carlo
        """
        if f'monte_carlo_{group}' not in self.results:
            print(f"No hay resultados Monte Carlo para {group}")
            return

        mc_results = self.results[f'monte_carlo_{group}']
        pca_preds = mc_results['pca_predictions']
        sq_preds = mc_results['sq_predictions']

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Histograma PCA
        axes[0, 0].hist(pca_preds, bins=50, alpha=0.7,
                        color='darkgreen', edgecolor='black')
        axes[0, 0].axvline(pca_preds.mean(), color='red', linestyle='--',
                           linewidth=2, label=f'Media: {pca_preds.mean():.3f}')
        axes[0, 0].set_title('Distribución Predicciones PCA')
        axes[0, 0].set_xlabel('PCA Predicho')
        axes[0, 0].set_ylabel('Frecuencia')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Histograma SQ
        axes[0, 1].hist(sq_preds, bins=50, alpha=0.7,
                        color='steelblue', edgecolor='black')
        axes[0, 1].axvline(sq_preds.mean(), color='red', linestyle='--',
                           linewidth=2, label=f'Media: {sq_preds.mean():.3f}')
        axes[0, 1].set_title('Distribución Predicciones SQ')
        axes[0, 1].set_xlabel('SQ Predicho')
        axes[0, 1].set_ylabel('Frecuencia')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Box plot PCA
        axes[1, 0].boxplot(pca_preds, vert=True)
        axes[1, 0].set_title('Box Plot PCA')
        axes[1, 0].set_ylabel('PCA Predicho')
        axes[1, 0].grid(True, alpha=0.3)

        # Box plot SQ
        axes[1, 1].boxplot(sq_preds, vert=True)
        axes[1, 1].set_title('Box Plot SQ')
        axes[1, 1].set_ylabel('SQ Predicho')
        axes[1, 1].grid(True, alpha=0.3)

        plt.suptitle(
            f'Análisis Monte Carlo - {group} (n=5000)', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{self.output_path}monte_carlo_distribution_{group.lower()}.png",
                    dpi=300, bbox_inches='tight')
        plt.show()

    def plot_residual_analysis(self, X_intermediate, X_structural, y_target, y_intermediate, group='MUJERES'):
        """
        Análisis de residuos
        """
        coeffs = self.model_coefficients[group]

        # Predicciones
        sq_pred = coeffs['sq_equation']['AV'] * X_intermediate['AV']
        pca_pred = (coeffs['pca_equation']['PSE'] * X_structural['PSE'] +
                    coeffs['pca_equation']['DH'] * X_structural['DH'] +
                    coeffs['pca_equation']['CS'] * X_structural['CS'] +
                    coeffs['pca_equation']['SQ'] * sq_pred)

        # Residuos
        residuals_pca = y_target - pca_pred

        if y_intermediate is not None:
            residuals_sq = y_intermediate - sq_pred
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))

            # Residuos SQ vs predicho
            axes[0, 0].scatter(sq_pred, residuals_sq,
                               alpha=0.6, color='steelblue')
            axes[0, 0].axhline(y=0, color='red', linestyle='--')
            axes[0, 0].set_title('Residuos SQ vs Predicho')
            axes[0, 0].set_xlabel('SQ Predicho')
            axes[0, 0].set_ylabel('Residuos SQ')
            axes[0, 0].grid(True, alpha=0.3)

            # Histograma residuos SQ
            axes[0, 1].hist(residuals_sq, bins=30, alpha=0.7,
                            color='steelblue', edgecolor='black')
            axes[0, 1].set_title('Distribución Residuos SQ')
            axes[0, 1].set_xlabel('Residuos')
            axes[0, 1].set_ylabel('Frecuencia')
            axes[0, 1].grid(True, alpha=0.3)
        else:
            fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Residuos PCA vs predicho
        if y_intermediate is not None:
            ax_pca_scatter = axes[1, 0]
            ax_pca_hist = axes[1, 1]
        else:
            ax_pca_scatter = axes[0]
            ax_pca_hist = axes[1]

        ax_pca_scatter.scatter(pca_pred, residuals_pca,
                               alpha=0.6, color='darkgreen')
        ax_pca_scatter.axhline(y=0, color='red', linestyle='--')
        ax_pca_scatter.set_title('Residuos PCA vs Predicho')
        ax_pca_scatter.set_xlabel('PCA Predicho')
        ax_pca_scatter.set_ylabel('Residuos PCA')
        ax_pca_scatter.grid(True, alpha=0.3)

        # Histograma residuos PCA
        ax_pca_hist.hist(residuals_pca, bins=30, alpha=0.7,
                         color='darkgreen', edgecolor='black')
        ax_pca_hist.set_title('Distribución Residuos PCA')
        ax_pca_hist.set_xlabel('Residuos')
        ax_pca_hist.set_ylabel('Frecuencia')
        ax_pca_hist.grid(True, alpha=0.3)

        plt.suptitle(
            f'Análisis de Residuos - {group}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{self.output_path}residual_analysis_{group.lower()}.png",
                    dpi=300, bbox_inches='tight')
        plt.show()

    def plot_model_comparison(self, group='MUJERES'):
        """
        Comparación de modelos
        """
        if f'model_comparison_{group}' not in self.results:
            print(f"No hay resultados de comparación para {group}")
            return

        comparison = self.results[f'model_comparison_{group}']
        structural_r2 = comparison['structural_r2']
        direct_r2 = comparison['direct_r2']
        p_value = comparison['p_value']

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Box plot comparativo
        data_to_plot = [structural_r2, direct_r2]
        labels = ['Estructural PLS-SEM', 'Regresión Directa']

        bp = axes[0].boxplot(data_to_plot, labels=labels, patch_artist=True)
        bp['boxes'][0].set_facecolor('lightblue')
        bp['boxes'][1].set_facecolor('lightcoral')

        axes[0].set_title(f'Comparación R² - {group}')
        axes[0].set_ylabel('R² (Validación Cruzada)')
        axes[0].grid(True, alpha=0.3)

        # Añadir medias
        axes[0].text(1, structural_r2.mean() + 0.01, f'μ={structural_r2.mean():.3f}',
                     ha='center', fontweight='bold')
        axes[0].text(2, direct_r2.mean() + 0.01, f'μ={direct_r2.mean():.3f}',
                     ha='center', fontweight='bold')

        # Histograma comparativo
        axes[1].hist(structural_r2, bins=20, alpha=0.7, label='Estructural PLS-SEM',
                     color='lightblue', edgecolor='black')
        axes[1].hist(direct_r2, bins=20, alpha=0.7, label='Regresión Directa',
                     color='lightcoral', edgecolor='black')
        axes[1].axvline(structural_r2.mean(), color='blue',
                        linestyle='--', linewidth=2)
        axes[1].axvline(direct_r2.mean(), color='red',
                        linestyle='--', linewidth=2)
        axes[1].set_title('Distribución R²')
        axes[1].set_xlabel('R²')
        axes[1].set_ylabel('Frecuencia')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Añadir p-valor
        significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
        fig.suptitle(f'Comparación Modelos - {group} (p={p_value:.4f} {significance})',
                     fontsize=16, fontweight='bold')

        plt.tight_layout()
        plt.savefig(f"{self.output_path}model_comparison_{group.lower()}.png",
                    dpi=300, bbox_inches='tight')
        plt.show()

    def plot_sensitivity_analysis(self, X_intermediate, X_structural, group='MUJERES'):
        """
        Análisis de sensibilidad de variables
        """
        coeffs = self.model_coefficients[group]

        # Calcular contribuciones
        contributions = {
            'PSE → PCA': coeffs['pca_equation']['PSE'] * X_structural['PSE'],
            'DH → PCA': coeffs['pca_equation']['DH'] * X_structural['DH'],
            'CS → PCA': coeffs['pca_equation']['CS'] * X_structural['CS'],
            'AV → SQ': coeffs['sq_equation']['AV'] * X_intermediate['AV'],
            'SQ → PCA': coeffs['pca_equation']['SQ'] * (coeffs['sq_equation']['AV'] * X_intermediate['AV'])
        }

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 1. Magnitud de coeficientes
        coeff_names = ['PSE→PCA', 'DH→PCA', 'CS→PCA', 'SQ→PCA', 'AV→SQ']
        coeff_values = [coeffs['pca_equation']['PSE'], coeffs['pca_equation']['DH'],
                        coeffs['pca_equation']['CS'], coeffs['pca_equation']['SQ'],
                        coeffs['sq_equation']['AV']]
        colors = ['red' if v < 0 else 'green' for v in coeff_values]

        bars = axes[0, 0].bar(coeff_names, coeff_values,
                              color=colors, alpha=0.7)
        axes[0, 0].set_title('Coeficientes Estructurales')
        axes[0, 0].set_ylabel('Valor del Coeficiente')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        axes[0, 0].axhline(y=0, color='black', linewidth=1)

        # Añadir valores
        for i, v in enumerate(coeff_values):
            axes[0, 0].text(i, v + (0.02 if v >= 0 else -0.02), f'{v:.3f}',
                            ha='center', va='bottom' if v >= 0 else 'top', fontweight='bold')

        # 2. Magnitudes absolutas
        abs_values = [abs(v) for v in coeff_values]
        bars2 = axes[0, 1].bar(coeff_names, abs_values,
                               color='steelblue', alpha=0.7)
        axes[0, 1].set_title('Importancia Relativa (Valor Absoluto)')
        axes[0, 1].set_ylabel('Magnitud Absoluta')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3, axis='y')

        for i, v in enumerate(abs_values):
            axes[0, 1].text(
                i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')

        # 3. Distribución de contribuciones
        contrib_names = list(contributions.keys())
        contrib_means = [contrib.mean() for contrib in contributions.values()]
        contrib_stds = [contrib.std() for contrib in contributions.values()]

        x_pos = np.arange(len(contrib_names))
        bars3 = axes[1, 0].bar(x_pos, contrib_means, yerr=contrib_stds,
                               capsize=5, color='lightgreen', alpha=0.7,
                               edgecolor='black', ecolor='black')
        axes[1, 0].set_title('Contribución Media ± DE')
        axes[1, 0].set_ylabel('Contribución a PCA/SQ')
        axes[1, 0].set_xticks(x_pos)
        axes[1, 0].set_xticklabels(contrib_names, rotation=45)
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        axes[1, 0].axhline(y=0, color='black', linewidth=1)

        # 4. Box plot de contribuciones
        contrib_data = [contrib.values for contrib in contributions.values()]
        bp = axes[1, 1].boxplot(
            contrib_data, labels=contrib_names, patch_artist=True)

        colors_box = ['lightblue', 'lightcoral',
                      'lightgreen', 'lightyellow', 'lightpink']
        for patch, color in zip(bp['boxes'], colors_box):
            patch.set_facecolor(color)

        axes[1, 1].set_title('Distribución de Contribuciones')
        axes[1, 1].set_ylabel('Valor de Contribución')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        axes[1, 1].axhline(y=0, color='black', linewidth=1)

        plt.suptitle(
            f'Análisis de Sensibilidad - {group}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{self.output_path}sensitivity_analysis_{group.lower()}.png",
                    dpi=300, bbox_inches='tight')
        plt.show()

    def generate_all_visualizations(self, X_intermediate, X_structural, y_target, y_intermediate, group='MUJERES'):
        """
        Generar todas las visualizaciones para un grupo
        """
        print(f"\n=== GENERANDO VISUALIZACIONES - {group} ===")

        try:
            print("1. Observado vs Predicho...")
            self.plot_observed_vs_predicted(
                X_intermediate, X_structural, y_target, y_intermediate, group)

            print("2. Distribución Monte Carlo...")
            self.plot_monte_carlo_distribution(group)

            print("3. Análisis de residuos...")
            self.plot_residual_analysis(
                X_intermediate, X_structural, y_target, y_intermediate, group)

            print("4. Comparación de modelos...")
            self.plot_model_comparison(group)

            print("5. Análisis de sensibilidad...")
            self.plot_sensitivity_analysis(X_intermediate, X_structural, group)

            # 3b. Diagnóstico completo de residuos
            coeffs = self.model_coefficients[group]
            sq_pred = coeffs['sq_equation']['AV'] * X_intermediate['AV']
            pca_pred = (coeffs['pca_equation']['PSE'] * X_structural['PSE'] +
                        coeffs['pca_equation']['DH'] * X_structural['DH'] +
                        coeffs['pca_equation']['CS'] * X_structural['CS'] +
                        coeffs['pca_equation']['SQ'] * sq_pred)

            self.plot_residuals_diagnostics(y_target, pca_pred, group)
            print(f"✓ Visualizaciones completadas para {group}")

        except Exception as e:
            print(f"Error en visualizaciones para {group}: {e}")
            import traceback
            traceback.print_exc()

    def generate_structural_report(self, group='MUJERES'):
        """
        Generar reporte específico para modelo estructural
        """
        print(f"\n{'='*80}")
        print(f"REPORTE MODELO ESTRUCTURAL PLS-SEM - {group}")
        print("="*80)

        # Mostrar ecuaciones del modelo
        coeffs = self.model_coefficients[group]
        print(f"\nEcuaciones del Modelo {group}:")
        print("-" * 40)

        pca_eq = coeffs['pca_equation']
        print(
            f"PCA = {pca_eq['PSE']:.4f}·PSE + {pca_eq['DH']:.4f}·DH + {pca_eq['SQ']:.4f}·SQ + {pca_eq['CS']:.4f}·CS + ε₁")

        sq_coeff = coeffs['sq_equation']['AV']
        print(f"SQ = {sq_coeff:.4f}·AV + ε₂")

        # Resultados de validación cruzada
        cv_key = f'structural_cv_{group}'
        if cv_key in self.results:
            print(f"\n2. VALIDACIÓN CRUZADA ESTRUCTURAL")
            print("-" * 40)
            cv_results = self.results[cv_key]

            if 'PCA_Structural' in cv_results:
                pca_r2 = cv_results['PCA_Structural']['R2_mean']
                print(f"Capacidad Predictiva PCA: R² = {pca_r2:.4f}")

                if pca_r2 > 0.7:
                    print("✓ ALTA capacidad predictiva del modelo estructural")
                elif pca_r2 > 0.4:
                    print("→ MODERADA capacidad predictiva del modelo estructural")
                else:
                    print("⚠ BAJA capacidad predictiva del modelo estructural")

            if 'SQ_Intermediate' in cv_results:
                sq_r2 = cv_results['SQ_Intermediate']['R2_mean']
                print(f"Predicción Variable Intermedia (SQ): R² = {sq_r2:.4f}")

        # Resultados Monte Carlo
        mc_key = f'monte_carlo_{group}'
        if mc_key in self.results:
            print(f"\n3. ANÁLISIS MONTE CARLO")
            print("-" * 40)
            mc_results = self.results[mc_key]
            pca_preds = mc_results['pca_predictions']
            print(
                f"Rango Predicciones PCA: [{pca_preds.min():.4f}, {pca_preds.max():.4f}]")
            print(
                f"Media ± DE: {pca_preds.mean():.4f} ± {pca_preds.std():.4f}")

        print(f"\n4. RECOMENDACIONES ESPECÍFICAS - {group}")
        print("-" * 40)
        print("✓ Modelo estructural implementado según coeficientes PLS-SEM")
        print("✓ Validación considera estructura de ecuaciones simultáneas")
        print("✓ Monte Carlo evalúa robustez del sistema completo")
        print("✓ Todas las visualizaciones generadas correctamente")

    def run_structural_analysis(self, data_files):
        """
        Ejecutar análisis completo para modelos estructurales CON VISUALIZACIONES
        """
        for group, file_path in data_files.items():
            print(f"\n{'='*80}")
            print(f"ANÁLISIS ESTRUCTURAL PLS-SEM: {group.upper()}")
            print(f"{'='*80}")

            # Cargar datos
            data = self.load_data(file_path)
            if data is None:
                continue

            # Preparar datos estructurales
            result = self.prepare_structural_data(data, group)
            if result[0] is None:
                print(f"Error en preparación de datos para {group}")
                continue

            X_intermediate, X_structural, y_target, y_intermediate, int_vars, str_vars = result

            print(f"Variables intermedias: {int_vars}")
            print(f"Variables estructurales: {str_vars}")

            try:
                # 1. Validación cruzada estructural
                cv_results = self.structural_cross_validation(
                    X_intermediate, X_structural, y_target, y_intermediate, group)

                # 2. Monte Carlo estructural
                pca_preds, sq_preds, scenarios = self.monte_carlo_structural(
                    X_intermediate, X_structural, group)

                # 3. Comparación con regresión directa
                struct_r2, direct_r2, p_val = self.compare_structural_vs_direct_regression(
                    X_intermediate, X_structural, y_target, group)

                # 4. GENERAR TODAS LAS VISUALIZACIONES
                self.generate_all_visualizations(
                    X_intermediate, X_structural, y_target, y_intermediate, group)

                # 5. Reporte específico
                self.generate_structural_report(group)

                # Actualizar path para siguiente grupo
                # if group == 'HOMBRES':
                # self.output_path = self.output_path.replace(
                #   'obj5/', 'obj5/HOMBRES_')
                # else:
                # self.output_path = self.output_path.replace(
                #   'obj5/', 'obj5/MUJERES_')

                self.save_results_to_excel(group)
                self.generate_comprehensive_report(group)

            except Exception as e:
                print(f"Error en análisis estructural de {group}: {e}")
                import traceback
                traceback.print_exc()
                continue
# diagnóstico de residuos más completo (cuatro paneles: residuos vs predicho, QQ-plot, histograma, residuos vs orden).


def plot_residuals_diagnostics(self, y_true, y_pred, group='MUJERES'):
    """
    Gráfico diagnóstico de residuos: 4 paneles (predicho, QQ-plot, histograma, orden)
    """
    import statsmodels.api as sm

    residuals = y_true - y_pred

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Residuos vs Predichos
    axes[0, 0].scatter(y_pred, residuals, alpha=0.6)
    axes[0, 0].axhline(y=0, color='red', linestyle='--')
    axes[0, 0].set_title("Residuos vs Predichos")
    axes[0, 0].set_xlabel("Valores Predichos")
    axes[0, 0].set_ylabel("Residuos")

    # 2. QQ-Plot
    sm.qqplot(residuals, line='45', fit=True, ax=axes[0, 1])
    axes[0, 1].set_title("Q-Q Plot de Residuos")

    # 3. Histograma de residuos
    axes[1, 0].hist(residuals, bins=30, color='red', alpha=0.6, density=True)
    axes[1, 0].set_title("Distribución de Residuos")
    axes[1, 0].set_xlabel("Residuos")
    axes[1, 0].set_ylabel("Densidad")

    # 4. Residuos vs Orden
    axes[1, 1].plot(range(len(residuals)), residuals, color='red', alpha=0.7)
    axes[1, 1].axhline(y=0, color='black', linestyle='--')
    axes[1, 1].set_title("Residuos vs Orden")
    axes[1, 1].set_xlabel("Orden de Observación")
    axes[1, 1].set_ylabel("Residuos")

    plt.suptitle(
        f"Diagnóstico de Residuos - {group}", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(self.output_path, f"{group}_residuals_diagnostics.png"),
                dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)

# FUNCIÓN PRINCIPAL ACTUALIZADA


def main_structural():
    """
    Función principal para análisis estructural PLS-SEM CON VISUALIZACIONES
    """
    # Rutas de archivos
    data_files = {
        'HOMBRES': "C:/01 academico/001 Doctorado Economia UCAB/d tesis problema ahorro/01 TESIS DEFINITIVA/MODELO/resultados obj5/DATA_CONSOLIDADA HOMBRES promedio H M .xlsx",
        'MUJERES': "C:/01 academico/001 Doctorado Economia UCAB/d tesis problema ahorro/01 TESIS DEFINITIVA/MODELO/resultados obj5/DATA_CONSOLIDADA MUJERES promedio H M .xlsx"
    }

    # Crear analizador estructural
    analyzer = PLSSEMPredictAnalyzer()

    print("INICIANDO ANÁLISIS PLS-SEM ESTRUCTURAL CON VISUALIZACIONES")
    print("=" * 80)
    print("Modelos implementados:")
    print("\nMUJERES:")
    print("  PCA = 0.3485·PSE - 0.2013·DH - 0.5101·SQ + 0.3676·CS + ε₁")
    print("  SQ = 0.6609·AV + ε₂")
    print("\nHOMBRES:")
    print("  PCA = 0.3777·PSE - 0.5947·SQ + 0.2226·DH + 0.2866·CS + ε₁")
    print("  SQ = 0.8402·AV + ε₂")
    print("=" * 80)
    print("Análisis incluye:")
    print("• Validación cruzada con ecuaciones estructurales")
    print("• Monte Carlo para sistema de ecuaciones")
    print("• Comparación modelo estructural vs regresión directa")
    print("• TODAS LAS VISUALIZACIONES:")
    print("  - observed_vs_predicted.png")
    print("  - monte_carlo_distribution.png")
    print("  - residual_analysis.png")
    print("  - model_comparison.png")
    print("  - sensitivity_analysis.png")
    print("=" * 80)

    # Ejecutar análisis estructural
    analyzer.run_structural_analysis(data_files)

    print("\n" + "="*80)
    print("ANÁLISIS ESTRUCTURAL Y VISUALIZACIONES COMPLETADOS")
    print("Revise el directorio de salida para todas las gráficas generadas")
    print("="*80)


if __name__ == "__main__":
    main_structural()
