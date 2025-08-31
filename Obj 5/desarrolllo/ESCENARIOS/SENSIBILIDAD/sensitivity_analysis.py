"""
ANÁLISIS DE SENSIBILIDAD PLS-SEM INTERACTIVO
Código completo para ejecutar localmente

Autor: MSc. JESUS FERNANDO SALAZAR ROJAS   Análisis Doctoral
Archivo de datos: "C:\\01 academico\\001 Doctorado Economia UCAB\\d tesis problema ahorro\\01 TESIS DEFINITIVA\\MODELO\\resultados obj5_1\\corrida scores sin intermedia\\SCORE HM.xlsx"
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.widgets import Slider, Button, RadioButtons
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as pyo
import warnings
warnings.filterwarnings('ignore')

# Configurar matplotlib para mejor visualización
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

os.system('cls' if os.name == 'nt' else 'clear')


class PLSSEMSensitivityAnalyzer:
    def __init__(self, excel_path):
        """
        Inicializar el analizador de sensibilidad PLS-SEM

        Parameters:
        excel_path (str): Ruta al archivo Excel con los scores
        """
        self.excel_path = excel_path
        self.data = None
        self.current_group = 'Hah'
        self.sensitivity_variable = 'edad'
        self.variable_value = 35

        # DATOS REALES - PESOS PARA HOMBRES (Hah)
        self.men_weights = {
            'PCA': {'PPCA': 1.0000},
            'PSE': {'PCA2': -0.3557, 'PCA4': 0.2800, 'PCA5': 0.8343},
            'DH': {'DH3': 0.7097, 'DH4': 0.4376},
            'SQ': {'SQ1': 0.3816, 'SQ2': 0.5930, 'SQ3': 0.3358},
            'AV': {'AV1': 0.1165, 'AV2': 0.3009, 'AV3': 0.6324, 'AV5': 0.3979},
            'CS': {'CS2': 0.5733, 'CS3': 0.4983, 'CS5': 0.1597}
        }

        # DATOS REALES - PESOS PARA MUJERES (Mah)
        self.women_weights = {
            'PCA': {'PPCA': 1.0000},
            'PSE': {'PCA2': -0.5168, 'PCA4': -0.0001, 'PCA5': 0.8496},
            'DH': {'DH2': 0.0305, 'DH3': 0.3290, 'DH4': 0.0660, 'DH5': 0.8397},
            'SQ': {'SQ1': 0.5458, 'SQ2': 0.4646, 'SQ3': 0.2946},
            'AV': {'AV1': 0.1920, 'AV2': 0.4430, 'AV3': 0.7001, 'AV5': 0.1276},
            'CS': {'CS2': 0.5452, 'CS3': 0.5117, 'CS5': 0.2631}
        }

        # ECUACIONES DEL MODELO
        self.men_equation = {
            'PSE': 0.3777, 'DH': 0.2226, 'SQ': -0.5947, 'CS': 0.2866, 'intercept': 0
        }

        self.women_equation = {
            'PSE': 0.3485, 'DH': -0.2013, 'SQ': -0.5101, 'CS': 0.3676, 'intercept': 0
        }

        # LOADINGS (para referencia)
        self.men_loadings = {
            'PCA': {'PPCA': 1.0000},
            'PSE': {'PCA2': -0.5276, 'PCA4': 0.1737, 'PCA5': 0.8561},
            'DH': {'DH2': 0.0324, 'DH3': 0.3490, 'DH4': 0.0700, 'DH5': 0.8906},
            'SQ': {'SQ1': 0.7392, 'SQ2': 0.6293, 'SQ3': 0.3990},
            'AV': {'AV1': 0.1897, 'AV2': 0.4379, 'AV3': 0.6920, 'AV5': 0.1262},
            'CS': {'CS2': 0.6885, 'CS3': 0.6462, 'CS5': 0.3323}
        }

        self.women_loadings = {
            'PCA': {'PPCA': 1.0000},
            'PSE': {'PCA2': -0.2947, 'PCA4': 0.4630, 'PCA5': 0.9176},
            'DH': {'DH3': 0.8921, 'DH4': 0.5501},
            'SQ': {'SQ1': 0.5145, 'SQ2': 0.7995, 'SQ3': 0.4527},
            'AV': {'AV1': 0.1366, 'AV2': 0.3530, 'AV3': 0.7419, 'AV5': 0.4667},
            'CS': {'CS2': 0.8235, 'CS3': 0.7157, 'CS5': 0.2294}
        }

        self.load_data()

    def load_data(self):
        """Cargar datos desde Excel"""
        try:
            print(f"Cargando datos desde: {self.excel_path}")
            self.data = pd.read_excel(self.excel_path)
            print(
                f"Datos cargados exitosamente: {self.data.shape[0]} filas, {self.data.shape[1]} columnas")
            print(f"Columnas disponibles: {list(self.data.columns)}")
            print(
                f"Grupos únicos: {self.data['GRUPO'].unique() if 'GRUPO' in self.data.columns else 'No encontrado'}")
            print("\nPrimeras 5 filas:")
            print(self.data.head())

        except Exception as e:
            print(f"Error al cargar datos: {e}")
            print("Generando datos de muestra...")
            self.generate_sample_data()

    def generate_sample_data(self):
        """Generar datos de muestra si no se puede cargar el archivo"""
        np.random.seed(42)
        n_samples = 200

        # Generar casos para hombres
        men_data = []
        for i in range(n_samples//2):
            men_data.append({
                'Case': i+1,
                'PCA': np.random.normal(-0.5, 1.2),
                'PSE': np.random.normal(0.2, 1.0),
                'SQ': np.random.normal(-0.1, 0.8),
                'DH': np.random.normal(0.3, 1.1),
                'AV': np.random.normal(-0.2, 0.9),
                'CS': np.random.normal(0.1, 1.0),
                'GRUPO': 'Hah'
            })

        # Generar casos para mujeres
        women_data = []
        for i in range(n_samples//2):
            women_data.append({
                'Case': i+1,
                'PCA': np.random.normal(0.3, 1.1),
                'PSE': np.random.normal(-0.1, 0.9),
                'SQ': np.random.normal(0.2, 1.0),
                'DH': np.random.normal(-0.4, 1.2),
                'AV': np.random.normal(0.4, 0.8),
                'CS': np.random.normal(-0.3, 1.1),
                'GRUPO': 'Mah'
            })

        self.data = pd.DataFrame(men_data + women_data)
        print("Datos de muestra generados exitosamente")

    def get_current_group_data(self):
        """Obtener datos del grupo actual"""
        if self.current_group == 'Hah':
            return {
                'weights': self.men_weights,
                'equation': self.men_equation,
                'loadings': self.men_loadings
            }
        else:
            return {
                'weights': self.women_weights,
                'equation': self.women_equation,
                'loadings': self.women_loadings
            }

    def calculate_indicator_effects(self, variable, value, baseline=35):
        """
        Calcular efectos en indicadores específicos
        """
        normalized_effect = (value - baseline) / 15

        effects = {
            'edad': {
                # Efectos diferenciados por grupo
                **({'DH3': 0.12 * normalized_effect, 'DH4': 0.08 * normalized_effect,
                   'DH5': 0.15 * normalized_effect, 'AV1': -0.06 * normalized_effect,
                    'AV3': -0.10 * normalized_effect, 'SQ1': 0.14 * normalized_effect,
                    'SQ2': 0.10 * normalized_effect, 'CS2': 0.09 * normalized_effect,
                    'PCA2': 0.18 * normalized_effect, 'PCA5': 0.12 * normalized_effect,
                    'PPCA': 0.25 * normalized_effect} if self.current_group == 'Hah' else
                   {'DH3': 0.18 * normalized_effect, 'DH4': 0.12 * normalized_effect,
                   'DH5': 0.10 * normalized_effect, 'AV2': -0.08 * normalized_effect,
                    'AV3': -0.12 * normalized_effect, 'SQ1': 0.16 * normalized_effect,
                    'SQ3': 0.08 * normalized_effect, 'CS2': 0.11 * normalized_effect,
                    'CS3': 0.09 * normalized_effect, 'PCA2': 0.20 * normalized_effect,
                    'PCA4': 0.08 * normalized_effect, 'PPCA': 0.28 * normalized_effect})
            },
            'ingreso': {
                **({'CS2': 0.20 * normalized_effect, 'CS3': 0.16 * normalized_effect,
                   'CS5': 0.12 * normalized_effect, 'DH3': 0.14 * normalized_effect,
                    'DH4': 0.10 * normalized_effect, 'AV2': 0.15 * normalized_effect,
                    'AV3': 0.12 * normalized_effect, 'PPCA': 0.22 * normalized_effect} if self.current_group == 'Hah' else
                   {'CS2': 0.22 * normalized_effect, 'CS3': 0.18 * normalized_effect,
                   'DH2': 0.08 * normalized_effect, 'DH3': 0.16 * normalized_effect,
                    'AV1': 0.10 * normalized_effect, 'AV2': 0.18 * normalized_effect,
                    'PCA4': 0.14 * normalized_effect, 'PPCA': 0.24 * normalized_effect})
            },
            'educacion': {
                **({'DH3': 0.25 * normalized_effect, 'DH4': 0.18 * normalized_effect,
                   'DH5': 0.22 * normalized_effect, 'AV1': 0.12 * normalized_effect,
                    'AV3': 0.16 * normalized_effect, 'SQ2': 0.14 * normalized_effect,
                    'PCA2': 0.20 * normalized_effect, 'PCA4': 0.15 * normalized_effect,
                    'PPCA': 0.28 * normalized_effect} if self.current_group == 'Hah' else
                   {'DH3': 0.30 * normalized_effect, 'DH4': 0.20 * normalized_effect,
                   'AV1': 0.14 * normalized_effect, 'AV2': 0.18 * normalized_effect,
                    'AV3': 0.20 * normalized_effect, 'SQ1': 0.12 * normalized_effect,
                    'SQ2': 0.16 * normalized_effect, 'PCA4': 0.18 * normalized_effect,
                    'PCA5': 0.14 * normalized_effect, 'PPCA': 0.32 * normalized_effect})
            }
        }

        return effects.get(variable, {})

    def recalculate_construct_scores(self, indicator_effects, base_scores):
        """
        Recalcular scores de constructos usando weights
        """
        group_data = self.get_current_group_data()
        new_scores = base_scores.copy()

        for construct in group_data['weights']:
            weighted_effect = 0
            total_abs_weight = 0

            for indicator, weight in group_data['weights'][construct].items():
                effect = indicator_effects.get(indicator, 0)
                weighted_effect += weight * effect
                total_abs_weight += abs(weight)

            if total_abs_weight > 0:
                new_scores[construct] = base_scores[construct] + \
                    weighted_effect

        return new_scores

    def calculate_pca(self, scores):
        """Calcular PCA usando la ecuación del modelo"""
        eq = self.get_current_group_data()['equation']
        return (eq['intercept'] +
                eq['PSE'] * scores['PSE'] +
                eq['DH'] * scores['DH'] +
                eq['SQ'] * scores['SQ'] +
                eq['CS'] * scores['CS'])

    def run_sensitivity_analysis(self, case_index=0, variable='edad', value_range=(18, 70), n_steps=20):
        """
        Ejecutar análisis de sensibilidad completo
        """
        print(f"\n=== ANÁLISIS DE SENSIBILIDAD ===")
        print(f"Grupo: {self.current_group}")
        print(f"Variable: {variable}")
        print(f"Rango: {value_range[0]} - {value_range[1]}")

        # Obtener datos del grupo
        group_data = self.data[self.data['GRUPO'] == self.current_group].copy()
        if len(group_data) == 0:
            print(
                f"No se encontraron datos para el grupo {self.current_group}")
            return None

        # Caso base
        if case_index >= len(group_data):
            case_index = 0

        base_case = group_data.iloc[case_index]
        print(f"Caso base: {base_case['Case']} (índice {case_index})")

        # Crear rango de valores
        values = np.linspace(value_range[0], value_range[1], n_steps)
        results = []

        baseline_pca = self.calculate_pca(base_case)

        for value in values:
            # Calcular efectos en indicadores
            indicator_effects = self.calculate_indicator_effects(
                variable, value)

            # Recalcular scores de constructos
            new_scores = self.recalculate_construct_scores(
                indicator_effects, base_case)

            # Calcular nuevo PCA
            calculated_pca = self.calculate_pca(new_scores)

            results.append({
                'variable_value': value,
                'PCA_Calculated': calculated_pca,
                'PCA_Original': new_scores['PCA'],
                'PCA_Delta': calculated_pca - baseline_pca,
                'PSE': new_scores['PSE'],
                'DH': new_scores['DH'],
                'SQ': new_scores['SQ'],
                'CS': new_scores['CS'],
                'AV': new_scores['AV'],
                'PSE_Delta': new_scores['PSE'] - base_case['PSE'],
                'DH_Delta': new_scores['DH'] - base_case['DH'],
                'SQ_Delta': new_scores['SQ'] - base_case['SQ'],
                'CS_Delta': new_scores['CS'] - base_case['CS'],
                'AV_Delta': new_scores['AV'] - base_case['AV']
            })

        results_df = pd.DataFrame(results)

        print(f"Análisis completado: {len(results)} escenarios")
        print(f"PCA baseline: {baseline_pca:.4f}")
        print(
            f"Rango PCA calculado: {results_df['PCA_Calculated'].min():.4f} - {results_df['PCA_Calculated'].max():.4f}")

        return results_df

    def create_interactive_dashboard(self, results_df, base_case, variable='edad'):
        """
        Crear dashboard interactivo con Plotly
        """
        # Crear subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                f'Efecto en PCA - {self.current_group}',
                'Cambios en Constructos',
                f'Delta PCA vs {variable}',
                'Efectos Relativos por Constructo'
            ),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"type": "bar"}]]
        )

        # 1. Efecto principal en PCA
        fig.add_trace(
            go.Scatter(x=results_df['variable_value'], y=results_df['PCA_Calculated'],
                       mode='lines+markers', name='PCA Calculado',
                       line=dict(color='blue', width=3)),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(x=results_df['variable_value'], y=results_df['PCA_Original'],
                       mode='lines+markers', name='PCA Score',
                       line=dict(color='red', width=2, dash='dash')),
            row=1, col=1
        )

        # 2. Cambios en constructos
        constructs = ['PSE', 'DH', 'SQ', 'CS', 'AV']
        colors = ['green', 'orange', 'red', 'purple', 'cyan']

        for i, construct in enumerate(constructs):
            fig.add_trace(
                go.Scatter(x=results_df['variable_value'], y=results_df[construct],
                           mode='lines', name=construct,
                           line=dict(color=colors[i], width=2)),
                row=1, col=2
            )

        # 3. Delta PCA
        fig.add_trace(
            go.Scatter(x=results_df['variable_value'], y=results_df['PCA_Delta'],
                       mode='lines+markers', name='Δ PCA',
                       line=dict(color='darkblue', width=3),
                       fill='tonexty'),
            row=2, col=1
        )

        # 4. Efectos relativos (barras)
        final_deltas = results_df.iloc[-1]
        construct_effects = [final_deltas[f'{c}_Delta'] for c in constructs]

        fig.add_trace(
            go.Bar(x=constructs, y=construct_effects, name='Efecto Final',
                   marker_color=['green', 'orange', 'red', 'purple', 'cyan']),
            row=2, col=2
        )

        # Configurar layout
        fig.update_layout(
            title=f'Análisis de Sensibilidad PLS-SEM - {self.current_group} ({variable.title()})',
            height=800,
            showlegend=True
        )

        fig.update_xaxes(title_text=variable.title(), row=1, col=1)
        fig.update_xaxes(title_text=variable.title(), row=1, col=2)
        fig.update_xaxes(title_text=variable.title(), row=2, col=1)
        fig.update_xaxes(title_text="Constructos", row=2, col=2)

        fig.update_yaxes(title_text="PCA", row=1, col=1)
        fig.update_yaxes(title_text="Scores", row=1, col=2)
        fig.update_yaxes(title_text="Δ PCA", row=2, col=1)
        fig.update_yaxes(title_text="Δ Score", row=2, col=2)

        return fig

    def create_comparison_analysis(self):
        """
        Crear análisis comparativo entre hombres y mujeres
        """
        print("\n=== ANÁLISIS COMPARATIVO HOMBRES vs MUJERES ===")

        results_comparison = []
        variables = ['edad', 'ingreso', 'educacion']

        for variable in variables:
            for group in ['Hah', 'Mah']:
                self.current_group = group
                results = self.run_sensitivity_analysis(
                    case_index=0,
                    variable=variable,
                    value_range=(20, 60) if variable == 'edad' else
                    (30, 120) if variable == 'ingreso' else (8, 20),
                    n_steps=15
                )

                if results is not None:
                    summary = {
                        'Variable': variable,
                        'Grupo': group,
                        'PCA_Min': results['PCA_Calculated'].min(),
                        'PCA_Max': results['PCA_Calculated'].max(),
                        'PCA_Range': results['PCA_Calculated'].max() - results['PCA_Calculated'].min(),
                        'Max_Delta': results['PCA_Delta'].abs().max(),
                        'PSE_Sensitivity': results['PSE_Delta'].abs().max(),
                        'DH_Sensitivity': results['DH_Delta'].abs().max(),
                        'SQ_Sensitivity': results['SQ_Delta'].abs().max(),
                        'CS_Sensitivity': results['CS_Delta'].abs().max()
                    }
                    results_comparison.append(summary)

        comparison_df = pd.DataFrame(results_comparison)

        # Visualizar comparación
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(
            'Análisis Comparativo: Sensibilidad por Grupo y Variable', fontsize=16)

        # Sensibilidad de PCA por variable y grupo
        pivot_range = comparison_df.pivot(
            index='Variable', columns='Grupo', values='PCA_Range')
        pivot_range.plot(
            kind='bar', ax=axes[0, 0], title='Rango de Variación PCA')
        axes[0, 0].set_ylabel('Rango PCA')
        axes[0, 0].tick_params(axis='x', rotation=45)

        # Máximo delta por grupo
        pivot_delta = comparison_df.pivot(
            index='Variable', columns='Grupo', values='Max_Delta')
        pivot_delta.plot(kind='bar', ax=axes[0, 1], title='Máximo Δ PCA')
        axes[0, 1].set_ylabel('Max |Δ PCA|')
        axes[0, 1].tick_params(axis='x', rotation=45)

        # Sensibilidad por constructo - Hombres
        men_data = comparison_df[comparison_df['Grupo'] == 'Hah']
        construct_cols = ['PSE_Sensitivity', 'DH_Sensitivity',
                          'SQ_Sensitivity', 'CS_Sensitivity']
        men_sensitivity = men_data.set_index('Variable')[construct_cols]
        men_sensitivity.plot(
            kind='bar', ax=axes[1, 0], title='Sensibilidad Constructos - Hombres')
        axes[1, 0].set_ylabel('Max |Δ Score|')
        axes[1, 0].tick_params(axis='x', rotation=45)

        # Sensibilidad por constructo - Mujeres
        women_data = comparison_df[comparison_df['Grupo'] == 'Mah']
        women_sensitivity = women_data.set_index('Variable')[construct_cols]
        women_sensitivity.plot(
            kind='bar', ax=axes[1, 1], title='Sensibilidad Constructos - Mujeres')
        axes[1, 1].set_ylabel('Max |Δ Score|')
        axes[1, 1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.show()

        return comparison_df

    def export_results(self, results_df, filename_prefix="sensitivity_analysis"):
        """Exportar resultados a Excel"""
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}_{self.current_group}_{timestamp}.xlsx"

        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            results_df.to_excel(
                writer, sheet_name='Sensitivity_Results', index=False)

            # Información del modelo
            model_info = pd.DataFrame([
                {'Parametro': 'Grupo', 'Valor': self.current_group},
                {'Parametro': 'Ecuacion_PSE', 'Valor': self.get_current_group_data()[
                    'equation']['PSE']},
                {'Parametro': 'Ecuacion_DH', 'Valor': self.get_current_group_data()[
                    'equation']['DH']},
                {'Parametro': 'Ecuacion_SQ', 'Valor': self.get_current_group_data()[
                    'equation']['SQ']},
                {'Parametro': 'Ecuacion_CS', 'Valor': self.get_current_group_data()[
                    'equation']['CS']},
            ])
            model_info.to_excel(writer, sheet_name='Model_Info', index=False)

        print(f"Resultados exportados a: {filename}")
        return filename

# FUNCIÓN PRINCIPAL PARA EJECUTAR EL ANÁLISIS


def main():
    """
    Función principal para ejecutar el análisis de sensibilidad
    """
    print("="*60)
    print("ANÁLISIS DE SENSIBILIDAD PLS-SEM")
    print("Tesis Doctoral - Problema del Ahorro")
    print("="*60)

    # Ruta del archivo
    excel_path = r"C:\01 academico\001 Doctorado Economia UCAB\d tesis problema ahorro\01 TESIS DEFINITIVA\MODELO\resultados obj5_1\corrida scores sin intermedia\SCORE HM.xlsx"

    # Inicializar analizador
    analyzer = PLSSEMSensitivityAnalyzer(excel_path)

    print(f"\nDatos cargados: {len(analyzer.data)} observaciones")
    print(
        f"Grupos disponibles: {analyzer.data['GRUPO'].value_counts().to_dict()}")

    # ANÁLISIS 1: HOMBRES - EFECTO DE LA EDAD
    print("\n" + "="*40)
    print("ANÁLISIS 1: HOMBRES - EFECTO EDAD")
    print("="*40)

    analyzer.current_group = 'Hah'
    results_men_age = analyzer.run_sensitivity_analysis(
        case_index=0,
        variable='edad',
        value_range=(20, 65),
        n_steps=25
    )

    if results_men_age is not None:
        # Dashboard interactivo
        fig1 = analyzer.create_interactive_dashboard(results_men_age,
                                                     analyzer.data[analyzer.data['GRUPO']
                                                                   == 'Hah'].iloc[0],
                                                     'edad')
        fig1.show()

        # Exportar resultados
        analyzer.export_results(results_men_age, "hombres_edad_sensitivity")

    # ANÁLISIS 2: MUJERES - EFECTO DE LA EDAD
    print("\n" + "="*40)
    print("ANÁLISIS 2: MUJERES - EFECTO EDAD")
    print("="*40)

    analyzer.current_group = 'Mah'
    results_women_age = analyzer.run_sensitivity_analysis(
        case_index=0,
        variable='edad',
        value_range=(20, 65),
        n_steps=25
    )

    if results_women_age is not None:
        # Dashboard interactivo
        fig2 = analyzer.create_interactive_dashboard(results_women_age,
                                                     analyzer.data[analyzer.data['GRUPO']
                                                                   == 'Mah'].iloc[0],
                                                     'edad')
        fig2.show()

        # Exportar resultados
        analyzer.export_results(results_women_age, "mujeres_edad_sensitivity")

    # ANÁLISIS 3: COMPARATIVO COMPLETO
    print("\n" + "="*40)
    print("ANÁLISIS 3: COMPARATIVO HOMBRES vs MUJERES")
    print("="*40)

    comparison_results = analyzer.create_comparison_analysis()
    print("\nTabla Comparativa:")
    print(comparison_results.to_string(index=False))

    # Exportar comparación
    comparison_results.to_excel(
        "comparative_sensitivity_analysis.xlsx", index=False)

    print("\n" + "="*60)
    print("ANÁLISIS COMPLETADO")
    print("="*60)
    print("Archivos generados:")
    print("- hombres_edad_sensitivity_[timestamp].xlsx")
    print("- mujeres_edad_sensitivity_[timestamp].xlsx")
    print("- comparative_sensitivity_analysis.xlsx")
    print("\nGráficos interactivos mostrados en el navegador")

    # ANÁLISIS ADICIONAL: EFECTOS POR VARIABLE
    print("\n" + "="*40)
    print("ANÁLISIS ADICIONAL: MÚLTIPLES VARIABLES")
    print("="*40)

    # Crear análisis por cada variable para cada grupo
    variables_analysis = {
        'edad': (18, 70),
        'ingreso': (25, 150),
        'educacion': (8, 20)
    }

    for variable, value_range in variables_analysis.items():
        print(f"\n--- Analizando {variable.upper()} ---")

        for group in ['Hah', 'Mah']:
            analyzer.current_group = group
            print(
                f"\nGrupo: {group} ({'Hombres' if group == 'Hah' else 'Mujeres'})")

            results = analyzer.run_sensitivity_analysis(
                case_index=0,
                variable=variable,
                value_range=value_range,
                n_steps=20
            )

            if results is not None:
                # Métricas de sensibilidad
                pca_sensitivity = results['PCA_Delta'].abs().max()
                construct_sensitivities = {
                    'PSE': results['PSE_Delta'].abs().max(),
                    'DH': results['DH_Delta'].abs().max(),
                    'SQ': results['SQ_Delta'].abs().max(),
                    'CS': results['CS_Delta'].abs().max(),
                    'AV': results['AV_Delta'].abs().max()
                }

                print(f"  Sensibilidad PCA: {pca_sensitivity:.4f}")
                print(
                    f"  Constructo más sensible: {max(construct_sensitivities, key=construct_sensitivities.get)} ({max(construct_sensitivities.values()):.4f})")

                # Crear gráfico específico
                fig = analyzer.create_interactive_dashboard(results,
                                                            analyzer.data[analyzer.data['GRUPO']
                                                                          == group].iloc[0],
                                                            variable)
                fig.update_layout(
                    title=f'Sensibilidad {variable.title()} - {group} ({'Hombres' if group == 'Hah' else 'Mujeres'})')

                # Guardar como HTML para revisión posterior
                filename_html = f"sensitivity_{variable}_{group}.html"
                pyo.plot(fig, filename=filename_html, auto_open=False)
                print(f"  Gráfico guardado: {filename_html}")

                # Exportar datos detallados
                analyzer.export_results(
                    results, f"detailed_{variable}_{group}")

# FUNCIONES AUXILIARES PARA ANÁLISIS ESPECÍFICOS


def analyze_age_segments(analyzer):
    """
    Análisis por segmentos de edad
    """
    print("\n" + "="*40)
    print("ANÁLISIS POR SEGMENTOS DE EDAD")
    print("="*40)

    age_segments = {
        'Jóvenes': (18, 30),
        'Adultos': (31, 50),
        'Maduros': (51, 70)
    }

    segment_results = []

    for segment_name, (min_age, max_age) in age_segments.items():
        print(f"\n--- Segmento: {segment_name} ({min_age}-{max_age} años) ---")

        for group in ['Hah', 'Mah']:
            analyzer.current_group = group

            # Análisis para el punto medio del segmento
            mid_age = (min_age + max_age) / 2

            # Simular efecto en el punto medio
            indicator_effects = analyzer.calculate_indicator_effects(
                'edad', mid_age)
            base_case = analyzer.data[analyzer.data['GRUPO'] == group].iloc[0]
            new_scores = analyzer.recalculate_construct_scores(
                indicator_effects, base_case)
            calculated_pca = analyzer.calculate_pca(new_scores)
            baseline_pca = analyzer.calculate_pca(base_case)

            segment_results.append({
                'Segmento': segment_name,
                'Grupo': group,
                'Edad_Media': mid_age,
                'PCA_Baseline': baseline_pca,
                'PCA_Ajustado': calculated_pca,
                'Delta_PCA': calculated_pca - baseline_pca,
                'PSE_Ajustado': new_scores['PSE'],
                'DH_Ajustado': new_scores['DH'],
                'SQ_Ajustado': new_scores['SQ'],
                'CS_Ajustado': new_scores['CS'],
                'AV_Ajustado': new_scores['AV']
            })

            print(
                f"  {group}: PCA = {calculated_pca:.4f} (Δ = {calculated_pca - baseline_pca:+.4f})")

    segments_df = pd.DataFrame(segment_results)

    # Visualizar por segmentos
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # PCA por segmento y grupo
    pivot_pca = segments_df.pivot(
        index='Segmento', columns='Grupo', values='Delta_PCA')
    pivot_pca.plot(kind='bar', ax=axes[0],
                   title='Efecto de Edad en PCA por Segmento')
    axes[0].set_ylabel('Δ PCA')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].axhline(y=0, color='black', linestyle='--', alpha=0.5)

    # Efectos promedio por grupo
    group_effects = segments_df.groupby(
        'Grupo')['Delta_PCA'].agg(['mean', 'std'])
    group_effects.plot(
        kind='bar', ax=axes[1], title='Efectos Promedio por Grupo')
    axes[1].set_ylabel('Δ PCA')
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()

    # Exportar resultados por segmentos
    segments_df.to_excel("age_segments_analysis.xlsx", index=False)
    print(f"\nResultados por segmentos exportados a: age_segments_analysis.xlsx")

    return segments_df


def create_sensitivity_report(analyzer):
    """
    Crear reporte ejecutivo de sensibilidad
    """
    print("\n" + "="*40)
    print("GENERANDO REPORTE EJECUTIVO")
    print("="*40)

    report_data = []

    # Análisis para cada combinación de grupo y variable
    variables = ['edad', 'ingreso', 'educacion']
    groups = ['Hah', 'Mah']

    for group in groups:
        analyzer.current_group = group
        group_name = 'Hombres' if group == 'Hah' else 'Mujeres'

        print(f"\nAnalizando grupo: {group_name}")

        for variable in variables:
            value_range = {
                'edad': (20, 60),
                'ingreso': (30, 120),
                'educacion': (10, 18)
            }[variable]

            results = analyzer.run_sensitivity_analysis(
                case_index=0,
                variable=variable,
                value_range=value_range,
                n_steps=15
            )

            if results is not None:
                # Calcular métricas clave
                max_pca_increase = results['PCA_Delta'].max()
                max_pca_decrease = results['PCA_Delta'].min()
                pca_volatility = results['PCA_Delta'].std()

                # Constructo más sensible
                construct_deltas = {
                    'PSE': results['PSE_Delta'].abs().max(),
                    'DH': results['DH_Delta'].abs().max(),
                    'SQ': results['SQ_Delta'].abs().max(),
                    'CS': results['CS_Delta'].abs().max(),
                    'AV': results['AV_Delta'].abs().max()
                }
                most_sensitive_construct = max(
                    construct_deltas, key=construct_deltas.get)
                max_construct_sensitivity = construct_deltas[most_sensitive_construct]

                report_data.append({
                    'Grupo': group_name,
                    'Variable': variable.title(),
                    'Rango_Analizado': f"{value_range[0]}-{value_range[1]}",
                    'Max_Incremento_PCA': max_pca_increase,
                    'Max_Decremento_PCA': max_pca_decrease,
                    'Volatilidad_PCA': pca_volatility,
                    'Constructo_Mas_Sensible': most_sensitive_construct,
                    'Max_Sensibilidad_Constructo': max_construct_sensitivity,
                    'Interpretacion': f"PCA varía entre {max_pca_decrease:.3f} y {max_pca_increase:.3f}"
                })

                print(
                    f"  {variable}: PCA Δ = [{max_pca_decrease:.3f}, {max_pca_increase:.3f}], Constructo más sensible: {most_sensitive_construct}")

    report_df = pd.DataFrame(report_data)

    # Crear reporte ejecutivo
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"executive_sensitivity_report_{timestamp}.xlsx"

    with pd.ExcelWriter(report_filename, engine='openpyxl') as writer:
        # Hoja principal con resultados
        report_df.to_excel(writer, sheet_name='Resumen_Ejecutivo', index=False)

        # Hoja con interpretaciones detalladas
        interpretations = pd.DataFrame([
            {'Aspecto': 'Objetivo', 'Descripcion': 'Analizar sensibilidad del modelo PLS-SEM ante cambios en variables demográficas'},
            {'Aspecto': 'Grupos',
                'Descripcion': 'Hah (Hombres) vs Mah (Mujeres)'},
            {'Aspecto': 'Variables', 'Descripcion': 'Edad, Ingreso, Educación'},
            {'Aspecto': 'Metodología',
                'Descripcion': 'Modificación de indicadores según weights y recálculo de constructos'},
            {'Aspecto': 'Ecuación Hombres',
                'Descripcion': 'PCA = 0.3777·PSE + 0.2226·DH - 0.5947·SQ + 0.2866·CS'},
            {'Aspecto': 'Ecuación Mujeres',
                'Descripcion': 'PCA = 0.3485·PSE - 0.2013·DH - 0.5101·SQ + 0.3676·CS'}
        ])
        interpretations.to_excel(writer, sheet_name='Metodologia', index=False)

        # Hoja con weights del modelo
        weights_summary = []
        for group_name, group_code in [('Hombres', 'Hah'), ('Mujeres', 'Mah')]:
            analyzer.current_group = group_code
            group_data = analyzer.get_current_group_data()

            for construct, indicators in group_data['weights'].items():
                for indicator, weight in indicators.items():
                    weights_summary.append({
                        'Grupo': group_name,
                        'Constructo': construct,
                        'Indicador': indicator,
                        'Weight': weight
                    })

        weights_df = pd.DataFrame(weights_summary)
        weights_df.to_excel(writer, sheet_name='Weights_Modelo', index=False)

    print(f"\nReporte ejecutivo generado: {report_filename}")

    # Mostrar resumen en consola
    print("\n" + "="*60)
    print("RESUMEN EJECUTIVO - ANÁLISIS DE SENSIBILIDAD")
    print("="*60)

    print("\nHALLAZGOS PRINCIPALES:")

    # Grupo más sensible
    max_volatility_row = report_df.loc[report_df['Volatilidad_PCA'].idxmax()]
    print(
        f"• Grupo más sensible: {max_volatility_row['Grupo']} (Variable: {max_volatility_row['Variable']})")
    print(f"  Volatilidad PCA: {max_volatility_row['Volatilidad_PCA']:.4f}")

    # Variable con mayor impacto
    impact_by_variable = report_df.groupby(
        'Variable')['Volatilidad_PCA'].mean().sort_values(ascending=False)
    print(
        f"• Variable con mayor impacto promedio: {impact_by_variable.index[0]} ({impact_by_variable.iloc[0]:.4f})")

    # Diferencias entre grupos
    print(f"\n• COMPARACIÓN POR GRUPOS:")
    for variable in variables:
        var_data = report_df[report_df['Variable'] == variable.title()]
        if len(var_data) == 2:
            hombres_vol = var_data[var_data['Grupo'] ==
                                   'Hombres']['Volatilidad_PCA'].iloc[0]
            mujeres_vol = var_data[var_data['Grupo'] ==
                                   'Mujeres']['Volatilidad_PCA'].iloc[0]
            print(
                f"  {variable.title()}: Hombres ({hombres_vol:.4f}) vs Mujeres ({mujeres_vol:.4f})")

    print(f"\n• CONSTRUCTOS MÁS SENSIBLES:")
    construct_counts = report_df['Constructo_Mas_Sensible'].value_counts()
    for construct, count in construct_counts.items():
        print(f"  {construct}: {count} casos")

    return report_df


# EJECUCIÓN COMPLETA DEL ANÁLISIS
if __name__ == "__main__":
    # Ejecutar análisis principal
    main()

    # Ejecutar análisis adicionales
    excel_path = r"C:\01 academico\001 Doctorado Economia UCAB\d tesis problema ahorro\01 TESIS DEFINITIVA\MODELO\resultados obj5_1\corrida scores sin intermedia\SCORE HM.xlsx"
    analyzer = PLSSEMSensitivityAnalyzer(excel_path)

    # Análisis por segmentos de edad
    segments_results = analyze_age_segments(analyzer)

    # Reporte ejecutivo
    executive_report = create_sensitivity_report(analyzer)

    print("\n" + "="*60)
    print("TODOS LOS ANÁLISIS COMPLETADOS EXITOSAMENTE")
    print("="*60)
    print("Revisa los archivos generados en el directorio actual:")
    print("1. Gráficos interactivos HTML")
    print("2. Resultados detallados Excel")
    print("3. Reporte ejecutivo")
    print("4. Análisis comparativo")
    print("5. Análisis por segmentos de edad")

    input("\nPresiona Enter para salir...")

# INSTRUCCIONES DE USO:
"""
INSTRUCCIONES PARA EJECUTAR EL CÓDIGO:

1. INSTALAR LIBRERÍAS REQUERIDAS:
   pip install pandas numpy matplotlib seaborn plotly openpyxl

2. VERIFICAR RUTA DEL ARCHIVO:
   Asegúrate de que la ruta del Excel sea correcta:
   "C:\\01 academico\\001 Doctorado Economia UCAB\\d tesis problema ahorro\\01 TESIS DEFINITIVA\\MODELO\\resultados obj5_1\\corrida scores sin intermedia\\SCORE HM.xlsx"

3. EJECUTAR:
   python nombre_del_archivo.py

4. EL CÓDIGO GENERARÁ:
   - Gráficos interactivos que se abren en el navegador
   - Múltiples archivos Excel con resultados detallados
   - Reporte ejecutivo con hallazgos principales
   - Análisis comparativo entre grupos
   - Análisis por segmentos de edad

5. ESTRUCTURA ESPERADA DEL EXCEL:
   Columnas: Case, PCA, PSE, SQ, DH, AV, CS, GRUPO
   Valores GRUPO: 'Hah' para hombres, 'Mah' para mujeres

6. PERSONALIZACIÓN:
   - Modifica las variables en variables_analysis
   - Ajusta rangos de valores según tus datos
   - Cambia el número de pasos en n_steps para más/menos granularidad

7. OUTPUTS PRINCIPALES:
   - Dashboard interactivo por grupo y variable
   - Tabla comparativa de sensibilidades
   - Reporte ejecutivo con interpretaciones
   - Análisis por segmentos demográficos
"""
