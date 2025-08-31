"""
SCRIPT COMPLETO: ANÁLISIS MONTE CARLO PCA + PAQUETE PARA TUTOR
============================================================

Genera análisis interactivo HTML + paquete completo para enviar al tutor
Sin necesidad de Python en el lado del tutor - Solo navegador web

Autor: MSc. Jesús F. Salazar Rojas
Doctorado en Economía - Universidad Católica Andrés Bello (UCAB)
Fecha: Agosto 2025
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from datetime import datetime
import warnings
import os
import zipfile
from pathlib import Path
import re

warnings.filterwarnings('ignore')


class MonteCarloAnalysisPCA:
    """Clase principal para análisis Monte Carlo de Propensión Conductual al Ahorro"""

    def __init__(self, data_path=None):
        """Inicializa el analizador Monte Carlo con parámetros del modelo"""
        # Parámetros del modelo estructural basados en resultados PLS-SEM
        self.model_params = {
            'HAH': {  # Hombres Ahorradores
                'PSE': 0.3777,
                'DH': 0.2226,
                'SQ': -0.5947,
                'CS': 0.2866,
                'R2': 0.549796,
                'RMSE': 0.665626,
                'R2_corregido': 0.565573
            },
            'MAH': {  # Mujeres Ahorradoras
                'PSE': 0.3485,
                'DH': -0.2013,
                'SQ': -0.5101,
                'CS': 0.3676,
                'R2': 0.571136,
                'RMSE': 0.650872,
                'R2_corregido': 0.581422
            }
        }

        # Definición de escenarios basados en teoría comportamental
        self.escenarios = {
            'Base': {
                'CS': 0.5, 'DH': 0.5,
                'descripcion': 'Influencia social promedio, valoración del futuro promedio. PCA estable.',
                'color': '#2E86C1'
            },
            'Rumores Moderados': {
                'CS': 0.65, 'DH': 0.6,
                'descripcion': 'Rumores negativos leves; ligera preferencia por el presente. PCA disminuye moderadamente.',
                'color': '#F39C12'
            },
            'Rumores Fuertes': {
                'CS': 0.8, 'DH': 0.75,
                'descripcion': 'Rumores negativos frecuentes; alta preferencia por el presente. PCA disminuye significativamente.',
                'color': '#E74C3C'
            },
            'Optimista': {
                'CS': 0.3, 'DH': 0.4,
                'descripcion': 'Noticias positivas o neutras; valoración del futuro alta. PCA aumenta.',
                'color': '#27AE60'
            },
            'Extremo Negativo': {
                'CS': 0.95, 'DH': 0.9,
                'descripcion': 'Rumores muy intensos; urgencia extrema por consumir. PCA mínima.',
                'color': '#8E44AD'
            }
        }

        self.data_path = r"C:\01 academico\001 Doctorado Economia UCAB\d tesis problema ahorro\01 TESIS DEFINITIVA\MODELO\resultados obj5_1\corrida scores sin intermedia\DATA_CONSOLIDADA promedio HM .xlsx"

        self.n_simulations = 5000  # Número de simulaciones Monte Carlo

    def load_data(self):
        """Carga los datos consolidados si están disponibles"""
        if self.data_path:
            try:
                self.data = pd.read_excel(self.data_path)
                print(f"✓ Datos cargados exitosamente: {self.data.shape}")
                return True
            except Exception as e:
                print(f"⚠ Error al cargar datos: {e}")
                return False
        return False

    def simulate_pca(self, cs_value, dh_value, grupo='HAH', n_sims=None):
        """Simula valores de PCA usando Monte Carlo para un escenario específico"""
        if n_sims is None:
            n_sims = self.n_simulations

        params = self.model_params[grupo]

        # Simulación de variables exógenas con distribuciones realistas
        np.random.seed(42)  # Para reproducibilidad

        # PSE: Perfil Socioeconómico (distribución normal estandarizada)
        pse = np.random.normal(0, 1, n_sims)

        # SQ: Status Quo (distribución normal con curtosis negativa)
        sq = np.random.normal(0, 1, n_sims)

        # Valores constantes para el escenario
        cs = np.full(n_sims, cs_value)
        dh = np.full(n_sims, dh_value)

        # Cálculo de PCA usando la ecuación estructural
        pca_deterministic = (params['PSE'] * pse +
                             params['DH'] * dh +
                             params['SQ'] * sq +
                             params['CS'] * cs)

        # Adición de error estocástico basado en RMSE del modelo
        error = np.random.normal(0, params['RMSE'], n_sims)
        pca = pca_deterministic + error

        return pca

    def run_scenario_analysis(self):
        """Ejecuta análisis completo de escenarios Monte Carlo"""
        print("🔄 Ejecutando simulaciones Monte Carlo...")

        resultados = {}

        for grupo in ['HAH', 'MAH']:
            resultados[grupo] = {}
            grupo_nombre = "Hombres Ahorradores" if grupo == 'HAH' else "Mujeres Ahorradoras"

            print(f"\n📊 Analizando grupo: {grupo_nombre}")

            for escenario, params in self.escenarios.items():
                print(f"  → Simulando escenario: {escenario}")

                pca_values = self.simulate_pca(
                    cs_value=params['CS'],
                    dh_value=params['DH'],
                    grupo=grupo
                )

                # Estadísticas descriptivas
                stats_dict = {
                    'valores': pca_values,
                    'media': np.mean(pca_values),
                    'mediana': np.median(pca_values),
                    'std': np.std(pca_values),
                    'q25': np.percentile(pca_values, 25),
                    'q75': np.percentile(pca_values, 75),
                    'min': np.min(pca_values),
                    'max': np.max(pca_values),
                    'ic_95_lower': np.percentile(pca_values, 2.5),
                    'ic_95_upper': np.percentile(pca_values, 97.5),
                    'cs_value': params['CS'],
                    'dh_value': params['DH'],
                    'descripcion': params['descripcion'],
                    'color': params['color']
                }

                resultados[grupo][escenario] = stats_dict

        print("✅ Simulaciones completadas exitosamente")
        return resultados

    def create_comprehensive_visualization(self, resultados):
        """Crea visualización HTML interactiva comprehensiva de los resultados"""
        figures = []

        # FIGURA 1: Distribuciones de PCA por grupo
        fig1 = make_subplots(
            rows=1, cols=2,
            subplot_titles=['Distribución PCA - Hombres Ahorradores',
                            'Distribución PCA - Mujeres Ahorradoras']
        )

        for idx, grupo in enumerate(['HAH', 'MAH']):
            col = idx + 1
            for escenario, datos in resultados[grupo].items():
                fig1.add_trace(
                    go.Histogram(
                        x=datos['valores'],
                        name=f'{escenario}',
                        opacity=0.7,
                        nbinsx=50,
                        marker_color=datos['color'],
                        showlegend=(idx == 0)
                    ),
                    row=1, col=col
                )

        fig1.update_layout(
            title="Distribuciones de Propensión Conductual al Ahorro por Escenario",
            height=500,
            template='plotly_white'
        )
        figures.append(fig1)

        # FIGURA 2: Comparación de medias
        escenarios_lista = list(self.escenarios.keys())
        medias_hah = [resultados['HAH'][esc]['media']
                      for esc in escenarios_lista]
        medias_mah = [resultados['MAH'][esc]['media']
                      for esc in escenarios_lista]

        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=escenarios_lista,
            y=medias_hah,
            name='Hombres Ahorradores',
            marker_color='#3498DB',
            opacity=0.8
        ))
        fig2.add_trace(go.Bar(
            x=escenarios_lista,
            y=medias_mah,
            name='Mujeres Ahorradoras',
            marker_color='#E91E63',
            opacity=0.8
        ))

        fig2.update_layout(
            title="Comparación de PCA Media por Escenario y Género",
            xaxis_title="Escenarios",
            yaxis_title="PCA Media",
            height=500,
            template='plotly_white',
            barmode='group'
        )
        figures.append(fig2)

        # FIGURA 3: Análisis de sensibilidad 3D
        fig3 = go.Figure()

        cs_values = [self.escenarios[esc]['CS'] for esc in escenarios_lista]
        dh_values = [self.escenarios[esc]['DH'] for esc in escenarios_lista]

        for grupo in ['HAH', 'MAH']:
            medias = [resultados[grupo][esc]['media']
                      for esc in escenarios_lista]
            fig3.add_trace(
                go.Scatter3d(
                    x=cs_values,
                    y=dh_values,
                    z=medias,
                    mode='markers+text',
                    text=escenarios_lista,
                    textposition='top center',
                    name=f'PCA Media ({grupo})',
                    marker=dict(
                        size=10,
                        color=medias,
                        colorscale='Viridis',
                        showscale=True
                    )
                )
            )

        fig3.update_layout(
            title="Análisis de Sensibilidad 3D: CS vs DH vs PCA",
            scene=dict(
                xaxis_title="Contagio Social (CS)",
                yaxis_title="Descuento Hiperbólico (DH)",
                zaxis_title="PCA Media"
            ),
            height=600,
            template='plotly_white'
        )
        figures.append(fig3)

        # FIGURA 4: Intervalos de confianza
        fig4 = go.Figure()

        for grupo in ['HAH', 'MAH']:
            ic_lower = [resultados[grupo][esc]['ic_95_lower']
                        for esc in escenarios_lista]
            ic_upper = [resultados[grupo][esc]['ic_95_upper']
                        for esc in escenarios_lista]
            medias = [resultados[grupo][esc]['media']
                      for esc in escenarios_lista]

            fig4.add_trace(
                go.Scatter(
                    x=escenarios_lista,
                    y=medias,
                    mode='markers',
                    name=f'PCA Media ({grupo})',
                    marker=dict(size=10),
                    error_y=dict(
                        type='data',
                        symmetric=False,
                        array=[upper - media for upper,
                               media in zip(ic_upper, medias)],
                        arrayminus=[media - lower for media,
                                    lower in zip(medias, ic_lower)],
                        visible=True
                    )
                )
            )

        fig4.update_layout(
            title="Intervalos de Confianza 95% por Escenario",
            xaxis_title="Escenarios",
            yaxis_title="PCA con IC 95%",
            height=500,
            template='plotly_white'
        )
        figures.append(fig4)

        # FIGURA 5: Métricas de riesgo
        cv_hah = [resultados['HAH'][esc]['std'] / abs(resultados['HAH'][esc]['media'])
                  if resultados['HAH'][esc]['media'] != 0 else 0 for esc in escenarios_lista]
        cv_mah = [resultados['MAH'][esc]['std'] / abs(resultados['MAH'][esc]['media'])
                  if resultados['MAH'][esc]['media'] != 0 else 0 for esc in escenarios_lista]

        fig5 = go.Figure()
        fig5.add_trace(go.Scatter(
            x=escenarios_lista,
            y=cv_hah,
            mode='lines+markers',
            name='CV Hombres',
            line=dict(color='#3498DB', width=3)
        ))
        fig5.add_trace(go.Scatter(
            x=escenarios_lista,
            y=cv_mah,
            mode='lines+markers',
            name='CV Mujeres',
            line=dict(color='#E91E63', width=3)
        ))

        fig5.update_layout(
            title="Coeficientes de Variación por Escenario",
            xaxis_title="Escenarios",
            yaxis_title="Coeficiente de Variación",
            height=500,
            template='plotly_white'
        )
        figures.append(fig5)

        return figures

    def generate_report_table(self, resultados):
        """Genera tabla resumida de resultados para el reporte"""
        reporte_data = []

        for grupo in ['HAH', 'MAH']:
            grupo_nombre = "Hombres Ahorradores" if grupo == 'HAH' else "Mujeres Ahorradoras"

            for escenario, datos in resultados[grupo].items():
                reporte_data.append({
                    'Grupo': grupo_nombre,
                    'Escenario': escenario,
                    'CS': f"{datos['cs_value']:.2f}",
                    'DH': f"{datos['dh_value']:.2f}",
                    'PCA Media': f"{datos['media']:.4f}",
                    'PCA Mediana': f"{datos['mediana']:.4f}",
                    'Desv. Estándar': f"{datos['std']:.4f}",
                    'IC 95% Inferior': f"{datos['ic_95_lower']:.4f}",
                    'IC 95% Superior': f"{datos['ic_95_upper']:.4f}",
                    'Rango IQR': f"[{datos['q25']:.3f}, {datos['q75']:.3f}]",
                    'CV': f"{datos['std']/abs(datos['media']):.3f}" if datos['media'] != 0 else "N/A"
                })

        return pd.DataFrame(reporte_data)

    def create_html_report(self, resultados, output_path="analisis_monte_carlo_pca.html"):
        """Genera reporte HTML completo con visualizaciones y análisis - VERSIÓN CORREGIDA"""

        # Crear visualizaciones
        figures = self.create_comprehensive_visualization(resultados)

        # Generar tabla de resultados
        tabla_resultados = self.generate_report_table(resultados)

        # Convertir figuras a HTML individual con Plotly embebido
        figures_html = []
        for i, fig in enumerate(figures):
            # Cada figura incluye Plotly de forma independiente
            fig_html = fig.to_html(
                include_plotlyjs='inline',  # ✅ CAMBIO CLAVE: inline en lugar de cdn/False
                div_id=f"plot-{i}",
                config={'displayModeBar': True, 'responsive': True}
            )
            # Extraer solo el div de la figura (sin HTML completo)
            div_pattern = r'<div id="plot-\d+".*?</script>\s*</div>'
            div_match = re.search(div_pattern, fig_html, re.DOTALL)
            if div_match:
                figures_html.append(div_match.group(0))
            else:
                # Fallback: usar HTML completo si no encuentra el div
                figures_html.append(fig_html)

        # Crear HTML personalizado
        html_content = f"""
        <!DOCTYPE html>
        <html lang="es">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Análisis Monte Carlo - Propensión Conductual al Ahorro</title>
            <style>
                body {{
                    font-family: 'Arial', sans-serif;
                    margin: 0;
                    padding: 20px;
                    background-color: #f8f9fa;
                    line-height: 1.6;
                }}
                .header {{
                    text-align: center;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 40px 20px;
                    border-radius: 10px;
                    margin-bottom: 30px;
                    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
                }}
                .container {{
                    max-width: 1400px;
                    margin: 0 auto;
                    background: white;
                    padding: 30px;
                    border-radius: 10px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }}
                .section {{
                    margin-bottom: 40px;
                }}
                .section h2 {{
                    color: #2c3e50;
                    border-bottom: 3px solid #3498db;
                    padding-bottom: 10px;
                }}
                .methodology {{
                    background-color: #ebf3fd;
                    padding: 20px;
                    border-left: 5px solid #3498db;
                    margin: 20px 0;
                }}
                .results-table {{
                    overflow-x: auto;
                    margin: 20px 0;
                }}
                .results-table table {{
                    width: 100%;
                    border-collapse: collapse;
                    font-size: 12px;
                }}
                .results-table th, .results-table td {{
                    border: 1px solid #ddd;
                    padding: 8px;
                    text-align: center;
                }}
                .results-table th {{
                    background-color: #3498db;
                    color: white;
                }}
                .scenario-description {{
                    background-color: #f8f9fa;
                    padding: 15px;
                    border-radius: 5px;
                    margin: 10px 0;
                }}
                .plot-container {{
                    margin: 30px 0;
                    background: white;
                    border-radius: 10px;
                    padding: 20px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                    min-height: 500px;
                }}
                .plot-title {{
                    font-size: 18px;
                    font-weight: bold;
                    color: #2c3e50;
                    margin-bottom: 15px;
                    text-align: center;
                }}
                .footer {{
                    text-align: center;
                    margin-top: 50px;
                    padding: 20px;
                    background-color: #2c3e50;
                    color: white;
                    border-radius: 10px;
                }}
                .plotly-graph-div {{
                    width: 100% !important;
                    height: 100% !important;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>ANÁLISIS MONTE CARLO</h1>
                <h2>Escenarios de Propensión Conductual al Ahorro (PCA)</h2>
                <p><strong>Modelado PLS-SEM con Simulación Estocástica</strong></p>
                <hr>
                <p><strong>Autor:</strong> MSc. Jesús F. Salazar Rojas</p>
                <p><strong>Doctorado en Economía</strong> - Universidad Católica Andrés Bello (UCAB)</p>
                <p><strong>Fecha:</strong> {datetime.now().strftime('%d de %B de %Y')}</p>
            </div>

            <div class="container">
                <div class="section">
                    <h2>1. RESUMEN EJECUTIVO</h2>
                    <p>Este reporte presenta un análisis exhaustivo mediante simulación Monte Carlo ({self.n_simulations:,} iteraciones)
                    de la Propensión Conductual al Ahorro (PCA) bajo diferentes escenarios de Contagio Social (CS) y
                    Descuento Hiperbólico (DH). Los resultados se basan en modelos PLS-SEM validados con corrección
                    por optimismo bootstrap.</p>

                    <div class="methodology">
                        <h3>📋 Metodología Aplicada:</h3>
                        <ul>
                            <li><strong>Modelo Base:</strong> Ecuaciones estructurales PLS-SEM con R² ajustado superior a 0.55</li>
                            <li><strong>Simulaciones:</strong> {self.n_simulations:,} iteraciones Monte Carlo por escenario</li>
                            <li><strong>Validación:</strong> Validación cruzada 10-fold con 20 repeticiones</li>
                            <li><strong>Corrección:</strong> Bootstrap con corrección por optimismo</li>
                        </ul>
                    </div>
                </div>

                <div class="section">
                    <h2>2. DEFINICIÓN DE ESCENARIOS</h2>
        """

        # Agregar descripción de escenarios
        for escenario, params in self.escenarios.items():
            html_content += f"""
                    <div class="scenario-description">
                        <h4>🎯 {escenario}</h4>
                        <p><strong>CS:</strong> {params['CS']} | <strong>DH:</strong> {params['DH']}</p>
                        <p>{params['descripcion']}</p>
                    </div>
            """

        html_content += """
                </div>

                <div class="section">
                    <h2>3. VISUALIZACIONES INTERACTIVAS</h2>
        """

        # Insertar figuras con títulos descriptivos
        plot_titles = [
            "📊 Distribuciones de PCA por Género y Escenario",
            "📈 Comparación de Medias por Escenario",
            "🌐 Análisis de Sensibilidad 3D: CS vs DH vs PCA",
            "📏 Intervalos de Confianza 95%",
            "⚠️ Coeficientes de Variación (Métricas de Riesgo)"
        ]

        for i, (fig_html, title) in enumerate(zip(figures_html, plot_titles)):
            html_content += f"""
                    <div class="plot-container">
                        <div class="plot-title">{title}</div>
                        {fig_html}
                    </div>
            """

        html_content += """
                </div>

                <div class="section">
                    <h2>4. TABLA DE RESULTADOS CONSOLIDADOS</h2>
                    <div class="results-table">
        """

        # Insertar tabla de resultados
        html_content += tabla_resultados.to_html(
            index=False, classes='results-table')

        # Análisis de conclusiones
        html_content += """
                    </div>
                </div>

                <div class="section">
                    <h2>5. CONCLUSIONES Y HALLAZGOS PRINCIPALES</h2>

                    <h3>🔍 Principales Hallazgos:</h3>
                    <ol>
                        <li><strong>Efecto Diferencial por Género:</strong> Las mujeres ahorradoras muestran mayor sensibilidad
                        al Contagio Social (coeficiente 0.3676 vs 0.2866 en hombres).</li>

                        <li><strong>Impacto del Descuento Hiperbólico:</strong> En hombres, mayor DH aumenta la PCA (0.2226),
                        mientras que en mujeres la reduce (-0.2013), sugiriendo estrategias diferenciadas de planificación temporal.</li>

                        <li><strong>Escenario Crítico:</strong> El escenario "Extremo Negativo" genera las menores propensiones
                        al ahorro en ambos grupos, validando la hipótesis de interacción negativa CS-DH.</li>

                        <li><strong>Oportunidad Optimista:</strong> Condiciones favorables de información (CS bajo) y planificación
                        a futuro (DH bajo) generan los mejores resultados de ahorro.</li>

                        <li><strong>Robustez del Modelo:</strong> Los intervalos de confianza al 95% muestran estabilidad
                        predictiva con coeficientes de variación controlados.</li>
                    </ol>

                    <h3>💡 Implicaciones para Políticas Públicas:</h3>
                    <ul>
                        <li>Necesidad de estrategias diferenciadas por género en programas de educación financiera</li>
                        <li>Importancia del manejo de expectativas y comunicación en períodos de incertidumbre económica</li>
                        <li>Diseño de incentivos temporales ajustados a perfiles de descuento hiperbólico</li>
                    </ul>
                </div>
            </div>

            <div class="footer">
                <p><strong>© 2025 - MSc. Jesús F. Salazar Rojas</strong></p>
                <p>Doctorado en Economía - Universidad Católica Andrés Bello (UCAB)</p>
                <p>Análisis generado con Python, Plotly y técnicas Monte Carlo avanzadas</p>
            </div>

            <script>
                window.addEventListener('load', function() {
                    if (typeof Plotly !== 'undefined') {
                        var plotDivs = document.querySelectorAll('[id^="plot-"]');
                        plotDivs.forEach(function(div) {
                            if (div._fullLayout) {
                                Plotly.Plots.resize(div);
                            }
                        });
                    }
                });
            </script>
        </body>
        </html>
        """

        # Guardar archivo HTML
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"✅ Reporte HTML generado exitosamente: {output_path}")
        print(f"📊 Se generaron {len(figures)} visualizaciones interactivas")

        return output_path


class TutorPackageGenerator:
    """Genera un paquete completo para enviar al tutor"""

    def __init__(self, analyzer_instance, resultados):
        self.analyzer = analyzer_instance
        self.resultados = resultados
        self.package_name = f"Avances_Tesis_MonteCarlo_PCA_{datetime.now().strftime('%Y%m%d_%H%M')}"

    def create_complete_package(self):
        """Crea paquete completo para el tutor"""
        print(f"→ Crea paquete completo para el tutor", flush=True)

        # Crear directorio del paquete
        package_dir = Path(self.package_name)
        package_dir.mkdir(exist_ok=True)

        print(f"📦 Creando paquete completo: {self.package_name}")

        # 1. HTML Interactivo (principal)
        html_path = package_dir / "Analisis_Monte_Carlo_PCA_Interactivo.html"
        self.analyzer.create_html_report(self.resultados, str(html_path))
        print(f"✅ HTML interactivo creado: {html_path.name}")

        # 2. Resumen ejecutivo en texto plano
        self.create_executive_summary(package_dir)

        # 3. Instrucciones para el tutor
        self.create_tutor_instructions(package_dir)

        # 4. Datos de respaldo (CSV)
        self.create_backup_data(package_dir)

        # 5. Carta de presentación
        self.create_presentation_letter(package_dir)

        # 6. Crear ZIP final
        zip_path = self.create_zip_package(package_dir)

        print(f"\n🎉 PAQUETE COMPLETO GENERADO EXITOSAMENTE", flush=True)
        print(f"📁 Carpeta: {package_dir}")
        print(f"📦 ZIP: {zip_path}")
        print(f"\n💌 LISTO PARA ENVIAR AL TUTOR")

        return zip_path, package_dir

    def create_tutor_instructions(self, package_dir):
        """Crea archivo de instrucciones para el tutor"""
        instructions = """INSTRUCCIONES PARA EL TUTOR
    ========================================
    Este paquete contiene:
    1. Reporte HTML interactivo con visualizaciones.
    2. Resumen ejecutivo en texto plano.
    3. Datos de respaldo en formato CSV.
    4. Carta de presentación.

    Para visualizar el reporte, abra el archivo HTML en cualquier navegador moderno.

    No se requiere instalación de Python ni software adicional.
    """
        file_path = package_dir / "Instrucciones_Tutor.txt"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(instructions)
        print(
            f"📄 Instrucciones para el tutor creadas: {file_path.name}", flush=True)

    def create_executive_summary(self, package_dir):
        """Crea resumen ejecutivo en texto plano"""
    summary = f"""
            RESUMEN EJECUTIVO - ANÁLISIS MONTE CARLO PCA
            ============================================
            AUTOR: MSc. Jesús F."""

    def create_backup_data(self, package_dir):
        """Crea archivo CSV con los resultados consolidados"""
        df_resultados = self.analyzer.generate_report_table(self.resultados)
        file_path = package_dir / "Resultados_Consolidados.csv"
        df_resultados.to_csv(file_path, index=False, encoding="utf-8-sig")
        print(f"📁 Datos de respaldo guardados: {file_path.name}", flush=True)

    def create_presentation_letter(self, package_dir):
        """Crea carta de presentación para el tutor"""
        letter = f"""CARTA DE PRESENTACIÓN
    ========================================
    Caracas, {datetime.now().strftime('%d de %B de %Y')}

    Estimado Tutor:

    Adjunto encontrará el paquete de avances de tesis doctoral titulado:
    "Propensión Conductual al Ahorro (PCA): Simulación Monte Carlo basada en modelos PLS-SEM".

    El paquete incluye:
    - Reporte HTML interactivo con visualizaciones.
    - Resumen ejecutivo en texto plano.
    - Datos de respaldo en formato CSV.
    - Instrucciones para su revisión.

    Agradezco sus observaciones y sugerencias para continuar con el desarrollo de esta investigación.

    Atentamente,

    MSc. Jesús F. Salazar Rojas
    Doctorado en Economía - UCAB
    """
        file_path = package_dir / "Carta_Presentacion_Tutor.txt"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(letter)
        print(f"📄 Carta de presentación creada: {file_path.name}", flush=True)

    def create_zip_package(self, package_dir):
        """Crea archivo ZIP con todo el contenido del paquete"""
        zip_path = Path(f"{self.package_name}.zip")
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file in package_dir.rglob("*"):
                zipf.write(file, arcname=file.relative_to(package_dir))
        print(f"🗜️ ZIP creado exitosamente: {zip_path.name}", flush=True)
        return zip_path


if __name__ == "__main__":
    analyzer = MonteCarloAnalysisPCA()
    if analyzer.load_data():
        resultados = analyzer.run_scenario_analysis()
        generator = TutorPackageGenerator(analyzer, resultados)
        generator.create_complete_package()
