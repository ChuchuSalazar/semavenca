import os
from tablas import cargar_datos_excel, exportar_tabla
from graficos import graficar_dispersion, graficar_histograma
from analisis import calcular_predicciones, bootstrap_mc, PLSPredictAnalyzer


def main():
    analyzer = PLSPredictAnalyzer()

    # --- Ruta absoluta de resultados ---
    ruta_resultados = r"C:\01 academico\001 Doctorado Economia UCAB\d tesis problema ahorro\01 TESIS DEFINITIVA\MODELO\resultados obj5_1"
    os.makedirs(ruta_resultados, exist_ok=True)

    for grupo in ['HOMBRES', 'MUJERES']:
        print(f"\n{'='*80}\nANÁLISIS PARA GRUPO: {grupo}\n{'='*80}")

        # Nombre del archivo con grupo al final
        nombre_archivo = f"resultados_{grupo}.xlsx"

        df_data = cargar_datos_excel(grupo, tipo='data')
        df_desc = cargar_datos_excel(grupo, tipo='desc')

        # Renombrar columnas
        df_data.rename(columns={
            'PROM_AV': 'AV',
            'PROM_DH': 'DH',
            'PROM_SQ': 'SQ',
            'PROM_CS': 'CS'
        }, inplace=True)

        resultados = calcular_predicciones(df_data, grupo)
        bootstrap_mc(resultados, df_data, grupo)

        X, y, _, _ = analyzer.prepare_data(df_data)
        r2_corrected, ci = analyzer.bootstrap_prediction_intervals(
            X, y, n_bootstrap=1000)
        print(
            f"[{grupo}] R² corregido por bootstrap: {r2_corrected:.4f}, IC 95%: {ci}")

        # Exportar resultados usando la ruta absoluta
        exportar_tabla(resultados, ruta_resultados, nombre_archivo)

        # Graficar resultados
        graficar_dispersion(resultados, ruta_resultados, grupo)
        graficar_histograma(resultados, ruta_resultados, grupo)


if __name__ == "__main__":
    main()
