# tablas.py
import pandas as pd
import os
from utilitis import crear_ruta_salida

DATA_FILES = {
    'HOMBRES': r"C:/.../DATA_CONSOLIDADA HOMBRES promedio H M .xlsx",
    'MUJERES': r"C:/.../DATA_CONSOLIDADA MUJERES promedio H M .xlsx"
}

DESC_FILES = {
    'HOMBRES': r"C:/.../descripiva HOMBRES ahorradores.xlsx",
    'MUJERES': r"C:/.../descripiva MUJERES ahorradores.xlsx"
}


def cargar_datos_excel(grupo: str, tipo: str = 'data') -> pd.DataFrame:
    """Carga datos de Excel según grupo y tipo."""
    if tipo == 'data':
        ruta = DATA_FILES[grupo]
    elif tipo == 'desc':
        ruta = DESC_FILES[grupo]
    else:
        raise ValueError("Tipo debe ser 'data' o 'desc'")

    df = pd.read_excel(ruta)
    print(f"{tipo.upper()} de {grupo} cargados. Filas: {len(df)}")
    return df


def exportar_tabla(df: pd.DataFrame, ruta_carpeta: str, nombre_archivo: str) -> str:
    """Exporta un DataFrame a Excel en la ruta especificada y devuelve la ruta completa."""
    # Crear carpeta si no existe
    ruta_salida = crear_ruta_salida(ruta_carpeta)

    # Ruta completa del archivo
    ruta_archivo = os.path.join(ruta_salida, nombre_archivo)

    # Guardar Excel
    df.to_excel(ruta_archivo, index=False)
    print(f"Archivo exportado en: {ruta_archivo}")
    return ruta_archivo


# --- Ejemplo de prueba ---
if __name__ == "__main__":
    df_test = cargar_datos_excel('HOMBRES', 'data')
    exportar_tabla(df_test, "./resultados/HOMBRES", "HOMBRES_resultados.xlsx")
