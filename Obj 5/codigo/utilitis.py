# utilitis.py
import os


def crear_ruta_salida(ruta_carpeta: str, nombre_archivo: str) -> str:
    """Crea ruta completa para guardar archivos"""
    if not os.path.exists(ruta_carpeta):
        os.makedirs(ruta_carpeta)
    return os.path.join(ruta_carpeta, nombre_archivo)
