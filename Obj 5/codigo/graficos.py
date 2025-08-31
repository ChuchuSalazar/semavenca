import matplotlib.pyplot as plt
import os


def graficar_dispersion(df, ruta_salida, grupo):
    """Genera gráfico de dispersión entre observados y predichos"""
    plt.figure()
    plt.scatter(df['observados'], df['predichos'], alpha=0.7)
    plt.axhline(0, color="red", linestyle="--", linewidth=1)
    plt.axvline(0, color="red", linestyle="--", linewidth=1)
    mayores = sum(df['predichos'] > 0)
    menores = sum(df['predichos'] <= 0)
    plt.legend([f">0: {mayores}", f"≤0: {menores}"])
    plt.xlabel("Observados")
    plt.ylabel("Predichos")
    plt.title(f"Dispersión observados vs predichos - {grupo}")
    plt.savefig(os.path.join(ruta_salida, f"dispersion_{grupo}.png"))
    plt.close()


def graficar_histograma(df, ruta_salida, grupo):
    """Genera histograma de residuales"""
    plt.figure()
    df['residuales'].hist(bins=20, alpha=0.7)
    plt.xlabel("Residuales")
    plt.ylabel("Frecuencia")
    plt.title(f"Histograma de residuales - {grupo}")
    plt.savefig(os.path.join(
        ruta_salida, f"histograma_residuales_{grupo}.png"))
    plt.close()
