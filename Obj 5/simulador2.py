# sim_pca_montecarlo_app.py
# -----------------------------------------------------------
# Dashboard de simulación Monte Carlo para PCA (PLS-SEM)
# - Lee data original (scores y variables socioeconómicas)
# - Ejecuta escenarios de simulación (base, mejora, envejecimiento, social, intervención)
# - Usa ecuaciones estructurales validadas (hombres/mujeres)
# - Valida supuestos de regresión (OLS auxiliar) y reporta ✅/⚠️/❌
# - Exporta todo a Excel (un archivo por corrida con timestamp)
# -----------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from io import BytesIO

from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan, normal_ad
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.outliers_influence import variance_inflation_factor

# -------------------------
# CONFIGURACIÓN INICIAL
# -------------------------
st.set_page_config(
    page_title="Simulador PCA (PLS-SEM) • Monte Carlo", layout="wide")

st.title("Simulador PCA (PLS-SEM) con Monte Carlo")
st.markdown(
    "Simula escenarios sobre la **Propensión Conductual al Ahorro (PCA)** usando las ecuaciones estructurales validadas."
)

# Tarjeta de presentación
st.markdown("""
---
**MSc Jesús F. Salazar Rojas**  •  **Doctorando**  
**Tesis:** *La Propensión Conductual al Ahorro: Un estudio desde los sesgos cognitivos para la toma de decisiones en el ahorro de los hogares*
---
""")

# -------------------------
# ENTRADA DE DATOS
# -------------------------
st.sidebar.header("1) Datos de entrada")

default_path = r"C:\01 academico\001 Doctorado Economia UCAB\d tesis problema ahorro\01 TESIS DEFINITIVA\MODELO\score\data y scores Hah y Mah.xlsx"
data_path = st.sidebar.text_input(
    "Ruta del archivo Excel (scores + socioeconómicas):", value=default_path)

st.sidebar.markdown("**Columnas esperadas** en el Excel:")
st.sidebar.code(
    "Case, PCA, PSE, SQ, DH, CS, AV, GRUPO, Item, PCA1, PCA2, PCA4, PCA5, PCA6, PCA7")

load_btn = st.sidebar.button("Cargar datos")


@st.cache_data(show_spinner=True)
def load_data(path):
    df = pd.read_excel(path)
    return df


df = None
if load_btn:
    try:
        df = load_data(data_path)
        st.success(f"Archivo cargado: {data_path}")
        st.write("Vista previa:", df.head())
    except Exception as e:
        st.error(f"No se pudo cargar el archivo. Detalle: {e}")

# -------------------------
# PARÁMETROS GENERALES
# -------------------------
st.sidebar.header("2) Parámetros de simulación")
grupo = st.sidebar.selectbox(
    "Selecciona el grupo a simular", ["Hombres", "Mujeres"])
n_boot = st.sidebar.number_input(
    "Tamaño de muestra por escenario (bootstrap)", min_value=50, max_value=10000, value=231, step=1)
n_mc = st.sidebar.number_input("N° iteraciones Monte Carlo por escenario",
                               min_value=100, max_value=20000, value=2000, step=100)
seed = st.sidebar.number_input(
    "Semilla (reproducibilidad)", min_value=0, max_value=10_000_000, value=1234, step=1)

np.random.seed(seed)

# -------------------------
# ECUACIONES PLS-SEM
# -------------------------
# Hombres:
coef_h = {
    "PCA": {"PSE": 0.3777, "SQ": -0.5947, "DH": 0.2226, "CS": 0.2866},
    "SQ": {"AV": 0.8402}
}
# Mujeres:
coef_m = {
    "PCA": {"PSE": 0.3485, "SQ": -0.5101, "DH": -0.2013, "CS": 0.3676},
    "SQ": {"AV": 0.6609}
}


def get_coefs(grupo_sel):
    return coef_h if grupo_sel == "Hombres" else coef_m


def compute_sq(av, coefs):
    return coefs["SQ"]["AV"] * av


def compute_pca(pse, sq, dh, cs, coefs):
    b = coefs["PCA"]
    return b["PSE"]*pse + b["SQ"]*sq + b["DH"]*dh + b["CS"]*cs

# -------------------------
# UTILIDADES
# -------------------------


def bootstrap_sample(df_g, size):
    idx = np.random.choice(df_g.index, size=size, replace=True)
    return df_g.loc[idx].copy()


def inc_variance(series, factor=1.5):
    """Aumenta varianza reescalando desviaciones respecto a la media."""
    mu = series.mean()
    return mu + factor*(series - mu)


def ols_diagnostics(y, X):
    """Ajusta OLS y calcula diagnósticos de supuestos (resumen compacto)."""
    Xc = sm.add_constant(X, has_constant='add')
    model = sm.OLS(y, Xc).fit()
    resid = model.resid
    # Durbin-Watson (independencia)
    dw = durbin_watson(resid)
    # Breusch-Pagan (homocedasticidad)
    bp_stat, bp_p, _, _ = het_breuschpagan(resid, Xc)
    # Normalidad (Anderson-Darling versión normal_ad; también se podría usar Shapiro para n<5000)
    ad_stat, ad_p = normal_ad(resid)
    # VIF (multicolinealidad)
    vif_data = []
    for i in range(1, Xc.shape[1]):  # ignorar constante
        vif = variance_inflation_factor(Xc.values, i)
        vif_data.append(vif)
    vif_max = np.nanmax(vif_data) if len(vif_data) > 0 else np.nan

    # Semáforos
    flag_dw = "✅" if 1.5 <= dw <= 2.5 else "⚠️"
    flag_bp = "✅" if bp_p > 0.05 else "❌"
    flag_ad = "✅" if ad_p > 0.05 else "❌"
    flag_vif = "✅" if (not np.isnan(vif_max)) and (vif_max < 5) else (
        "⚠️" if (not np.isnan(vif_max)) and (vif_max < 10) else "❌")

    summary = {
        "R2_OLS": model.rsquared,
        "DW": dw, "DW_flag": flag_dw,
        "BP_p": bp_p, "BP_flag": flag_bp,
        "AD_p": ad_p, "AD_flag": flag_ad,
        "VIF_max": vif_max, "VIF_flag": flag_vif
    }
    return model, summary, resid


def scenario_metrics(pca_vals):
    arr = np.asarray(pca_vals)
    return {
        "mean": np.mean(arr),
        "std": np.std(arr, ddof=1),
        "p10": np.percentile(arr, 10),
        "p50": np.percentile(arr, 50),
        "p90": np.percentile(arr, 90),
    }


def run_scenario(df_g, coefs, scen_name, n_boot, n_mc,
                 tweak_fn=None, note=""):
    """
    Ejecuta un escenario:
    - bootstrap de n_boot filas,
    - N iteraciones MC variando inputs (ruido leve),
    - aplica tweak_fn para modificar inputs según el escenario,
    - calcula SQ:=b*AV y PCA:=f(PSE, SQ, DH, CS),
    - retorna dataframe con resultados y métricas.
    """
    # Muestra bootstrap
    df_b = bootstrap_sample(df_g, n_boot)

    # Si hay ajustes específicos del escenario a nivel "muestra base"
    if tweak_fn:
        df_b = tweak_fn(df_b.copy())

    # Monte Carlo: aquí añadimos pequeñas perturbaciones normales (5%) a PSE, DH, CS, AV
    # para simular incertidumbre paramétrica alrededor de la muestra bootstrap.
    def jitter(s):
        s0 = s.fillna(0)
        scale = 0.05 * (s0.std(ddof=1) if s0.std(ddof=1) > 0 else 1.0)
        return s0 + np.random.normal(0, scale, size=len(s0))

    pca_list, sq_list = [], []
    X_store = []

    for _ in range(n_mc):
        # Perturbar entradas
        pse = jitter(df_b["PSE"])
        dh = jitter(df_b["DH"])
        cs = jitter(df_b["CS"])
        av = jitter(df_b["AV"])

        # SQ desde AV (estructura PLS-SEM)
        sq = compute_sq(av, coefs)

        # Ajustes adicionales por escenario (si se definieron como función sobre series)
        # (NO se usa aquí; el tweak_fn se aplicó a df_b)
        pca = compute_pca(pse, sq, dh, cs, coefs)

        pca_list.append(pca)
        sq_list.append(sq)
        X_store.append(pd.DataFrame(
            {"PSE": pse, "DH": dh, "CS": cs, "SQ": sq}))

    # Unir resultados
    df_pca = pd.DataFrame({"PCA": np.concatenate(pca_list)})
    df_sq = pd.DataFrame({"SQ":  np.concatenate(sq_list)})
    X_all = pd.concat(X_store, ignore_index=True)

    # Diagnósticos OLS auxiliar (PCA ~ PSE + DH + CS + SQ)
    _, diag, _ = ols_diagnostics(
        df_pca["PCA"].values, X_all[["PSE", "DH", "CS", "SQ"]])

    # Métricas agregadas de PCA
    mets = scenario_metrics(df_pca["PCA"].values)

    # Resumen
    info = {
        "escenario": scen_name,
        "n_boot": n_boot,
        "n_mc": n_mc,
        "nota": note,
        **{f"PCA_{k}": v for k, v in mets.items()},
        **diag
    }
    out_df = pd.concat([df_pca, df_sq, X_all], axis=1)

    return info, out_df


# -------------------------
# DEFINICIÓN DE ESCENARIOS
# -------------------------
st.sidebar.header("3) Escenarios (se aplican a la muestra bootstrap)")
st.sidebar.caption(
    "Los ajustes actúan sobre columnas PSE, AV, CS, etc. (inputs del modelo).")

# Parámetros de intervención
inc_ingreso_pct = st.sidebar.slider(
    "Mejora socioeconómica: ∆PSE (%)", 0, 100, 20, 5)
envej_anios = st.sidebar.slider("Envejecimiento: +años (proxy)", 0, 20, 10, 1)
cs_var_factor = st.sidebar.slider(
    "Mayor influencia social: factor de varianza CS", 1.0, 5.0, 2.0, 0.1)
sq_shock_pct = st.sidebar.slider(
    "Intervención conductual: reducción SQ (%)", 0, 90, 30, 5)

# Notas:
# - Mejora socioeconómica: elevamos PSE en +inc_ingreso_pct %
# - Envejecimiento: se aplica un pequeño aumento de PSE como proxy (ajuste editable); p.ej. +0.1 SD por 10 años
# - Mayor influencia social: aumentar varianza de CS
# - Intervención conductual: reducimos SQ en un % (implementado como multiplicador sobre el resultado de SQ)


def scen_mejora(df_b):
    df_b = df_b.copy()
    df_b["PSE"] = df_b["PSE"] * (1 + inc_ingreso_pct/100.0)
    # (Opcional) si quieres reflejar más educación: podrías sumar +1 a PCA4 y recalcular PSE si tu pipeline lo hace.
    return df_b


def scen_envejec(df_b):
    df_b = df_b.copy()
    # Proxy simple: por +10 años, +0.10*SD(PSE)
    bump = (envej_anios / 10.0) * 0.10 * df_b["PSE"].std(ddof=1)
    df_b["PSE"] = df_b["PSE"] + bump
    return df_b


def scen_social(df_b):
    df_b = df_b.copy()
    df_b["CS"] = inc_variance(df_b["CS"], factor=cs_var_factor)
    return df_b


def scen_interv(df_b):
    # Aplicaremos el shock sobre SQ *después* de calcular desde AV.
    # Para mantener el framework (SQ desde AV), implementaremos el shock dentro de run_scenario
    # modificando PSE/DH/CS/AV no es necesario; así que aquí no cambiamos df_b.
    return df_b

# Para el shock de SQ, modificaremos compute_sq al vuelo con un wrapper:


def make_sq_func(coefs, shock_pct=None):
    def _f(av):
        sq_clean = compute_sq(av, coefs)
        if (shock_pct is not None) and (shock_pct > 0):
            return (1 - shock_pct/100.0) * sq_clean
        return sq_clean
    return _f


# -------------------------
# EJECUCIÓN
# -------------------------
if df is not None:
    # Filtrar por grupo
    df_g = df[df["GRUPO"].str.lower().isin(
        [grupo.lower(), grupo[0].lower()]) | (df["GRUPO"] == grupo)].copy()
    if df_g.empty:
        st.warning(
            "No se encontraron filas para el grupo seleccionado. Revisa la columna 'GRUPO'.")
    else:
        st.success(f"Observaciones disponibles para {grupo}: {len(df_g)}")
        coefs = get_coefs(grupo)

        # Escenario Base (empírico): sin tweaks
        with st.spinner("Simulando escenario base…"):
            base_info, base_df = run_scenario(df_g, coefs, "Base (Empírico)", n_boot, n_mc, tweak_fn=None,
                                              note="Bootstrap sobre muestra original; jitter 5% en inputs.")

        # Escenario Mejora Socioeconómica
        with st.spinner("Simulando mejora socioeconómica…"):
            mej_info, mej_df = run_scenario(df_g, coefs, "Mejora Socioeconómica", n_boot, n_mc, tweak_fn=scen_mejora,
                                            note=f"PSE aumentado en {inc_ingreso_pct}%.")

        # Escenario Envejecimiento
        with st.spinner("Simulando envejecimiento de la población…"):
            env_info, env_df = run_scenario(df_g, coefs, "Envejecimiento", n_boot, n_mc, tweak_fn=scen_envejec,
                                            note=f"PSE incrementado proxy por +{envej_anios} años.")

        # Escenario Mayor Influencia Social
        with st.spinner("Simulando mayor influencia social…"):
            soc_info, soc_df = run_scenario(df_g, coefs, "Mayor Influencia Social", n_boot, n_mc, tweak_fn=scen_social,
                                            note=f"Varianza de CS multiplicada por {cs_var_factor}.")

        # Escenario Intervención Conductual (shock a SQ)
        # Para aplicar el shock en SQ, volveremos a ejecutar run_scenario pero reemplazando compute_sq dentro
        # NOTA: para simplicidad, duplicamos la lógica de run_scenario con el shock aplicado:
        def run_scenario_interv(df_g, coefs, n_boot, n_mc, shock_pct, note):
            df_b = bootstrap_sample(df_g, n_boot)
            # Monte Carlo

            def jitter(s):
                s0 = s.fillna(0)
                scale = 0.05 * (s0.std(ddof=1) if s0.std(ddof=1) > 0 else 1.0)
                return s0 + np.random.normal(0, scale, size=len(s0))

            pca_list, sq_list = [], []
            X_store = []
            sq_fun = make_sq_func(coefs, shock_pct)

            for _ in range(n_mc):
                pse = jitter(df_b["PSE"])
                dh = jitter(df_b["DH"])
                cs = jitter(df_b["CS"])
                av = jitter(df_b["AV"])

                sq = sq_fun(av)  # shock aplicado aquí
                pca = compute_pca(pse, sq, dh, cs, coefs)

                pca_list.append(pca)
                sq_list.append(sq)
                X_store.append(pd.DataFrame(
                    {"PSE": pse, "DH": dh, "CS": cs, "SQ": sq}))

            df_pca = pd.DataFrame({"PCA": np.concatenate(pca_list)})
            df_sq = pd.DataFrame({"SQ":  np.concatenate(sq_list)})
            X_all = pd.concat(X_store, ignore_index=True)

            _, diag, _ = ols_diagnostics(
                df_pca["PCA"].values, X_all[["PSE", "DH", "CS", "SQ"]])
            mets = scenario_metrics(df_pca["PCA"].values)

            info = {
                "escenario": f"Intervención Conductual (−{shock_pct}% SQ)",
                "n_boot": n_boot,
                "n_mc": n_mc,
                "nota": note,
                **{f"PCA_{k}": v for k, v in mets.items()},
                **diag
            }
            out_df = pd.concat([df_pca, df_sq, X_all], axis=1)
            return info, out_df

        with st.spinner("Simulando intervención conductual (reducción SQ)…"):
            int_info, int_df = run_scenario_interv(
                df_g, coefs, n_boot, n_mc, sq_shock_pct,
                note=f"SQ reducido en {sq_shock_pct}% respecto al valor inducido por AV."
            )

        # -------------------------
        # MOSTRAR RESUMEN EN PANTALLA
        # -------------------------
        st.subheader("Resumen de métricas por escenario (PCA)")
        summary_df = pd.DataFrame(
            [base_info, mej_info, env_info, soc_info, int_info])
        cols_order = ["escenario", "nota", "n_boot", "n_mc",
                      "PCA_mean", "PCA_std", "PCA_p10", "PCA_p50", "PCA_p90",
                      "R2_OLS", "DW", "DW_flag", "BP_p", "BP_flag", "AD_p", "AD_flag", "VIF_max", "VIF_flag"]
        summary_df = summary_df[cols_order]
        st.dataframe(summary_df, use_container_width=True)

        # Semáforo global de supuestos (rápido):
        st.markdown("**Validación discreta de supuestos (por escenario):**")
        for _, r in summary_df.iterrows():
            st.write(
                f"- **{r['escenario']}** → "
                f"DW {r['DW']:.2f} {r['DW_flag']} | "
                f"BP p={r['BP_p']:.3f} {r['BP_flag']} | "
                f"Normalidad (AD) p={r['AD_p']:.3f} {r['AD_flag']} | "
                f"VIF max={r['VIF_max']:.2f} {r['VIF_flag']}"
            )

        # -------------------------
        # EXPORTAR A EXCEL
        # -------------------------
        st.subheader("Exportación a Excel")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"sim_pca_pls_montecarlo_{grupo}_{timestamp}.xlsx"

        output = BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            # Resumen
            summary_df.to_excel(writer, index=False, sheet_name="Resumen")

            # Detalle por escenario (muestra concatenada de MC)
            base_df.to_excel(writer, index=False, sheet_name="Base")
            mej_df.to_excel(writer, index=False, sheet_name="Mejora")
            env_df.to_excel(writer, index=False, sheet_name="Envejecimiento")
            soc_df.to_excel(writer, index=False, sheet_name="Social")
            int_df.to_excel(writer, index=False, sheet_name="Intervencion")

            # Metadatos (coeficientes)
            coef_df = pd.DataFrame({
                "Grupo": [grupo]*5,
                "Trayectoria": ["PSE→PCA", "SQ→PCA", "DH→PCA", "CS→PCA", "AV→SQ"],
                "Beta": [
                    coefs["PCA"]["PSE"],
                    coefs["PCA"]["SQ"],
                    coefs["PCA"]["DH"],
                    coefs["PCA"]["CS"],
                    coefs["SQ"]["AV"]
                ]
            })
            coef_df.to_excel(writer, index=False, sheet_name="Coeficientes")

            # Nota metodológica
            note_txt = pd.DataFrame({
                "Nota": [
                    "Simulaciones con bootstrap (muestra) + jitter normal 5% en inputs.",
                    "SQ se genera desde AV vía ecuación estructural: SQ = beta_AV→SQ * AV.",
                    "PCA = b1*PSE + b2*SQ + b3*DH + b4*CS (coef. por grupo).",
                    "Validación de supuestos: OLS auxiliar sobre datos simulados.",
                    "Señales: ✅ (aceptable) / ⚠️ (revisar) / ❌ (violación)."
                ]
            })
            note_txt.to_excel(writer, index=False, sheet_name="Notas")

        st.download_button(
            label="⬇️ Descargar Excel de resultados",
            data=output.getvalue(),
            file_name=filename,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        st.caption(
            "Cada hoja contiene el conjunto completo de observaciones simuladas por escenario.")
else:
    st.info("Carga el archivo Excel para habilitar las simulaciones.")
