
import os
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')


class PLSSEMPredictAnalyzer:
    """
    Simulador de capacidad predictiva PLS-SEM estructural con:
      - Validación cruzada
      - Monte Carlo
      - Comparación vs regresión directa
      - Exportación a Excel (.xlsx)
      - Generación de reporte TXT
      - Visualizaciones PNG
    """

    def __init__(self, base_output_path: str):
        self.base_output_path = base_output_path
        os.makedirs(self.base_output_path, exist_ok=True)
        self.results = {}
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

    def _get_coeffs(self, group: str):
        return self.model_coefficients[group]

    def load_data(self, file_path: str) -> pd.DataFrame:
        try:
            df = pd.read_excel(file_path)
            print(f"Datos cargados ({file_path}): {df.shape}")
            return df
        except Exception as e:
            print(f"Error cargando datos: {e}")
            return None

    def prepare_structural_data(self, data: pd.DataFrame, group: str):
        required = ['PSE', 'DH', 'CS', 'AV', 'PCA']
        alternative_names = {
            'PSE': ['PSEP', 'PSE_PROMEDIO'],
            'DH': ['PROM_DH', 'DH_PROMEDIO'],
            'CS': ['PROM_CS', 'CS_PROMEDIO'],
            'AV': ['PROM_AV', 'AV_PROMEDIO'],
            'PCA': ['PPCA', 'PCA_PROMEDIO']
        }

        missing = set(required) - set(data.columns)
        for var in missing & set(alternative_names):
            for alt in alternative_names[var]:
                if alt in data.columns:
                    data[var] = data[alt]
                    print(f"Mapeado {alt} → {var}")
                    break

        try:
            X_int = data[['AV']].copy()
            X_str = data[['PSE', 'DH', 'CS']].copy()
            y_pca = data['PCA'].copy()
            y_sq = data['SQ'].copy() if 'SQ' in data.columns else None
        except KeyError as e:
            print(f"Variable no encontrada: {e}")
            return (None,)*6

        mask = ~(X_int.isnull().any(axis=1) |
                 X_str.isnull().any(axis=1) |
                 y_pca.isnull())
        if y_sq is not None:
            mask &= ~y_sq.isnull()

        X_int = X_int[mask]
        X_str = X_str[mask]
        y_pca = y_pca[mask]
        y_sq = y_sq[mask] if y_sq is not None else None

        print(f"{group}: {X_str.shape[0]} observaciones limpias")
        return X_int, X_str, y_pca, y_sq, ['AV'], ['PSE', 'DH', 'CS']

    def structural_cross_validation(self, X_int, X_str, y_pca, y_sq, group: str,
                                    n_splits=10, n_repeats=20):
        print(f"\n=== VALIDACIÓN CRUZADA - {group} ===")
        rkf = RepeatedKFold(n_splits=n_splits,
                            n_repeats=n_repeats,
                            random_state=42)
        coeffs = self._get_coeffs(group)
        cv_results = {
            'PCA_Structural': {'RMSE': [], 'MAE': [], 'R2': []},
            'SQ_Intermediate': {'RMSE': [], 'MAE': [], 'R2': []}
        }

        for train_idx, test_idx in rkf.split(X_str):
            Xi_te = X_int.iloc[test_idx]
            Xs_te = X_str.iloc[test_idx]
            y_te = y_pca.iloc[test_idx]
            sq_te = y_sq.iloc[test_idx] if y_sq is not None else None

            # SQ predicho
            sq_pred = coeffs['sq_equation']['AV'] * Xi_te['AV']
            if sq_te is not None:
                cv_results['SQ_Intermediate']['RMSE'].append(
                    np.sqrt(mean_squared_error(sq_te, sq_pred)))
                cv_results['SQ_Intermediate']['MAE'].append(
                    mean_absolute_error(sq_te, sq_pred))
                cv_results['SQ_Intermediate']['R2'].append(
                    r2_score(sq_te, sq_pred))

            # PCA predicho
            pca_pred = (
                coeffs['pca_equation']['PSE'] * Xs_te['PSE'] +
                coeffs['pca_equation']['DH'] * Xs_te['DH'] +
                coeffs['pca_equation']['CS'] * Xs_te['CS'] +
                coeffs['pca_equation']['SQ'] * sq_pred
            )
            cv_results['PCA_Structural']['RMSE'].append(
                np.sqrt(mean_squared_error(y_te, pca_pred)))
            cv_results['PCA_Structural']['MAE'].append(
                mean_absolute_error(y_te, pca_pred))
            cv_results['PCA_Structural']['R2'].append(
                r2_score(y_te, pca_pred))

        # Calcular medias y desviaciones
        for model in cv_results:
            for metric in ['RMSE', 'MAE', 'R2']:
                arr = np.array(cv_results[model][metric])
                cv_results[model][f"{metric}_mean"] = arr.mean()
                cv_results[model][f"{metric}_std"] = arr.std()

        key = f"structural_cv_{group}"
        self.results[key] = cv_results
        print(f"[DEBUG] CV guardado: {key}")
        return cv_results

    def monte_carlo_structural(self, X_int, X_str, group: str, n_simulations=5000):
        print(f"\n=== MONTE CARLO - {group} ({n_simulations}) ===")
        stats_int = X_int.describe()
        coeffs = self._get_coeffs(group)

        pca_preds, sq_preds = [], []
        scenarios = []

        np.random.seed(42)
        for _ in range(n_simulations):
            av_sim = np.random.normal(
                stats_int.loc['mean', 'AV'],
                stats_int.loc['std', 'AV'],
                size=len(X_int)
            )
            sq_sim = coeffs['sq_equation']['AV'] * av_sim
            pca_sim = (
                coeffs['pca_equation']['PSE'] * X_str['PSE'] +
                coeffs['pca_equation']['DH'] * X_str['DH'] +
                coeffs['pca_equation']['CS'] * X_str['CS'] +
                coeffs['pca_equation']['SQ'] * sq_sim
            )

            sq_preds.extend(sq_sim)
            pca_preds.extend(pca_sim)
            scenarios.append(
                {'AV_sim': av_sim, 'SQ_sim': sq_sim, 'PCA_sim': pca_sim})

        # DataFrame de escenarios
        scenarios_df = pd.DataFrame({
            k: np.concatenate([d[k] for d in scenarios])
            for k in scenarios[0]
        })

        percentiles = [10, 25, 50, 75, 90]
        pca_pct = dict(zip(percentiles, np.percentile(pca_preds, percentiles)))
        sq_pct = dict(zip(percentiles, np.percentile(sq_preds, percentiles)))

        key = f"monte_carlo_{group}"
        self.results[key] = {
            'scenarios': scenarios_df,
            'pca_predictions': np.array(pca_preds),
            'sq_predictions': np.array(sq_preds),
            'pca_percentiles': pca_pct,
            'sq_percentiles': sq_pct
        }
        print(f"[DEBUG] Monte Carlo guardado: {key}")
        return scenarios_df, pca_preds, sq_preds

    def compare_structural_vs_direct_regression(self, X_int, X_str, y_pca, group: str):
        print(f"\n=== COMPARACIÓN vs REGRESIÓN DIRECTA - {group} ===")
        Xc = pd.concat([X_int, X_str], axis=1)
        rkf = RepeatedKFold(n_splits=10, n_repeats=20, random_state=42)

        struct_r2, direct_r2 = [], []
        coeffs = self._get_coeffs(group)

        for tr, te in rkf.split(Xc):
            Xi_te = X_int.iloc[te]
            Xs_te = X_str.iloc[te]
            y_te = y_pca.iloc[te]

            # Estructural predicho
            sq_pred = coeffs['sq_equation']['AV'] * Xi_te['AV']
            pca_pred = (
                coeffs['pca_equation']['PSE'] * Xs_te['PSE'] +
                coeffs['pca_equation']['DH'] * Xs_te['DH'] +
                coeffs['pca_equation']['CS'] * Xs_te['CS'] +
                coeffs['pca_equation']['SQ'] * sq_pred
            )
            struct_r2.append(r2_score(y_te, pca_pred))

            # Regresión directa
            scaler = StandardScaler()
            Xc_tr = scaler.fit_transform(Xc.iloc[tr])
            Xc_te = scaler.transform(Xc.iloc[te])
            lr = LinearRegression().fit(Xc_tr, y_pca.iloc[tr])
            direct_r2.append(r2_score(y_te, lr.predict(Xc_te)))

        stat, p_val = ttest_rel(struct_r2, direct_r2)
        key = f"model_comparison_{group}"
        self.results[key] = {
            'structural_r2': np.array(struct_r2),
            'direct_r2': np.array(direct_r2),
            'p_value': p_val
        }
        print(f"[DEBUG] Comparación guardada: {key} (p={p_val:.4f})")
        return struct_r2, direct_r2, p_val

    # ================= Visualizaciones =================

    def plot_observed_vs_predicted(self, X_int, X_str, y_pca, y_sq, group: str):
        coeffs = self._get_coeffs(group)
        sq_pred = coeffs['sq_equation']['AV'] * X_int['AV']
        pca_pred = (
            coeffs['pca_equation']['PSE'] * X_str['PSE'] +
            coeffs['pca_equation']['DH'] * X_str['DH'] +
            coeffs['pca_equation']['CS'] * X_str['CS'] +
            coeffs['pca_equation']['SQ'] * sq_pred
        )

        fig, ax = plt.subplots(1, 2, figsize=(14, 5))
        if y_sq is not None:
            ax[0].scatter(y_sq, sq_pred, alpha=0.6, color='steelblue')
            ax[0].plot([y_sq.min(), y_sq.max()], [
                       y_sq.min(), y_sq.max()], 'r--')
            ax[0].set_title(
                f"SQ Obs vs Pred\nR²={r2_score(y_sq, sq_pred):.3f}")
        ax[1].scatter(y_pca, pca_pred, alpha=0.6, color='darkgreen')
        ax[1].plot([y_pca.min(), y_pca.max()], [
                   y_pca.min(), y_pca.max()], 'r--')
        ax[1].set_title(f"PCA Obs vs Pred\nR²={r2_score(y_pca, pca_pred):.3f}")

        filename = f"{group}_observed_vs_predicted.png"
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_path, filename),
                    dpi=300, bbox_inches='tight')
        plt.close()
        print(f"PNG guardado: {filename}")

    def plot_monte_carlo_distribution(self, group: str):
        key = f"monte_carlo_{group}"
        if key not in self.results:
            print(f"No hay MC para {group}")
            return
        mc = self.results[key]
        pca, sq = mc['pca_predictions'], mc['sq_predictions']

        fig, ax = plt.subplots(1, 2, figsize=(14, 5))
        ax[0].hist(pca, bins=50, color='darkgreen', alpha=0.7)
        ax[0].set_title("Distribución PCA MC")
        ax[1].hist(sq, bins=50, color='steelblue', alpha=0.7)
        ax[1].set_title("Distribución SQ MC")

        filename = f"{group}_monte_carlo_distribution.png"
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_path, filename),
                    dpi=300, bbox_inches='tight')
        plt.close()
        print(f"PNG guardado: {filename}")

    def plot_residual_analysis(self, X_int, X_str, y_pca, y_sq, group: str):
        coeffs = self._get_coeffs(group)
        sq_pred = coeffs['sq_equation']['AV'] * X_int['AV']
        pca_pred = (
            coeffs['pca_equation']['PSE'] * X_str['PSE'] +
            coeffs['pca_equation']['DH'] * X_str['DH'] +
            coeffs['pca_equation']['CS'] * X_str['CS'] +
            coeffs['pca_equation']['SQ'] * sq_pred
        )
        res_pca = y_pca - pca_pred
        res_sq = y_sq - sq_pred if y_sq is not None else None

        if res_sq is not None:
            fig, axes = plt.subplots(2, 2, figsize=(14, 8))
            axes[0, 0].scatter(sq_pred, res_sq, alpha=0.6)
            axes[0, 0].axhline(0, color='red', linestyle='--')
            axes[0, 0].set_title("Res SQ vs Pred")
            axes[0, 1].hist(res_sq, bins=30, color='steelblue', alpha=0.7)
            axes[0, 1].set_title("Hist Res SQ")
            axes[1, 0].scatter(pca_pred, res_pca, alpha=0.6, color='darkgreen')
            axes[1, 0].axhline(0, color='red', linestyle='--')
            axes[1, 0].set_title("Res PCA vs Pred")
            axes[1, 1].hist(res_pca, bins=30, color='darkgreen', alpha=0.7)
            axes[1, 1].set_title("Hist Res PCA")
        else:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            axes[0].scatter(pca_pred, res_pca, alpha=0.6, color='darkgreen')
            axes[0].axhline(0, color='red', linestyle='--')
            axes[0].set_title("Res PCA vs Pred")
            axes[1].hist(res_pca, bins=30, color='darkgreen', alpha=0.7)
            axes[1].set_title("Hist Res PCA")

        filename = f"{group}_residual_analysis.png"
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_path, filename),
                    dpi=300, bbox_inches='tight')
        plt.close()
        print(f"PNG guardado: {filename}")

    def plot_model_comparison(self, group: str):
        key = f"model_comparison_{group}"
        if key not in self.results:
            print(f"No hay comparación para {group}")
            return
        comp = self.results[key]
        s, d, p = comp['structural_r2'], comp['direct_r2'], comp['p_value']

        fig, ax = plt.subplots(1, 2, figsize=(14, 5))
        ax[0].boxplot([s, d], labels=['Estructural', 'Directa'])
        ax[0].set_title("R² CV")
        ax[1].hist(s, bins=20, alpha=0.7, label='Estructural')
        ax[1].hist(d, bins=20, alpha=0.7, label='Directa')
        ax[1].legend()
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        fig.suptitle(f"Comparación Modelos (p={p:.3f} {sig})")

        filename = f"{group}_model_comparison.png"
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_path, filename),
                    dpi=300, bbox_inches='tight')
        plt.close()
        print(f"PNG guardado: {filename}")

    def plot_sensitivity_analysis(self, X_int, X_str, group: str):
        coeffs = self._get_coeffs(group)
        contrib = {
            'PSE→PCA': coeffs['pca_equation']['PSE'],
            'DH→PCA': coeffs['pca_equation']['DH'],
            'CS→PCA': coeffs['pca_equation']['CS'],
            'AV→SQ': coeffs['sq_equation']['AV'],
            'SQ→PCA': coeffs['pca_equation']['SQ'] * coeffs['sq_equation']['AV']
        }
        names, vals = zip(*contrib.items())
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(names, vals, color=['green', 'green', 'green', 'blue', 'blue'])
        ax.set_title("Sensibilidad Coeficientes")
        ax.tick_params(axis='x', rotation=45)

        filename = f"{group}_sensitivity_analysis.png"
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_path, filename),
                    dpi=300, bbox_inches='tight')
        plt.close()
        print(f"PNG guardado: {filename}")

    # ============== Export Excel & TXT ==============

    def save_results_to_excel(self, group='MUJERES'):
        """
        Exporta:
          - {group}_validacion_cruzada.xlsx
          - {group}_monte_carlo_simulaciones.xlsx
          - {group}_comparacion_modelos.xlsx
        """
        print(f"\n[DEBUG] Guardando Excel para {group}")

        # 1) Validación Cruzada
        cv_key = f"structural_cv_{group}"
        if cv_key in self.results:
            rows = []
            for m, met in self.results[cv_key].items():
                for metric in ('RMSE', 'MAE', 'R2'):
                    rows.append({
                        'Grupo': group,
                        'Modelo': m,
                        'Métrica': metric,
                        'Media':    met[f"{metric}_mean"],
                        'Desv_Est': met[f"{metric}_std"]
                    })
            df_cv = pd.DataFrame(rows)
            fn_cv = os.path.join(
                self.output_path, f"{group}_validacion_cruzada.xlsx")
            df_cv.to_excel(fn_cv, index=False)
            print(f"  ✓ CV → {fn_cv}")

        # 2) Monte Carlo (reexporta escenarios, percentiles y estadísticas)
        mc_key = f"monte_carlo_{group}"
        if mc_key in self.results:
            mc = self.results[mc_key]
            fn_mc = os.path.join(
                self.output_path, f"{group}_monte_carlo_simulaciones.xlsx")
            with pd.ExcelWriter(fn_mc, engine='openpyxl') as w:
                mc['scenarios'].to_excel(w, 'Simulaciones', index=False)
                pd.DataFrame({
                    'Percentil': list(mc['pca_percentiles'].keys()),
                    'PCA_Valor': list(mc['pca_percentiles'].values()),
                    'SQ_Valor': list(mc['sq_percentiles'].values())
                }).to_excel(w, 'Percentiles', index=False)
                stats = pd.DataFrame({
                    'Variable': ['PCA', 'SQ'],
                    'Media':    [mc['pca_predictions'].mean(), mc['sq_predictions'].mean()],
                    'Desv_Est': [mc['pca_predictions'].std(),  mc['sq_predictions'].std()],
                    'Min':      [mc['pca_predictions'].min(),  mc['sq_predictions'].min()],
                    'Max':      [mc['pca_predictions'].max(),  mc['sq_predictions'].max()]
                })
                stats.to_excel(w, 'Estadísticas', index=False)
            print(f"  ✓ Monte Carlo → {fn_mc}")

        # 3) Comparación de Modelos
        comp_key = f"model_comparison_{group}"
        if comp_key in self.results:
            comp = self.results[comp_key]
            df_cmp = pd.DataFrame({
                'Fold': range(1, len(comp['structural_r2'])+1),
                'R2_Estructural': comp['structural_r2'],
                'R2_Directa':     comp['direct_r2'],
                'Diferencia':     comp['structural_r2'] - comp['direct_r2']
            })
            fn_cmp = os.path.join(
                self.output_path, f"{group}_comparacion_modelos.xlsx")
            df_cmp.to_excel(fn_cmp, index=False)
            print(f"  ✓ Comparación → {fn_cmp}")

    def generate_comprehensive_report(self, group='MUJERES'):
        """
        Genera TXT: {group}_reporte_completo.txt
        """
        print(f"\n[DEBUG] Generando TXT para {group}")
        rpt = os.path.join(self.output_path, f"{group}_reporte_completo.txt")
        coeffs = self.model_coefficients[group]

        with open(rpt, 'w', encoding='utf-8') as f:
            f.write(f"REPORTE COMPLETO - {group}\n")
            f.write("="*60 + "\n\n")
            # Ecuaciones
            pe = coeffs['pca_equation']
            f.write("1. MODELO ESTRUCTURAL\n")
            f.write(
                f"   PCA = {pe['PSE']}·PSE + {pe['DH']}·DH + {pe['SQ']}·SQ + {pe['CS']}·CS + ε₁\n")
            se = coeffs['sq_equation']['AV']
            f.write(f"   SQ  = {se}·AV + ε₂\n\n")
            # CV
            key_cv = f"structural_cv_{group}"
            if key_cv in self.results:
                f.write("2. VALIDACIÓN CRUZADA\n")
                for m, met in self.results[key_cv].items():
                    f.write(
                        f"   {m}: R² = {met['R2_mean']:.4f} ± {met['R2_std']:.4f}\n")
                f.write("\n")
            # Monte Carlo
            key_mc = f"monte_carlo_{group}"
            if key_mc in self.results:
                mc = self.results[key_mc]
                f.write("3. MONTE CARLO\n")
                f.write(
                    f"   PCA media ± std: {mc['pca_predictions'].mean():.4f} ± {mc['pca_predictions'].std():.4f}\n")
                f.write(
                    f"   SQ  media ± std: {mc['sq_predictions'].mean():.4f} ± {mc['sq_predictions'].std():.4f}\n\n")
            # Comparación
            key_cmp = f"model_comparison_{group}"
            if key_cmp in self.results:
                cmp = self.results[key_cmp]
                f.write("4. COMPARACIÓN MODELOS\n")
                f.write(f"   p-valor: {cmp['p_value']:.6f}\n")
                diff = cmp['structural_r2'].mean() - cmp['direct_r2'].mean()
                f.write(f"   Diferencia media R²: {diff:.4f}\n\n")
            # Archivos
            f.write("5. ARCHIVOS GENERADOS\n")
            files = [
                f"{group}_validacion_cruzada.xlsx",
                f"{group}_monte_carlo_simulaciones.xlsx",
                f"{group}_comparacion_modelos.xlsx",
                f"{group}_reporte_completo.txt"
            ]
            for fn in files:
                f.write(f"   • {fn}\n")

        print(f"  ✓ TXT → {rpt}")

    def generate_all_visualizations(self, X_int, X_str, y_pca, y_sq, group: str):
        self.plot_observed_vs_predicted(X_int, X_str, y_pca, y_sq, group)
        self.plot_monte_carlo_distribution(group)
        self.plot_residual_analysis(X_int, X_str, y_pca, y_sq, group)
        self.plot_model_comparison(group)
        self.plot_sensitivity_analysis(X_int, X_str, group)

    def generate_structural_report(self, group: str):
        """
        Imprime en consola un reporte resumido del análisis
        """
        print(f"\n{'='*60}")
        print(f"REPORTE ESTRUCTURAL PLS-SEM - {group}")
        print("="*60)
        coeffs = self._get_coeffs(group)
        pe = coeffs['pca_equation']
        print(
            f"PCA = {pe['PSE']}·PSE + {pe['DH']}·DH + {pe['SQ']}·SQ + {pe['CS']}·CS + ε₁")
        se = coeffs['sq_equation']['AV']
        print(f"SQ  = {se}·AV + ε₂")

        key_cv = f"structural_cv_{group}"
        if key_cv in self.results:
            print("\nValidación Cruzada:")
            for m, met in self.results[key_cv].items():
                print(
                    f"  {m}: R² = {met['R2_mean']:.4f} ± {met['R2_std']:.4f}")

        key_mc = f"monte_carlo_{group}"
        if key_mc in self.results:
            mc = self.results[key_mc]
            print(
                f"\nMonte Carlo PCA media ± std: {mc['pca_predictions'].mean():.4f} ± {mc['pca_predictions'].std():.4f}")

        key_cmp = f"model_comparison_{group}"
        if key_cmp in self.results:
            cmp = self.results[key_cmp]
            print(f"\nComparación modelos p-valor: {cmp['p_value']:.4f}")

    def run_structural_analysis(self, data_files: dict):
        """
        Ejecuta todo el pipeline para cada grupo y genera:
          - Validación Cruzada
          - Monte Carlo
          - Comparación vs Directa
          - Visualizaciones PNG
          - Excel y TXT
        """
        for group, file_path in data_files.items():
            print(f"\n=== ANÁLISIS ESTRUCTURAL: {group} ===")

            # Crear carpeta de salida
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_path = os.path.join(
                self.base_output_path, f"{group}_{ts}")
            os.makedirs(self.output_path, exist_ok=True)
            print(f"Directorio de salida: {self.output_path}")

            # Carga y preparación
            df = self.load_data(file_path)
            if df is None:
                continue
            Xi, Xs, y_pca, y_sq, _, _ = self.prepare_structural_data(df, group)
            if Xi is None:
                continue

            # 1) Validación Cruzada
            self.structural_cross_validation(Xi, Xs, y_pca, y_sq, group)

            # 2) Monte Carlo Estructural
            self.monte_carlo_structural(Xi, Xs, group)

            # 3) Comparación vs regresión directa
            self.compare_structural_vs_direct_regression(Xi, Xs, y_pca, group)

            # 4) Visualizaciones
            self.generate_all_visualizations(Xi, Xs, y_pca, y_sq, group)

            # 5) Reporte consola
            self.generate_structural_report(group)

            # 6) Exportar Excel y TXT
            self.save_results_to_excel(group)
            self.generate_comprehensive_report(group)

        print("\n=== ANÁLISIS COMPLETADO PARA TODOS LOS GRUPOS ===")


if __name__ == "__main__":
    data_files = {
        'MUJERES': "path/to/DATA_CONSOLIDADA_MUJERES.xlsx",
        'HOMBRES': "path/to/DATA_CONSOLIDADA_HOMBRES.xlsx"
    }
    base_out = "./resultados_plssem/"

    analyzer = PLSSEMPredictAnalyzer(base_output_path=base_out)
    analyzer.run_structural_analysis(data_files)
