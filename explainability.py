"""
explainability.py — SHAP Explainability Module
================================================
Explains WHY the medical AI model makes biased predictions.

What It Does
-------------
1. SHAP Values        → Which features drive predictions globally
2. Group SHAP         → How feature importance differs per demographic
3. Waterfall Plot     → Explain a single patient prediction
4. Summary Plot       → All features ranked by importance
5. Bias SHAP          → Which features cause unfair predictions

Answers The Key Question
-------------------------
"The model is biased against Female patients —
 but WHY? Which medical features cause this bias?"
"""

import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")   # Non-interactive backend for Gradio
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Optional, Tuple
import warnings
import io
import base64
warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════
#  SHAP Explainer Builder
# ═══════════════════════════════════════════════════════════
class ClinicalSHAPExplainer:
    """
    SHAP-based explainability for medical AI bias detection.
    Works with RandomForest and GradientBoosting models.
    """

    def __init__(self, model, feature_cols: List[str], dataset_name: str):
        self.model        = model
        self.feature_cols = feature_cols
        self.dataset_name = dataset_name
        self.explainer    = None
        self.shap_values  = None
        self.X_explain    = None

    def build_explainer(self, X_train: np.ndarray, sample_size: int = 100):
        """
        Build SHAP TreeExplainer.
        Uses a background sample for efficiency.
        """
        print(f"\n  Building SHAP explainer for {self.dataset_name}...")

        # Use a small background sample for speed
        n = min(sample_size, len(X_train))
        background = shap.sample(X_train, n, random_state=42)

        self.explainer = shap.TreeExplainer(
            self.model,
            background,
            feature_perturbation="tree_path_dependent",
        )
        print(f"  Explainer ready. Background samples: {n}")
        return self

    def compute_shap_values(self, X_test: np.ndarray, max_samples: int = 200):
        """
        Compute SHAP values for test set.
        Limits to max_samples for speed.
        """
        n = min(max_samples, len(X_test))
        self.X_explain   = X_test[:n]

        print(f"  Computing SHAP values for {n} samples...")
        raw = self.explainer.shap_values(self.X_explain, check_additivity=False)
        if isinstance(raw, list):
            raw = raw[1] if len(raw) == 2 else raw[0]
        if raw.ndim == 3:
            raw = raw[:, :, 1]
        self.shap_values = raw
        return self.shap_values

        

        print(f"  SHAP values shape: {self.shap_values.shape}")
        return self.shap_values

    # ── Global feature importance
    def global_importance(self) -> pd.DataFrame:
        """
        Compute mean absolute SHAP value per feature.
        Higher = more important for predictions.
        """
        mean_abs = np.abs(self.shap_values).mean(axis=0)
        df = pd.DataFrame({
            "feature":    self.feature_cols,
            "importance": mean_abs,
        }).sort_values("importance", ascending=False).reset_index(drop=True)
        return df

    # ── Per-group SHAP comparison
    def group_shap_comparison(
        self,
        sensitive_values: np.ndarray,
        max_samples: int = 200,
    ) -> Dict:
        """
        Compare SHAP feature importance across demographic groups.
        Shows which features drive predictions differently per group.
        """
        sens = sensitive_values[:max_samples]
        groups = np.unique(sens)
        group_importance = {}

        for g in groups:
            mask = sens == g
            if mask.sum() == 0:
                continue
            shap_g = self.shap_values[mask]
            mean_abs = np.abs(shap_g).mean(axis=0)
            group_importance[str(g)] = dict(zip(self.feature_cols, mean_abs))

        # Find features with biggest disparity across groups
        disparities = {}
        for feat in self.feature_cols:
            vals = [group_importance[g][feat]
                    for g in group_importance if feat in group_importance[g]]
            if vals:
                disparities[feat] = round(max(vals) - min(vals), 4)

        top_disparate = sorted(disparities.items(), key=lambda x: x[1], reverse=True)[:5]

        return {
            "group_importance": group_importance,
            "disparities":      disparities,
            "top_disparate_features": top_disparate,
        }

    # ── Single patient explanation
    def explain_patient(self, patient_idx: int) -> Dict:
        """
        Explain prediction for a single patient.
        Returns feature contributions (positive = toward disease).
        """
        if patient_idx >= len(self.X_explain):
            patient_idx = 0

        shap_patient  = self.shap_values[patient_idx]
        contributions = dict(zip(self.feature_cols, shap_patient))
        contributions_sorted = sorted(
            contributions.items(), key=lambda x: abs(x[1]), reverse=True
        )

        return {
            "patient_idx":   patient_idx,
            "contributions": contributions_sorted,
            "top_positive":  [(f, v) for f, v in contributions_sorted if v > 0][:3],
            "top_negative":  [(f, v) for f, v in contributions_sorted if v < 0][:3],
            "base_value":    float(self.explainer.expected_value[1])
                             if isinstance(self.explainer.expected_value, np.ndarray)
                             else float(self.explainer.expected_value),
        }


# ═══════════════════════════════════════════════════════════
#  Plot Generators (return base64 images for Gradio)
# ═══════════════════════════════════════════════════════════
def fig_to_base64(fig) -> str:
    """Convert matplotlib figure to base64 string for Gradio display."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                facecolor="#0f111a", edgecolor="none")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return img_base64


def plot_global_importance(
    importance_df: pd.DataFrame,
    dataset_name:  str,
    top_n:         int = 10,
) -> plt.Figure:
    """
    Bar chart of global SHAP feature importance.
    Dark themed to match Gradio UI.
    """
    df = importance_df.head(top_n)

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#0f111a")
    ax.set_facecolor("#0f111a")

    colors = cm.RdYlGn_r(np.linspace(0.1, 0.9, len(df)))
    bars   = ax.barh(df["feature"][::-1], df["importance"][::-1],
                     color=colors, edgecolor="none", height=0.6)

    ax.set_xlabel("Mean |SHAP Value|", color="white", fontsize=11)
    ax.set_title(f"Feature Importance — {dataset_name}\n(SHAP Global Explanation)",
                 color="white", fontsize=13, fontweight="bold", pad=15)
    ax.tick_params(colors="white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#2a2d3e")
    ax.spines["bottom"].set_color("#2a2d3e")
    ax.xaxis.label.set_color("white")

    # Add value labels
    for bar, val in zip(bars, df["importance"][::-1]):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", color="white", fontsize=9)

    plt.tight_layout()
    return fig


def plot_group_shap_comparison(
    group_comparison: Dict,
    dataset_name:     str,
    top_features:     int = 8,
) -> plt.Figure:
    """
    Grouped bar chart comparing SHAP importance per demographic group.
    Shows which features drive bias between groups.
    """
    group_imp = group_comparison["group_importance"]
    groups    = list(group_imp.keys())

    # Get top features by average importance
    all_feats = list(next(iter(group_imp.values())).keys())
    avg_imp   = {f: np.mean([group_imp[g].get(f, 0) for g in groups])
                 for f in all_feats}
    top_feats = sorted(avg_imp.items(), key=lambda x: x[1], reverse=True)[:top_features]
    top_feat_names = [f for f, _ in top_feats]

    x     = np.arange(len(top_feat_names))
    width = 0.8 / len(groups)
    colors = ["#4f8ef7", "#f7934f", "#6ee7a0", "#d46ef7", "#f76e6e"]

    fig, ax = plt.subplots(figsize=(13, 6))
    fig.patch.set_facecolor("#0f111a")
    ax.set_facecolor("#0f111a")

    for i, (group, color) in enumerate(zip(groups, colors)):
        vals = [group_imp[group].get(f, 0) for f in top_feat_names]
        offset = (i - len(groups) / 2) * width + width / 2
        ax.bar(x + offset, vals, width * 0.9, label=group,
               color=color, alpha=0.85, edgecolor="none")

    ax.set_xticks(x)
    ax.set_xticklabels(top_feat_names, rotation=30, ha="right", color="white", fontsize=9)
    ax.set_ylabel("Mean |SHAP Value|", color="white")
    ax.set_title(
        f"SHAP Feature Importance by Demographic Group — {dataset_name}\n"
        f"(Differences reveal which features cause bias)",
        color="white", fontsize=12, fontweight="bold", pad=12
    )
    ax.legend(facecolor="#1e2235", labelcolor="white", edgecolor="#2a2d3e")
    ax.tick_params(colors="white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#2a2d3e")
    ax.spines["bottom"].set_color("#2a2d3e")

    plt.tight_layout()
    return fig


def plot_waterfall(
    patient_explanation: Dict,
    dataset_name:        str,
) -> plt.Figure:
    """
    Waterfall chart for a single patient's prediction explanation.
    Shows how each feature pushes the prediction toward/away from disease.
    """
    contribs = patient_explanation["contributions"][:10]
    features = [f for f, _ in contribs]
    values   = [v for _, v in contribs]

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#0f111a")
    ax.set_facecolor("#0f111a")

    colors = ["#f76e6e" if v > 0 else "#6ee7a0" for v in values]
    bars   = ax.barh(features[::-1], values[::-1],
                     color=colors[::-1], edgecolor="none", height=0.6)

    ax.axvline(0, color="white", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("SHAP Value (→ Disease  |  ← No Disease)", color="white", fontsize=10)
    ax.set_title(
        f"Single Patient Explanation — {dataset_name}\n"
        f"(Red = pushes toward disease, Green = pushes away)",
        color="white", fontsize=12, fontweight="bold", pad=12
    )
    ax.tick_params(colors="white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#2a2d3e")
    ax.spines["bottom"].set_color("#2a2d3e")

    # Value labels
    for bar, val in zip(bars, values[::-1]):
        ax.text(val + (0.002 if val >= 0 else -0.002),
                bar.get_y() + bar.get_height() / 2,
                f"{val:+.4f}", va="center", color="white", fontsize=9,
                ha="left" if val >= 0 else "right")

    plt.tight_layout()
    return fig


def plot_disparate_features(
    group_comparison: Dict,
    dataset_name:     str,
) -> plt.Figure:
    """
    Highlight top features with biggest SHAP disparity across groups.
    These are the features CAUSING the bias.
    """
    top_disparate = group_comparison["top_disparate_features"]
    if not top_disparate:
        return None

    features    = [f for f, _ in top_disparate]
    disparities = [d for _, d in top_disparate]

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("#0f111a")
    ax.set_facecolor("#0f111a")

    cmap   = cm.Reds(np.linspace(0.4, 0.9, len(features)))
    bars   = ax.barh(features[::-1], disparities[::-1],
                     color=cmap[::-1], edgecolor="none", height=0.55)

    ax.set_xlabel("SHAP Importance Disparity Across Groups", color="white", fontsize=10)
    ax.set_title(
        f"⚠ Bias-Causing Features — {dataset_name}\n"
        f"(Higher disparity = feature treats groups differently)",
        color="white", fontsize=12, fontweight="bold", pad=12
    )
    ax.tick_params(colors="white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#2a2d3e")
    ax.spines["bottom"].set_color("#2a2d3e")

    for bar, val in zip(bars, disparities[::-1]):
        ax.text(bar.get_width() + 0.0005,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", color="white", fontsize=9)

    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════
#  Main Runner — Both Datasets
# ═══════════════════════════════════════════════════════════
def run_explainability(fairness_results: Dict) -> Dict:
    """
    Run SHAP explainability on both datasets.
    Takes output from fairness_metrics.run_fairness_analysis().
    Returns all plots and explanations.
    """
    explain_results = {}

    for dataset_key in ["heart_disease", "diabetes"]:
        if dataset_key not in fairness_results:
            continue

        data     = fairness_results[dataset_key]
        meta     = data["meta"]
        df_enc   = data["df_encoded"]
        df_orig  = data["df_original"]
        model    = data["model"]
        scaler   = data["scaler"]
        td       = data["test_data"]

        print(f"\n{'='*50}")
        print(f"SHAP EXPLAINABILITY — {meta['name']}")
        print(f"{'='*50}")

        # Prepare data
        X_all   = scaler.transform(df_enc[meta["feature_cols"]].values)
        X_test  = scaler.transform(
            df_enc.iloc[td["idx"]][meta["feature_cols"]].values
        )

        # Build explainer
        explainer = ClinicalSHAPExplainer(model, meta["feature_cols"], meta["name"])
        explainer.build_explainer(X_all, sample_size=100)
        explainer.compute_shap_values(X_test, max_samples=150)

        # Global importance
        importance_df = explainer.global_importance()
        print(f"\n  Top 5 Important Features:")
        print(importance_df.head(5).to_string(index=False))

        # Group SHAP comparison (use first sensitive col)
        first_sens_col = meta["sensitive_cols"][0]
        sens_vals = df_orig.iloc[td["idx"]][first_sens_col].values.astype(str)[:150]
        group_comparison = explainer.group_shap_comparison(sens_vals)

        print(f"\n  Top Bias-Causing Features (by {first_sens_col}):")
        for feat, disp in group_comparison["top_disparate_features"]:
            print(f"    {feat}: disparity = {disp:.4f}")

        # Single patient explanation
        patient_exp = explainer.explain_patient(patient_idx=0)

        # Generate plots
        print("\n  Generating plots...")
        fig_global    = plot_global_importance(importance_df, meta["name"])
        fig_group     = plot_group_shap_comparison(group_comparison, meta["name"])
        fig_waterfall = plot_waterfall(patient_exp, meta["name"])
        fig_disparate = plot_disparate_features(group_comparison, meta["name"])

        explain_results[dataset_key] = {
            "meta":              meta,
            "importance_df":     importance_df,
            "group_comparison":  group_comparison,
            "patient_explanation": patient_exp,
            "plots": {
                "global_importance": fig_global,
                "group_comparison":  fig_group,
                "waterfall":         fig_waterfall,
                "disparate_features": fig_disparate,
            },
            "sensitive_col_used": first_sens_col,
        }

        print(f"  ✅ {meta['name']} explainability complete")

    return explain_results


# ═══════════════════════════════════════════════════════════
#  Quick Test
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    from fairness_metrics import run_fairness_analysis

    fairness_results = run_fairness_analysis(
        heart_path="heart_disease_uci.csv",
        diabetes_path="diabetes.csv",
    )

    explain_results = run_explainability(fairness_results)

    for key, result in explain_results.items():
        print(f"\n{key}:")
        print(f"  Top feature: {result['importance_df'].iloc[0]['feature']}")
        print(f"  Top bias-causing: {result['group_comparison']['top_disparate_features'][0]}")

    # Save one plot to verify
    fig = explain_results["heart_disease"]["plots"]["global_importance"]
    fig.savefig("test_shap_plot.png", dpi=100, bbox_inches="tight")
    print("\n✅ Test plot saved: test_shap_plot.png")
