"""
bias_detector.py — Bias Discovery Engine
==========================================
The CORE module of ClinicalFairness.
Uses unsupervised ML to discover HIDDEN bias clusters
that fairness metrics alone cannot detect.

Pipeline
---------
1. HDBSCAN  → Find natural demographic clusters in data
2. GMM      → Soft cluster assignments (probability based)
3. UMAP     → Reduce to 2D for visualization
4. Analysis → Check if certain clusters get worse predictions
5. Chi-Square → Confirm statistical significance of bias

Datasets
---------
- Heart Disease UCI  → Gender + Age + Origin clusters
- Diabetes Pima      → Age group clusters
"""

import numpy as np
import pandas as pd
import hdbscan
import umap
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from scipy.stats import chi2_contingency
from sklearn.metrics import accuracy_score, f1_score
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════
#  HDBSCAN Clusterer
# ═══════════════════════════════════════════════════════════
class HDBSCANBiasDiscovery:
    """
    Discovers hidden demographic clusters using HDBSCAN.
    Then checks if model predictions are unfair across clusters.
    """

    def __init__(
        self,
        min_cluster_size: int = 15,
        min_samples:      int = 5,
        n_components_umap: int = 2,
        random_state:     int = 42,
    ):
        self.min_cluster_size  = min_cluster_size
        self.min_samples       = min_samples
        self.random_state      = random_state
        self.n_components_umap = n_components_umap

        self.scaler     = StandardScaler()
        self.umap_model = None
        self.hdbscan_model = None
        self.gmm_model  = None

        # Results stored after fit
        self.X_scaled   = None
        self.X_umap     = None
        self.labels_    = None
        self.probs_     = None
        self.n_clusters_= 0

    # ── Step 1: Scale + UMAP reduction
    
    def _reduce_dimensions(self, X: np.ndarray) -> np.ndarray:
        """Scale features and reduce to 2D using UMAP."""
        print("  Scaling features...")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
         
        self.X_scaled = self.scaler.fit_transform(X)
        self.X_scaled = np.nan_to_num(self.X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        
        print("  Running UMAP dimensionality reduction...")
        self.umap_model = umap.UMAP(
            n_components=self.n_components_umap,
            n_neighbors=15,
            min_dist=0.1,
            random_state=self.random_state,
            verbose=False,
        )
        self.X_scaled = np.nan_to_num(self.X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        X_umap = self.umap_model.fit_transform(self.X_scaled)
        print(f"  UMAP output shape: {X_umap.shape}")
        return X_umap
        
        

    # ── Step 2: HDBSCAN clustering
    def _run_hdbscan(self, X_umap: np.ndarray) -> np.ndarray:
        """Find natural clusters using HDBSCAN."""
        print("  Running HDBSCAN clustering...")
        self.hdbscan_model = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            prediction_data=True,
        )
        labels = self.hdbscan_model.fit_predict(X_umap)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise    = (labels == -1).sum()
        print(f"  Clusters found: {n_clusters}  |  Noise points: {n_noise}")
        self.n_clusters_ = n_clusters
        return labels

    # ── Step 3: GMM soft assignments
    def _run_gmm(self, X_umap: np.ndarray, n_clusters: int) -> np.ndarray:
        """Soft cluster probabilities using Gaussian Mixture Model."""
        if n_clusters < 2:
            n_clusters = 2
        print(f"  Running GMM with {n_clusters} components...")
        self.gmm_model = GaussianMixture(
            n_components=min(n_clusters, 8),
            random_state=self.random_state,
            covariance_type="full",
        )
        self.gmm_model.fit(X_umap)
        probs = self.gmm_model.predict_proba(X_umap)
        return probs

    # ── Step 4: Bias analysis per cluster
    def _analyze_cluster_bias(
        self,
        labels:          np.ndarray,
        y_true:          np.ndarray,
        y_pred:          np.ndarray,
        sensitive_data:  Dict[str, np.ndarray],
    ) -> Dict:
        """
        For each cluster, check:
        - Demographic composition (who is in this cluster?)
        - Prediction accuracy (is model worse for this cluster?)
        - Sensitive attribute distribution
        """
        cluster_ids = [l for l in np.unique(labels) if l != -1]
        cluster_analysis = {}

        for cid in cluster_ids:
            mask = labels == cid
            yt   = y_true[mask]
            yp   = y_pred[mask]

            if len(yt) == 0:
                continue

            acc = accuracy_score(yt, yp) if len(np.unique(yt)) > 0 else 0
            f1  = f1_score(yt, yp, zero_division=0)

            # Sensitive attribute composition
            composition = {}
            for attr, values in sensitive_data.items():
                vals_in_cluster = values[mask]
                unique, counts  = np.unique(vals_in_cluster, return_counts=True)
                composition[attr] = {
                    str(u): round(c / len(vals_in_cluster), 3)
                    for u, c in zip(unique, counts)
                }

            cluster_analysis[f"Cluster_{cid}"] = {
                "size":        int(mask.sum()),
                "accuracy":    round(acc, 4),
                "f1_score":    round(f1, 4),
                "composition": composition,
                "positive_rate": round(yp.mean(), 4),
            }

        return cluster_analysis

    # ── Step 5: Chi-Square significance test
    def _chi_square_test(
        self,
        labels:         np.ndarray,
        sensitive_vals: np.ndarray,
        attr_name:      str,
    ) -> Dict:
        """
        Test if cluster membership is statistically associated
        with a sensitive attribute using Chi-Square test.
        High chi2 + low p-value = significant bias.
        """
        # Remove noise points
        mask   = labels != -1
        labs   = labels[mask]
        sens   = sensitive_vals[mask]

        contingency = pd.crosstab(labs, sens)
        chi2, p_val, dof, expected = chi2_contingency(contingency)

        return {
            "attribute":   attr_name,
            "chi2_stat":   round(float(chi2), 4),
            "p_value":     round(float(p_val), 6),
            "dof":         int(dof),
            "significant": p_val < 0.05,
            "interpretation": (
                f"SIGNIFICANT: {attr_name} is statistically associated with cluster membership "
                f"(chi2={chi2:.2f}, p={p_val:.4f}). This indicates demographic clustering — potential bias."
                if p_val < 0.05 else
                f"NOT SIGNIFICANT: {attr_name} shows no significant cluster association "
                f"(chi2={chi2:.2f}, p={p_val:.4f})."
            )
        }

    # ── Main fit method
    def fit_discover(
        self,
        X:              np.ndarray,
        y_true:         np.ndarray,
        y_pred:         np.ndarray,
        sensitive_data: Dict[str, np.ndarray],
        dataset_name:   str = "Dataset",
    ) -> Dict:
        """
        Full bias discovery pipeline.
        Returns complete analysis results.
        """
        print(f"\n{'='*50}")
        print(f"BIAS DISCOVERY ENGINE — {dataset_name}")
        print(f"{'='*50}")
        print(f"Input: {X.shape[0]} samples × {X.shape[1]} features")

        # ── Reduce dimensions
        self.X_umap = self._reduce_dimensions(X)

        # ── Cluster with HDBSCAN
        self.labels_ = self._run_hdbscan(self.X_umap)

        # ── Soft assignments with GMM
        self.probs_ = self._run_gmm(self.X_umap, self.n_clusters_)

        # ── Cluster bias analysis
        print("  Analyzing bias per cluster...")
        cluster_analysis = self._analyze_cluster_bias(
            self.labels_, y_true, y_pred, sensitive_data
        )

        # ── Chi-Square tests
        print("  Running Chi-Square significance tests...")
        chi2_results = {}
        for attr, values in sensitive_data.items():
            chi2_results[attr] = self._chi_square_test(
                self.labels_, values, attr
            )
            sig = chi2_results[attr]["significant"]
            print(f"  {attr}: {'⚠ SIGNIFICANT BIAS' if sig else '✓ No significant bias'}")

        # ── Overall bias verdict
        n_significant = sum(r["significant"] for r in chi2_results.values())
        if n_significant == 0:
            bias_verdict = "LOW BIAS"
            bias_color   = "green"
        elif n_significant <= len(chi2_results) // 2:
            bias_verdict = "MODERATE BIAS"
            bias_color   = "orange"
        else:
            bias_verdict = "HIGH BIAS"
            bias_color   = "red"

        print(f"\n  Overall Bias Verdict: {bias_verdict}")

        return {
            "dataset_name":    dataset_name,
            "n_samples":       X.shape[0],
            "n_clusters":      self.n_clusters_,
            "X_umap":          self.X_umap,
            "labels":          self.labels_,
            "gmm_probs":       self.probs_,
            "cluster_analysis": cluster_analysis,
            "chi2_results":    chi2_results,
            "bias_verdict":    bias_verdict,
            "bias_color":      bias_color,
            "n_significant":   n_significant,
        }


# ═══════════════════════════════════════════════════════════
#  UMAP Visualization Data Builder
# ═══════════════════════════════════════════════════════════
def build_umap_plot_data(
    discovery_result: Dict,
    sensitive_data:   Dict[str, np.ndarray],
    y_true:           np.ndarray,
    y_pred:           np.ndarray,
) -> Dict:
    """
    Build data structures for Plotly UMAP visualizations.
    Returns data for 3 plots:
      1. Colored by HDBSCAN cluster
      2. Colored by sensitive attribute
      3. Colored by prediction correctness
    """
    X_umap  = discovery_result["X_umap"]
    labels  = discovery_result["labels"]

    plot_data = {
        "umap_x": X_umap[:, 0].tolist(),
        "umap_y": X_umap[:, 1].tolist(),
    }

    # Plot 1 — by cluster
    plot_data["cluster_labels"] = labels.tolist()

    # Plot 2 — by sensitive attribute (first one)
    if sensitive_data:
        first_attr = list(sensitive_data.keys())[0]
        plot_data["sensitive_attr"]       = first_attr
        plot_data["sensitive_values"]     = sensitive_data[first_attr].tolist()
        plot_data["all_sensitive_data"]   = {
            k: v.tolist() for k, v in sensitive_data.items()
        }

    # Plot 3 — by prediction correctness
    correct = (y_true == y_pred).astype(int)
    plot_data["correct_predictions"] = correct.tolist()
    plot_data["y_true"]              = y_true.tolist()
    plot_data["y_pred"]              = y_pred.tolist()

    return plot_data


# ═══════════════════════════════════════════════════════════
#  Cluster Summary Text (for Gemini prompt)
# ═══════════════════════════════════════════════════════════
def build_cluster_summary(discovery_result: Dict) -> str:
    """
    Build a text summary of bias discovery findings
    for Gemini LLM to use in report generation.
    """
    lines = []
    lines.append(f"Dataset: {discovery_result['dataset_name']}")
    lines.append(f"Samples: {discovery_result['n_samples']}")
    lines.append(f"Clusters discovered: {discovery_result['n_clusters']}")
    lines.append(f"Overall Bias Verdict: {discovery_result['bias_verdict']}")
    lines.append("")

    lines.append("Chi-Square Test Results:")
    for attr, result in discovery_result["chi2_results"].items():
        sig = "SIGNIFICANT BIAS DETECTED" if result["significant"] else "No significant bias"
        lines.append(f"  - {attr}: {sig} (chi2={result['chi2_stat']}, p={result['p_value']})")

    lines.append("")
    lines.append("Cluster Performance Summary:")
    for cluster_id, info in discovery_result["cluster_analysis"].items():
        lines.append(
            f"  {cluster_id}: size={info['size']}, "
            f"accuracy={info['accuracy']}, "
            f"f1={info['f1_score']}"
        )
        for attr, comp in info["composition"].items():
            top = max(comp, key=comp.get)
            lines.append(f"    Dominant {attr}: {top} ({comp[top]:.1%})")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════
#  Main Runner — Both Datasets
# ═══════════════════════════════════════════════════════════
def run_bias_discovery(fairness_results: Dict) -> Dict:
    """
    Run Bias Discovery Engine on both datasets.
    Takes output from fairness_metrics.run_fairness_analysis().
    """
    discovery_results = {}

    for dataset_key in ["heart_disease", "diabetes"]:
        if dataset_key not in fairness_results:
            continue

        data    = fairness_results[dataset_key]
        meta    = data["meta"]
        df_enc  = data["df_encoded"]
        df_orig = data["df_original"]
        td      = data["test_data"]

        # Features for clustering
        X       = df_enc.iloc[td["idx"]][meta["feature_cols"]].values
        y_true  = td["y_true"]
        y_pred  = td["y_pred"]

        # Sensitive data
        sensitive_data = {}
        for col in meta["sensitive_cols"]:
            if col in df_orig.columns:
                sensitive_data[col] = df_orig.iloc[td["idx"]][col].values.astype(str)

        # Run discovery
        engine = HDBSCANBiasDiscovery(min_cluster_size=10)
        result = engine.fit_discover(
            X=X,
            y_true=y_true,
            y_pred=y_pred,
            sensitive_data=sensitive_data,
            dataset_name=meta["name"],
        )

        # Build plot data
        result["plot_data"] = build_umap_plot_data(
            result, sensitive_data, y_true, y_pred
        )
        result["summary_text"] = build_cluster_summary(result)

        discovery_results[dataset_key] = result
        print(f"\n✅ {meta['name']} — Discovery complete")
        print(f"   Verdict: {result['bias_verdict']}")

    return discovery_results


# ═══════════════════════════════════════════════════════════
#  Quick Test
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    from fairness_metrics import run_fairness_analysis

    # Run fairness first
    fairness_results = run_fairness_analysis(
        heart_path="heart_disease_uci.csv",
        diabetes_path="diabetes.csv",
    )

    # Run bias discovery
    discovery = run_bias_discovery(fairness_results)

    for key, result in discovery.items():
        print(f"\n{key}: {result['n_clusters']} clusters — {result['bias_verdict']}")
        print(result["summary_text"])
