"""
app.py — ClinicalFairness Gradio Application
==============================================
Main entry point for HuggingFace Spaces deployment.

4 Tabs
-------
Tab 1 — Upload & Analyze     : Upload datasets, run full analysis
Tab 2 — Fairness Metrics     : View scores and metric details
Tab 3 — Bias Discovery       : UMAP + HDBSCAN cluster visualizations
Tab 4 — Audit Report         : Gemini summary + PDF download

Run Locally
-----------
  python app.py

Deploy on HuggingFace
---------------------
  1. Create new Space at huggingface.co
  2. Select Gradio SDK
  3. Upload all project files
  4. Add GEMINI_API_KEY in Space Secrets
  5. App goes live automatically
"""

import os
import io
import json
import traceback
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gradio as gr
from typing import Dict, Optional, Tuple
import warnings
warnings.filterwarnings("ignore")

# ── Project modules
from fairness_metrics  import run_fairness_analysis, compute_fairness_score
from bias_detector     import run_bias_discovery
from explainability    import run_explainability
from report_generator  import generate_full_report

# ── Gemini API key from environment
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# ── Global state (stores analysis results between tabs)
ANALYSIS_STATE = {}


# ═══════════════════════════════════════════════════════════
#  Helper: Status message builder
# ═══════════════════════════════════════════════════════════
def status_msg(text: str, type: str = "info") -> str:
    icons = {"info": "ℹ️", "success": "✅", "error": "❌", "warning": "⚠️"}
    return f"{icons.get(type, 'ℹ️')} {text}"


# ═══════════════════════════════════════════════════════════
#  Tab 1 — Upload & Analyze
# ═══════════════════════════════════════════════════════════
def run_analysis(heart_file, diabetes_file, model_type, gemini_key, progress=gr.Progress()):
    global ANALYSIS_STATE
    try:
        if heart_file is None or diabetes_file is None:
            return (status_msg("Please upload both dataset files.", "error"), None, None, None, None)
        api_key = gemini_key.strip() or GEMINI_API_KEY
        if not api_key:
            return (status_msg("Please enter your Gemini API key.", "error"), None, None, None, None)
        progress(0.1, desc="Running fairness metrics...")
        fairness_results = run_fairness_analysis(heart_path=heart_file.name, diabetes_path=diabetes_file.name, model_type=model_type)
        progress(0.4, desc="Running Bias Discovery Engine...")
        discovery_results = run_bias_discovery(fairness_results)
        progress(0.65, desc="Computing SHAP explanations...")
        explain_results = run_explainability(fairness_results)
        ANALYSIS_STATE = {"fairness": fairness_results, "discovery": discovery_results, "explain": explain_results, "api_key": api_key}
        progress(0.85, desc="Building summary...")
        summary_html = _build_summary_html(fairness_results, discovery_results)
        metrics_fig = _build_metrics_figure(fairness_results)
        umap_fig = _build_umap_figure(discovery_results, "heart_disease")
        progress(1.0, desc="Analysis complete!")
        return (status_msg("Analysis complete! See tabs below for detailed results.", "success"), summary_html, metrics_fig, umap_fig, None)
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"FULL ERROR:\n{tb}")
        return (status_msg(f"Error: {str(e)}", "error"), f"<pre>{tb}</pre>", None, None, None)
    """
    Main analysis runner triggered by the Analyze button.
    Runs all 3 modules and stores results in global state.
    """
    
        

# ═══════════════════════════════════════════════════════════
#  HTML Summary Builder
# ═══════════════════════════════════════════════════════════
def _build_summary_html(fairness_results: Dict, discovery_results: Dict) -> str:
    """Build HTML summary cards for Tab 1."""
    cards = []

    for dataset_key in ["heart_disease", "diabetes"]:
        if dataset_key not in fairness_results:
            continue

        data = fairness_results[dataset_key]
        meta = data["meta"]
        disc = discovery_results.get(dataset_key, {})

        cards.append(f"""
        <div style="
            background: #0f111a;
            border: 1px solid #1e2235;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 16px;
        ">
            <h3 style="color: #4f8ef7; margin: 0 0 12px 0;">
                🏥 {meta['name']}
            </h3>
            <div style="
                display: grid;
                grid-template-columns: repeat(3, 1fr);
                gap: 12px;
                margin-bottom: 16px;
            ">
                <div style="background:#08090d;border-radius:8px;padding:12px;text-align:center;">
                    <div style="color:#5a607a;font-size:11px;margin-bottom:4px;">PATIENTS</div>
                    <div style="color:white;font-size:20px;font-weight:bold;">{meta['n_samples']}</div>
                </div>
                <div style="background:#08090d;border-radius:8px;padding:12px;text-align:center;">
                    <div style="color:#5a607a;font-size:11px;margin-bottom:4px;">CLUSTERS FOUND</div>
                    <div style="color:#6ee7a0;font-size:20px;font-weight:bold;">{disc.get('n_clusters', 'N/A')}</div>
                </div>
                <div style="background:#08090d;border-radius:8px;padding:12px;text-align:center;">
                    <div style="color:#5a607a;font-size:11px;margin-bottom:4px;">BIAS VERDICT</div>
                    <div style="color:{'#f76e6e' if 'HIGH' in disc.get('bias_verdict','') else '#f7c96e' if 'MODERATE' in disc.get('bias_verdict','') else '#6ee7a0'};font-size:14px;font-weight:bold;">
                        {disc.get('bias_verdict', 'N/A')}
                    </div>
                </div>
            </div>
            <div>
        """)

        for sens_col, results in data["results"].items():
            score = results.get("fairness_score", {})
            s     = score.get("score", 0)
            v     = score.get("verdict", "N/A")
            color = "#6ee7a0" if "FAIR" in v and "SIGNIFICANT" not in v else \
                    "#f7c96e" if "MODERATE" in v else "#f76e6e"

            cards.append(f"""
                <div style="
                    display:flex; justify-content:space-between;
                    align-items:center; padding:8px 0;
                    border-bottom:1px solid #1e2235;
                ">
                    <span style="color:#a0a8c0;font-size:12px;">{sens_col}</span>
                    <span style="color:{color};font-weight:bold;font-size:13px;">
                        {s}/100 — {v}
                    </span>
                </div>
            """)

        cards.append("</div></div>")

    return "".join(cards)


# ═══════════════════════════════════════════════════════════
#  Plotly Figures
# ═══════════════════════════════════════════════════════════
def _build_metrics_figure(fairness_results: Dict) -> go.Figure:
    """Grouped bar chart of fairness scores."""
    datasets, attributes, scores, verdicts = [], [], [], []

    for dataset_key in ["heart_disease", "diabetes"]:
        if dataset_key not in fairness_results:
            continue
        data = fairness_results[dataset_key]
        meta = data["meta"]
        for sens_col, results in data["results"].items():
            s = results.get("fairness_score", {})
            datasets.append(meta["name"])
            attributes.append(sens_col)
            scores.append(s.get("score", 0))
            verdicts.append(s.get("verdict", "N/A"))

    colors_list = []
    for v in verdicts:
        if "FAIR" in v and "SIGNIFICANT" not in v:
            colors_list.append("#6ee7a0")
        elif "MODERATE" in v:
            colors_list.append("#f7c96e")
        else:
            colors_list.append("#f76e6e")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[f"{d}<br>{a}" for d, a in zip(datasets, attributes)],
        y=scores,
        marker_color=colors_list,
        text=[f"{s}/100" for s in scores],
        textposition="outside",
        textfont=dict(color="white", size=11),
    ))

    fig.add_hline(y=80, line_dash="dash", line_color="#6ee7a0",
                  annotation_text="Fair Threshold (80)",
                  annotation_font_color="#6ee7a0")
    fig.add_hline(y=60, line_dash="dash", line_color="#f7c96e",
                  annotation_text="Moderate Threshold (60)",
                  annotation_font_color="#f7c96e")

    fig.update_layout(
        title="Fairness Scores by Dataset and Sensitive Attribute",
        title_font_color="white",
        paper_bgcolor="#08090d",
        plot_bgcolor="#0f111a",
        font=dict(color="white"),
        xaxis=dict(gridcolor="#1e2235"),
        yaxis=dict(gridcolor="#1e2235", range=[0, 115]),
        showlegend=False,
        height=420,
    )
    return fig


def _build_umap_figure(
    discovery_results: Dict,
    dataset_key: str = "heart_disease",
    color_by: str = "cluster",
) -> go.Figure:
    """Interactive UMAP scatter plot."""
    if dataset_key not in discovery_results:
        return go.Figure()

    disc      = discovery_results[dataset_key]
    plot_data = disc.get("plot_data", {})

    if not plot_data:
        return go.Figure()

    x = plot_data["umap_x"]
    y = plot_data["umap_y"]

    if color_by == "cluster":
        color  = [str(l) for l in plot_data["cluster_labels"]]
        title  = f"UMAP — Bias Clusters — {disc['dataset_name']}"
        legend = "Cluster"
    elif color_by == "sensitive" and "sensitive_values" in plot_data:
        color  = [str(v) for v in plot_data["sensitive_values"]]
        title  = f"UMAP — {plot_data.get('sensitive_attr','Sensitive')} — {disc['dataset_name']}"
        legend = plot_data.get("sensitive_attr", "Group")
    else:
        color  = ["Correct" if c == 1 else "Wrong"
                  for c in plot_data.get("correct_predictions", [0]*len(x))]
        title  = f"UMAP — Prediction Correctness — {disc['dataset_name']}"
        legend = "Prediction"

    fig = px.scatter(
        x=x, y=y, color=color,
        title=title,
        labels={"x": "UMAP 1", "y": "UMAP 2", "color": legend},
        color_discrete_sequence=px.colors.qualitative.Vivid,
        opacity=0.75,
    )
    fig.update_traces(marker=dict(size=5))
    fig.update_layout(
        paper_bgcolor="#08090d",
        plot_bgcolor="#0f111a",
        font=dict(color="white"),
        xaxis=dict(gridcolor="#1e2235", zeroline=False),
        yaxis=dict(gridcolor="#1e2235", zeroline=False),
        legend=dict(bgcolor="#0f111a", bordercolor="#1e2235"),
        height=460,
        title_font_color="white",
    )
    return fig


def update_umap(dataset_choice: str, color_choice: str) -> go.Figure:
    """Update UMAP plot based on user selections."""
    if not ANALYSIS_STATE:
        return go.Figure()
    key_map = {"Heart Disease": "heart_disease", "Diabetes": "diabetes"}
    col_map = {"By Cluster": "cluster", "By Sensitive Attribute": "sensitive",
               "By Prediction": "correctness"}
    return _build_umap_figure(
        ANALYSIS_STATE.get("discovery", {}),
        key_map.get(dataset_choice, "heart_disease"),
        col_map.get(color_choice, "cluster"),
    )


def update_shap(dataset_choice: str, plot_choice: str):
    if not ANALYSIS_STATE:
        return None
    import tempfile, os
    key_map = {"Heart Disease": "heart_disease", "Diabetes": "diabetes"}
    plot_map = {
        "Global Importance": "global_importance",
        "Group Comparison": "group_comparison",
        "Waterfall (Patient)": "waterfall",
        "Bias-Causing Features": "disparate_features",
    }
    dataset_key = key_map.get(dataset_choice, "heart_disease")
    plot_key = plot_map.get(plot_choice, "global_importance")
    fig = ANALYSIS_STATE.get("explain", {}).get(dataset_key, {}).get("plots", {}).get(plot_key)
    if fig is None:
        return None
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False, dir="D:\\Projects\\ClinicalFairness")
    fig.savefig(tmp.name, dpi=100, bbox_inches="tight", facecolor="#0f111a")
    tmp.close()
    return tmp.name




# ═══════════════════════════════════════════════════════════
#  PDF Generator (Tab 4)
# ═══════════════════════════════════════════════════════════
def generate_report() -> Tuple:
    """Generate PDF report using stored analysis results."""
    if not ANALYSIS_STATE:
        return status_msg("Please run analysis first.", "error"), None

    try:
        output_path = generate_full_report(
            fairness_results  = ANALYSIS_STATE["fairness"],
            discovery_results = ANALYSIS_STATE["discovery"],
            explain_results   = ANALYSIS_STATE["explain"],
            gemini_api_key    = ANALYSIS_STATE["api_key"],
            output_path       = "ClinicalFairness_Audit_Report.pdf",
        )
        return (
            status_msg("PDF report generated successfully!", "success"),
            output_path,
        )
    except Exception as e:
        return status_msg(f"Report error: {str(e)}", "error"), None


def get_gemini_summary() -> str:
    """Get just the Gemini text summary."""
    if not ANALYSIS_STATE:
        return "Please run analysis first."
    try:
        from report_generator import GeminiReportWriter
        writer  = GeminiReportWriter(api_key=ANALYSIS_STATE["api_key"],     model_name="gemini-2.5-flash"

    

)
        summary = writer.generate_summary(
            ANALYSIS_STATE["fairness"],
            ANALYSIS_STATE["discovery"],
            ANALYSIS_STATE["explain"],
        )
        return summary
    except Exception as e:
        return f"Gemini error: {str(e)}"


# ═══════════════════════════════════════════════════════════
#  Gradio UI Layout
# ═══════════════════════════════════════════════════════════
CSS = """
body { background-color: #08090d !important; }
.gradio-container { background-color: #08090d !important; }
.tab-nav button { background: #0f111a !important; color: #a0a8c0 !important; }
.tab-nav button.selected { color: #4f8ef7 !important; border-bottom: 2px solid #4f8ef7 !important; }
.gr-button-primary { background: #4f8ef7 !important; border: none !important; }
footer { display: none !important; }
"""

DESCRIPTION = """
<div style="text-align:center; padding: 20px 0;">
    <h1 style="color:#4f8ef7; font-size:32px; margin-bottom:8px;">
        ClinicalFairness
    </h1>
    <p style="color:#a0a8c0; font-size:16px; margin-bottom:4px;">
        A Bias Discovery Engine for Medical AI Diagnosis Systems
    </p>
    <p style="color:#5a607a; font-size:13px;">
        BERT · HDBSCAN · UMAP · SHAP · Demographic Parity · Equalized Odds · Google Gemini
    </p>
</div>
"""

with gr.Blocks(css=CSS, theme=gr.themes.Base()) as demo:

    gr.HTML(DESCRIPTION)

    # ── Tab 1: Upload & Analyze
    with gr.Tab("📁 Upload & Analyze"):
        gr.Markdown("### Upload Medical Datasets")
        gr.Markdown(
            "Upload the Heart Disease UCI and Diabetes Pima datasets. "
            "The system will automatically detect sensitive attributes and run full bias analysis."
        )

        with gr.Row():
            heart_file   = gr.File(label="Heart Disease UCI (.csv)", file_types=[".csv"])
            diabetes_file = gr.File(label="Diabetes Pima (.csv)",    file_types=[".csv"])

        with gr.Row():
            model_type = gr.Dropdown(
                choices=["random_forest", "gradient_boost", "logistic"],
                value="random_forest",
                label="Classifier Model",
            )
            gemini_key = gr.Textbox(
                label="Gemini API Key",
                placeholder="Paste your Gemini API key here (get free at aistudio.google.com)",
                type="password",
            )

        analyze_btn = gr.Button("🚀 Run Full Bias Analysis", variant="primary", size="lg")

        status_out  = gr.Textbox(label="Status", interactive=False)
        summary_out = gr.HTML(label="Analysis Summary")

        # Hidden outputs passed to other tabs
        metrics_fig_out = gr.Plot(visible=False)
        umap_fig_out    = gr.Plot(visible=False)
        shap_fig_out    = gr.Image(visible=False)

        analyze_btn.click(
            fn=run_analysis,
            inputs=[heart_file, diabetes_file, model_type, gemini_key],
            outputs=[status_out, summary_out, metrics_fig_out, umap_fig_out, shap_fig_out],
        )

    # ── Tab 2: Fairness Metrics
    with gr.Tab("📊 Fairness Metrics"):
        gr.Markdown("### Fairness Scores & Detailed Metrics")
        gr.Markdown(
            "View fairness scores across all demographic groups. "
            "Green = Fair (>80), Orange = Moderate Bias (60-80), Red = Significant Bias (<60)."
        )

        metrics_plot = gr.Plot(label="Fairness Scores Overview")

        with gr.Row():
            with gr.Column():
                gr.Markdown("#### Heart Disease — Detailed Results")
                hd_table = gr.Dataframe(
                    headers=["Metric", "Group", "Value", "Disparity", "Status"],
                    label="Heart Disease Fairness Metrics",
                    interactive=False,
                )
            with gr.Column():
                gr.Markdown("#### Diabetes — Detailed Results")
                db_table = gr.Dataframe(
                    headers=["Metric", "Group", "Value", "Disparity", "Status"],
                    label="Diabetes Fairness Metrics",
                    interactive=False,
                )

        refresh_metrics_btn = gr.Button("🔄 Load Metrics", variant="secondary")

        def load_metrics():
            if not ANALYSIS_STATE:
                return None, pd.DataFrame(), pd.DataFrame()

            fig = _build_metrics_figure(ANALYSIS_STATE["fairness"])

            def extract_table(dataset_key):
                rows = []
                data = ANALYSIS_STATE["fairness"].get(dataset_key, {})
                for sens_col, results in data.get("results", {}).items():
                    for metric_key in ["demographic_parity", "equalized_odds",
                                       "equal_opportunity", "per_group_accuracy"]:
                        m = results.get(metric_key, {})
                        if not m:
                            continue
                        rows.append([
                            m.get("metric", metric_key),
                            sens_col,
                            str(m.get("rates", m.get("accuracy", m.get("tpr_per_group", "")))),
                            str(m.get("disparity", m.get("tpr_disparity", "N/A"))),
                            "✓ PASS" if m.get("passed") else "✗ FAIL",
                        ])
                return pd.DataFrame(rows, columns=["Metric", "Group", "Value", "Disparity", "Status"])

            return fig, extract_table("heart_disease"), extract_table("diabetes")

        refresh_metrics_btn.click(
            fn=load_metrics,
            outputs=[metrics_plot, hd_table, db_table],
        )

    # ── Tab 3: Bias Discovery + SHAP
    with gr.Tab("🔍 Bias Discovery"):
        gr.Markdown("### HDBSCAN + UMAP Bias Discovery Engine")

        with gr.Row():
            umap_dataset = gr.Dropdown(
                choices=["Heart Disease", "Diabetes"],
                value="Heart Disease",
                label="Dataset",
            )
            umap_color = gr.Dropdown(
                choices=["By Cluster", "By Sensitive Attribute", "By Prediction"],
                value="By Cluster",
                label="Color By",
            )
            umap_refresh = gr.Button("🔄 Update Plot", variant="secondary")

        umap_plot = gr.Plot(label="UMAP Visualization")

        umap_refresh.click(
            fn=update_umap,
            inputs=[umap_dataset, umap_color],
            outputs=[umap_plot],
        )

        gr.Markdown("---")
        gr.Markdown("### SHAP Explainability")

        with gr.Row():
            shap_dataset = gr.Dropdown(
                choices=["Heart Disease", "Diabetes"],
                value="Heart Disease",
                label="Dataset",
            )
            shap_plot_type = gr.Dropdown(
                choices=["Global Importance", "Group Comparison",
                         "Waterfall (Patient)", "Bias-Causing Features"],
                value="Global Importance",
                label="Plot Type",
            )
            shap_refresh = gr.Button("🔄 Update Plot", variant="secondary")

        shap_plot = gr.Image(label="SHAP Plot", type="filepath")

        shap_refresh.click(
            fn=update_shap,
            inputs=[shap_dataset, shap_plot_type],
            outputs=[shap_plot],
        )

        # Chi-Square results table
        gr.Markdown("### Chi-Square Significance Tests")
        chi2_table = gr.Dataframe(
            headers=["Dataset", "Attribute", "Chi² Stat", "P-Value", "Significant?"],
            label="Statistical Bias Significance",
            interactive=False,
        )

        load_chi2_btn = gr.Button("🔄 Load Chi-Square Results", variant="secondary")

        def load_chi2():
            if not ANALYSIS_STATE:
                return pd.DataFrame()
            rows = []
            for key in ["heart_disease", "diabetes"]:
                disc = ANALYSIS_STATE.get("discovery", {}).get(key, {})
                for attr, result in disc.get("chi2_results", {}).items():
                    rows.append([
                        disc.get("dataset_name", key),
                        attr,
                        f"{result['chi2_stat']:.2f}",
                        f"{result['p_value']:.4f}",
                        "⚠ YES" if result["significant"] else "✓ NO",
                    ])
            return pd.DataFrame(
                rows,
                columns=["Dataset", "Attribute", "Chi² Stat", "P-Value", "Significant?"]
            )

        load_chi2_btn.click(fn=load_chi2, outputs=[chi2_table])

    # ── Tab 4: Audit Report
    with gr.Tab("📄 Audit Report"):
        gr.Markdown("### AI-Powered Audit Report")
        gr.Markdown(
            "Generate a complete bias audit report with Gemini LLM summary "
            "and downloadable PDF. Make sure you have run the analysis first."
        )

        with gr.Row():
            gemini_btn = gr.Button("🤖 Generate Gemini Summary", variant="secondary")
            pdf_btn    = gr.Button("📥 Generate + Download PDF", variant="primary")

        report_status  = gr.Textbox(label="Report Status", interactive=False)
        gemini_summary = gr.Textbox(
            label="Gemini AI Audit Summary",
            lines=20,
            interactive=False,
            placeholder="Click 'Generate Gemini Summary' to see the AI-written audit report...",
        )
        pdf_download = gr.File(label="Download PDF Report")

        gemini_btn.click(
            fn=get_gemini_summary,
            outputs=[gemini_summary],
        )

        pdf_btn.click(
            fn=generate_report,
            outputs=[report_status, pdf_download],
        )

        gr.Markdown("""
        ---
        ### About This Report
        The PDF audit report contains:
        - **Cover page** with project details and date
        - **Fairness scores** color-coded table
        - **Detailed metrics** — Demographic Parity, Equalized Odds, Equal Opportunity
        - **Bias Discovery results** — HDBSCAN clusters + Chi-Square tests
        - **SHAP plots** — Feature importance and bias-causing features
        - **Gemini AI narrative** — Plain English summary and recommendations
        """)

# ═══════════════════════════════════════════════════════════
#  Launch
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
