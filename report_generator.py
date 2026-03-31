"""
report_generator.py — Gemini LLM + PDF Audit Report
======================================================
Two Parts
----------
Part 1 — Gemini LLM
  - Reads all fairness metrics + bias discovery findings
  - Generates human readable medical audit summary
  - Gives specific recommendations to fix bias

Part 2 — PDF Report Generator (ReportLab)
  - Professional PDF with cover page
  - Fairness scores with color coding
  - SHAP plots embedded as images
  - UMAP cluster visualizations
  - Gemini written narrative
  - Actionable recommendations

Output
-------
  ClinicalFairness_Audit_Report.pdf
"""

import os
import io
import json
import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings("ignore")

# ── Gemini

from google import genai


# ── ReportLab PDF
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm, mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, Image, PageBreak, KeepTogether
)
from reportlab.platypus.flowables import BalancedColumns
from reportlab.graphics.shapes import Drawing, Rect
from reportlab.pdfgen import canvas


# ═══════════════════════════════════════════════════════════
#  Color Palette (matches dark UI theme)
# ═══════════════════════════════════════════════════════════
DARK_BG     = colors.HexColor("#08090d")
SURFACE     = colors.HexColor("#0f111a")
BORDER      = colors.HexColor("#1e2235")
TEXT_MAIN   = colors.HexColor("#e2e4ef")
TEXT_MUTED  = colors.HexColor("#5a607a")
ACCENT_BLUE = colors.HexColor("#4f8ef7")
ACCENT_ORG  = colors.HexColor("#f7934f")
ACCENT_GRN  = colors.HexColor("#6ee7a0")
ACCENT_PRP  = colors.HexColor("#d46ef7")
RED_BIAS    = colors.HexColor("#f76e6e")
GREEN_FAIR  = colors.HexColor("#6ef7a0")
ORANGE_MOD  = colors.HexColor("#f7c96e")


def verdict_color(verdict: str):
    v = verdict.upper()
    if "FAIR" in v and "SIGNIFICANT" not in v:
        return GREEN_FAIR
    elif "MODERATE" in v:
        return ORANGE_MOD
    else:
        return RED_BIAS


# ═══════════════════════════════════════════════════════════
#  Part 1 — Gemini LLM Summary Generator
# ═══════════════════════════════════════════════════════════
class GeminiReportWriter:
    """
    Uses Google Gemini API to generate human readable
    medical bias audit summaries and recommendations.
    """

    
    def __init__(self, api_key: str, model_name: str = "gemini-2.0-flash"):
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def _build_prompt(
        self,
        fairness_results:   Dict,
        discovery_results:  Dict,
        explain_results:    Dict,
    ) -> str:
        """Build a detailed prompt from all analysis results."""

        prompt_parts = []
        prompt_parts.append(
            "You are a medical AI ethics expert writing a clinical bias audit report. "
            "Based on the following fairness analysis findings, write a professional "
            "audit summary. Use clear, non-technical language where possible. "
            "Be specific about which patient groups are affected and what the clinical "
            "implications are.\n\n"
        )

        # ── Fairness metrics summary
        prompt_parts.append("=== FAIRNESS METRICS FINDINGS ===\n")
        for dataset_key in ["heart_disease", "diabetes"]:
            if dataset_key not in fairness_results:
                continue
            data = fairness_results[dataset_key]
            meta = data["meta"]
            prompt_parts.append(f"\nDataset: {meta['name']} ({meta['n_samples']} patients)")
            prompt_parts.append(f"Disease: {meta['disease']}")

            for sens_col, results in data["results"].items():
                score = results.get("fairness_score", {})
                prompt_parts.append(f"\n  Sensitive Attribute: {sens_col}")
                prompt_parts.append(f"  Fairness Score: {score.get('score', 'N/A')}/100 — {score.get('verdict', '')}")

                dp = results.get("demographic_parity", {})
                if "disparity" in dp:
                    prompt_parts.append(f"  Demographic Parity Disparity: {dp['disparity']:.2%}")
                    prompt_parts.append(f"  Rates per group: {dp.get('rates', {})}")

                eo = results.get("equalized_odds", {})
                if "tpr_disparity" in eo:
                    prompt_parts.append(f"  True Positive Rate Disparity: {eo['tpr_disparity']:.2%}")
                    prompt_parts.append(f"  TPR per group: {eo.get('tpr_per_group', {})}")

                pg = results.get("per_group_accuracy", {})
                if "accuracy" in pg:
                    prompt_parts.append(f"  Accuracy per group: {pg['accuracy']}")

        # ── Bias discovery summary
        prompt_parts.append("\n\n=== BIAS DISCOVERY ENGINE FINDINGS ===\n")
        for dataset_key in ["heart_disease", "diabetes"]:
            if dataset_key not in discovery_results:
                continue
            disc = discovery_results[dataset_key]
            prompt_parts.append(f"\nDataset: {disc['dataset_name']}")
            prompt_parts.append(f"Clusters discovered: {disc['n_clusters']}")
            prompt_parts.append(f"Overall Bias Verdict: {disc['bias_verdict']}")
            prompt_parts.append("Chi-Square Test Results:")
            for attr, chi2 in disc.get("chi2_results", {}).items():
                sig = "SIGNIFICANT" if chi2["significant"] else "NOT significant"
                prompt_parts.append(
                    f"  {attr}: {sig} (chi2={chi2['chi2_stat']}, p={chi2['p_value']})"
                )

        # ── SHAP explainability
        prompt_parts.append("\n\n=== SHAP EXPLAINABILITY FINDINGS ===\n")
        for dataset_key in ["heart_disease", "diabetes"]:
            if dataset_key not in explain_results:
                continue
            exp = explain_results[dataset_key]
            meta = exp["meta"]
            prompt_parts.append(f"\nDataset: {meta['name']}")

            imp = exp["importance_df"].head(5)
            prompt_parts.append("Top 5 most important features:")
            for _, row in imp.iterrows():
                prompt_parts.append(f"  - {row['feature']}: {row['importance']:.4f}")

            top_disp = exp["group_comparison"]["top_disparate_features"]
            prompt_parts.append(f"Top bias-causing features (by {exp['sensitive_col_used']}):")
            for feat, disp in top_disp[:3]:
                prompt_parts.append(f"  - {feat}: disparity = {disp:.4f}")

        # ── Instructions for Gemini
        prompt_parts.append("""

=== YOUR TASK ===
Write a professional clinical AI bias audit report with these exact sections:

1. EXECUTIVE SUMMARY (3-4 sentences)
   Summarize the key bias findings and their clinical significance.

2. KEY FINDINGS (bullet points)
   List the most important bias findings with specific numbers.

3. CLINICAL IMPLICATIONS
   Explain what these biases mean for real patients in plain language.
   Which patient groups are at risk? What could go wrong in clinical practice?

4. RECOMMENDATIONS (numbered list)
   Provide 5 specific, actionable recommendations to reduce the detected bias.
   Be specific — mention which algorithms, techniques, or data changes would help.

5. CONCLUSION (2-3 sentences)
   Summarize urgency and next steps.

Use professional medical ethics language. Be specific with numbers from the findings above.
""")

        return "\n".join(prompt_parts)

    def generate_summary(
        self,
        fairness_results:  Dict,
        discovery_results: Dict,
        explain_results:   Dict,
    ) -> str:
        """Generate complete audit summary using Gemini."""
        print("\n  Calling Gemini API...")
        try:
            hd = fairness_results.get("heart_disease", {})
            db = fairness_results.get("diabetes", {})
            hd_scores = {k: v.get("fairness_score", {}).get("score", 0) for k, v in hd.get("results", {}).items()}
            db_scores = {k: v.get("fairness_score", {}).get("score", 0) for k, v in db.get("results", {}).items()}
            hd_disc = discovery_results.get("heart_disease", {}).get("bias_verdict", "N/A")
            db_disc = discovery_results.get("diabetes", {}).get("bias_verdict", "N/A")
            prompt = f"Write a 5 sentence medical AI bias audit report covering executive summary, key findings, clinical implications and conclusion. Heart Disease bias verdict: {hd_disc}, fairness scores: {hd_scores}. Diabetes bias verdict: {db_disc}, fairness scores: {db_scores}. End with 5 actionable recommendations."
            response = self.client.models.generate_content(model=self.model_name, contents=prompt)
            summary  = response.text
            print("  ✅ Gemini summary generated successfully")
            return summary
        except Exception as e:
            print(f"  ⚠ Gemini API error: {e}")
            return self._fallback_summary(fairness_results, discovery_results)

    def _fallback_summary(self, fairness_results: Dict, discovery_results: Dict) -> str:
        """Fallback summary if Gemini API fails."""
        lines = ["EXECUTIVE SUMMARY\n"]
        lines.append(
            "This ClinicalFairness audit analyzed medical AI diagnosis systems "
            "for demographic bias across patient groups.\n"
        )
        lines.append("KEY FINDINGS\n")
        for key in ["heart_disease", "diabetes"]:
            if key in fairness_results:
                meta = fairness_results[key]["meta"]
                lines.append(f"• {meta['name']}: Fairness analysis completed across {meta['n_samples']} patients.")
        lines.append("\nRECOMMENDATIONS\n")
        lines.append("1. Apply fairness-aware resampling techniques (SMOTE per demographic group).")
        lines.append("2. Use adversarial debiasing during model training.")
        lines.append("3. Implement fairness constraints in the objective function.")
        lines.append("4. Collect more representative data for underrepresented groups.")
        lines.append("5. Conduct regular bias audits on deployed models.")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════
#  Part 2 — PDF Report Generator
# ═══════════════════════════════════════════════════════════
class PDFReportGenerator:
    """
    Generates a professional PDF audit report using ReportLab.
    """

    def __init__(self, output_path: str = "ClinicalFairness_Audit_Report.pdf"):
        self.output_path = output_path
        self.styles      = self._build_styles()
        self.story       = []

    def _build_styles(self):
        styles = getSampleStyleSheet()

        custom = {
            "cover_title": ParagraphStyle(
                "cover_title",
                fontSize=28, fontName="Helvetica-Bold",
                textColor=colors.white, alignment=TA_CENTER,
                spaceAfter=10,
            ),
            "cover_sub": ParagraphStyle(
                "cover_sub",
                fontSize=14, fontName="Helvetica",
                textColor=colors.HexColor("#a0a8c0"), alignment=TA_CENTER,
                spaceAfter=6,
            ),
            "section_header": ParagraphStyle(
                "section_header",
                fontSize=15, fontName="Helvetica-Bold",
                textColor=ACCENT_BLUE, spaceBefore=16, spaceAfter=8,
                borderPad=4,
            ),
            "sub_header": ParagraphStyle(
                "sub_header",
                fontSize=12, fontName="Helvetica-Bold",
                textColor=ACCENT_PRP, spaceBefore=10, spaceAfter=6,
            ),
            "body": ParagraphStyle(
                "body",
                fontSize=10, fontName="Helvetica",
                textColor=TEXT_MAIN, leading=15,
                spaceAfter=6, alignment=TA_JUSTIFY,
            ),
            "bullet": ParagraphStyle(
                "bullet",
                fontSize=10, fontName="Helvetica",
                textColor=TEXT_MAIN, leading=14,
                leftIndent=16, spaceAfter=4,
            ),
            "metric_label": ParagraphStyle(
                "metric_label",
                fontSize=9, fontName="Helvetica-Bold",
                textColor=TEXT_MUTED,
            ),
            "metric_value": ParagraphStyle(
                "metric_value",
                fontSize=11, fontName="Helvetica-Bold",
                textColor=TEXT_MAIN,
            ),
            "caption": ParagraphStyle(
                "caption",
                fontSize=8, fontName="Helvetica",
                textColor=TEXT_MUTED, alignment=TA_CENTER,
                spaceAfter=8,
            ),
            "gemini_text": ParagraphStyle(
                "gemini_text",
                fontSize=10, fontName="Helvetica",
                textColor=TEXT_MAIN, leading=16,
                spaceAfter=8, alignment=TA_JUSTIFY,
            ),
        }
        styles.add(custom["cover_title"])
        styles.add(custom["cover_sub"])
        styles.add(custom["section_header"])
        styles.add(custom["sub_header"])
        styles.add(custom["body"])
        styles.add(custom["bullet"])
        styles.add(custom["metric_label"])
        styles.add(custom["metric_value"])
        styles.add(custom["caption"])
        styles.add(custom["gemini_text"])
        return styles

    # ── Cover Page
    def _add_cover(self):
        date_str = datetime.datetime.now().strftime("%B %d, %Y")

        # Dark background rectangle
        d = Drawing(500, 200)
        r = Rect(0, 0, 500, 200, fillColor=SURFACE, strokeColor=ACCENT_BLUE, strokeWidth=1)
        d.add(r)
        self.story.append(d)
        self.story.append(Spacer(1, 0.5*cm))

        self.story.append(Paragraph("ClinicalFairness", self.styles["cover_title"]))
        self.story.append(Paragraph(
            "A Bias Discovery Engine for Medical AI Diagnosis Systems",
            self.styles["cover_sub"]
        ))
        self.story.append(Paragraph(
            "Bias Audit Report — Confidential",
            self.styles["cover_sub"]
        ))
        self.story.append(Spacer(1, 0.3*cm))
        self.story.append(HRFlowable(
            width="100%", thickness=1,
            color=ACCENT_BLUE, spaceAfter=10
        ))
        self.story.append(Paragraph(
            f"Generated: {date_str}  |  Datasets: Heart Disease UCI + Diabetes Pima  |  "
            f"Models: Random Forest  |  LLM: Google Gemini",
            self.styles["caption"]
        ))
        self.story.append(PageBreak())

    # ── Section header
    def _section(self, title: str):
        self.story.append(HRFlowable(
            width="100%", thickness=0.5,
            color=BORDER, spaceBefore=8, spaceAfter=4
        ))
        self.story.append(Paragraph(title, self.styles["section_header"]))

    # ── Fairness score badge table
    def _add_fairness_scores(self, fairness_results: Dict):
        self._section("📊 Fairness Scores Overview")

        rows = [["Dataset", "Sensitive Attribute", "Score", "Verdict"]]
        for dataset_key in ["heart_disease", "diabetes"]:
            if dataset_key not in fairness_results:
                continue
            data = fairness_results[dataset_key]
            meta = data["meta"]
            for sens_col, results in data["results"].items():
                score = results.get("fairness_score", {})
                rows.append([
                    meta["name"],
                    sens_col,
                    f"{score.get('score', 'N/A')}/100",
                    score.get("verdict", "N/A"),
                ])

        col_widths = [5*cm, 4.5*cm, 2.5*cm, 5*cm]
        t = Table(rows, colWidths=col_widths, repeatRows=1)

        style = TableStyle([
            ("BACKGROUND",    (0, 0), (-1, 0),  SURFACE),
            ("TEXTCOLOR",     (0, 0), (-1, 0),  ACCENT_BLUE),
            ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
            ("FONTSIZE",      (0, 0), (-1, 0),  9),
            ("FONTSIZE",      (0, 1), (-1, -1), 9),
            ("TEXTCOLOR",     (0, 1), (-1, -1), TEXT_MAIN),
            ("BACKGROUND",    (0, 1), (-1, -1), DARK_BG),
            ("ROWBACKGROUNDS",(0, 1), (-1, -1), [DARK_BG, SURFACE]),
            ("GRID",          (0, 0), (-1, -1), 0.5, BORDER),
            ("ALIGN",         (2, 0), (3, -1),  "CENTER"),
            ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING",    (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ])

        # Color verdict cells
        for i, row in enumerate(rows[1:], start=1):
            verdict = row[3]
            if "FAIR" in verdict.upper() and "SIGNIFICANT" not in verdict.upper():
                style.add("TEXTCOLOR", (3, i), (3, i), GREEN_FAIR)
            elif "MODERATE" in verdict.upper():
                style.add("TEXTCOLOR", (3, i), (3, i), ORANGE_MOD)
            else:
                style.add("TEXTCOLOR", (3, i), (3, i), RED_BIAS)

        t.setStyle(style)
        self.story.append(t)
        self.story.append(Spacer(1, 0.5*cm))

    # ── Detailed metrics per dataset
    def _add_detailed_metrics(self, fairness_results: Dict):
        self._section("📋 Detailed Fairness Metrics")

        for dataset_key in ["heart_disease", "diabetes"]:
            if dataset_key not in fairness_results:
                continue
            data = fairness_results[dataset_key]
            meta = data["meta"]

            self.story.append(Paragraph(
                f"▸ {meta['name']}  —  {meta['n_samples']} patients  |  Disease: {meta['disease']}",
                self.styles["sub_header"]
            ))

            for sens_col, results in data["results"].items():
                self.story.append(Paragraph(
                    f"Sensitive Attribute: {sens_col}",
                    self.styles["metric_label"]
                ))

                # Metric rows
                metric_rows = [["Metric", "Value", "Disparity", "Status"]]

                dp = results.get("demographic_parity", {})
                if dp:
                    metric_rows.append([
                        "Demographic Parity",
                        str(dp.get("rates", {})),
                        f"{dp.get('disparity', 0):.4f}",
                        "✓ PASS" if dp.get("passed") else "✗ FAIL",
                    ])

                eo = results.get("equalized_odds", {})
                if eo:
                    metric_rows.append([
                        "Equalized Odds (TPR)",
                        str(eo.get("tpr_per_group", {})),
                        f"{eo.get('tpr_disparity', 0):.4f}",
                        "✓ PASS" if eo.get("passed") else "✗ FAIL",
                    ])

                pg = results.get("per_group_accuracy", {})
                if pg:
                    metric_rows.append([
                        "Per-Group Accuracy",
                        str(pg.get("accuracy", {})),
                        f"{pg.get('disparity', 0):.4f}",
                        "✓ PASS" if pg.get("passed") else "✗ FAIL",
                    ])

                col_widths = [4*cm, 6*cm, 2.5*cm, 2.5*cm]
                t = Table(metric_rows, colWidths=col_widths, repeatRows=1)
                t.setStyle(TableStyle([
                    ("BACKGROUND",    (0, 0), (-1, 0),  SURFACE),
                    ("TEXTCOLOR",     (0, 0), (-1, 0),  ACCENT_PRP),
                    ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
                    ("FONTSIZE",      (0, 0), (-1, -1), 8),
                    ("TEXTCOLOR",     (0, 1), (-2, -1), TEXT_MAIN),
                    ("BACKGROUND",    (0, 1), (-1, -1), DARK_BG),
                    ("GRID",          (0, 0), (-1, -1), 0.3, BORDER),
                    ("ALIGN",         (2, 0), (3, -1),  "CENTER"),
                    ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
                    ("TOPPADDING",    (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                    ("WORDWRAP",      (1, 1), (1, -1),  True),
                ]))

                # Color pass/fail
                for i, row in enumerate(metric_rows[1:], start=1):
                    if "PASS" in row[3]:
                        t.setStyle(TableStyle([("TEXTCOLOR", (3, i), (3, i), GREEN_FAIR)]))
                    else:
                        t.setStyle(TableStyle([("TEXTCOLOR", (3, i), (3, i), RED_BIAS)]))

                self.story.append(t)
                self.story.append(Spacer(1, 0.4*cm))

    # ── Bias Discovery section
    def _add_bias_discovery(self, discovery_results: Dict):
        self._section("🔍 Bias Discovery Engine Results")

        for dataset_key in ["heart_disease", "diabetes"]:
            if dataset_key not in discovery_results:
                continue
            disc = discovery_results[dataset_key]

            self.story.append(Paragraph(
                f"▸ {disc['dataset_name']}",
                self.styles["sub_header"]
            ))

            verdict_col = verdict_color(disc["bias_verdict"])

            summary_rows = [
                ["Clusters Discovered", str(disc["n_clusters"])],
                ["Overall Bias Verdict", disc["bias_verdict"]],
                ["Significant Biases Found", str(disc["n_significant"])],
            ]
            t = Table(summary_rows, colWidths=[5*cm, 10*cm])
            t.setStyle(TableStyle([
                ("FONTSIZE",      (0, 0), (-1, -1), 9),
                ("TEXTCOLOR",     (0, 0), (0, -1),  TEXT_MUTED),
                ("FONTNAME",      (0, 0), (0, -1),  "Helvetica-Bold"),
                ("TEXTCOLOR",     (1, 0), (1, -1),  TEXT_MAIN),
                ("BACKGROUND",    (0, 0), (-1, -1), DARK_BG),
                ("GRID",          (0, 0), (-1, -1), 0.3, BORDER),
                ("TOPPADDING",    (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]))
            t.setStyle(TableStyle([("TEXTCOLOR", (1, 1), (1, 1), verdict_col)]))
            self.story.append(t)
            self.story.append(Spacer(1, 0.3*cm))

            # Chi-square results
            self.story.append(Paragraph("Statistical Significance (Chi-Square Tests):", self.styles["metric_label"]))
            chi2_rows = [["Attribute", "Chi² Stat", "P-Value", "Significant?"]]
            for attr, chi2 in disc.get("chi2_results", {}).items():
                chi2_rows.append([
                    attr,
                    f"{chi2['chi2_stat']:.2f}",
                    f"{chi2['p_value']:.4f}",
                    "YES ⚠" if chi2["significant"] else "NO ✓",
                ])
            t2 = Table(chi2_rows, colWidths=[4*cm, 3*cm, 3*cm, 5*cm], repeatRows=1)
            t2.setStyle(TableStyle([
                ("BACKGROUND",    (0, 0), (-1, 0),  SURFACE),
                ("TEXTCOLOR",     (0, 0), (-1, 0),  ACCENT_ORG),
                ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
                ("FONTSIZE",      (0, 0), (-1, -1), 8),
                ("TEXTCOLOR",     (0, 1), (-1, -1), TEXT_MAIN),
                ("BACKGROUND",    (0, 1), (-1, -1), DARK_BG),
                ("GRID",          (0, 0), (-1, -1), 0.3, BORDER),
                ("ALIGN",         (1, 0), (-1, -1), "CENTER"),
                ("TOPPADDING",    (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]))
            for i, row in enumerate(chi2_rows[1:], start=1):
                if "YES" in row[3]:
                    t2.setStyle(TableStyle([("TEXTCOLOR", (3, i), (3, i), RED_BIAS)]))
                else:
                    t2.setStyle(TableStyle([("TEXTCOLOR", (3, i), (3, i), GREEN_FAIR)]))
            self.story.append(t2)
            self.story.append(Spacer(1, 0.5*cm))

    # ── SHAP plots
    def _add_shap_plots(self, explain_results: Dict):
        self._section("🔬 SHAP Explainability Analysis")

        for dataset_key in ["heart_disease", "diabetes"]:
            if dataset_key not in explain_results:
                continue
            exp  = explain_results[dataset_key]
            meta = exp["meta"]
            plots = exp["plots"]

            self.story.append(Paragraph(
                f"▸ {meta['name']}", self.styles["sub_header"]
            ))

            # Top features table
            imp = exp["importance_df"].head(8)
            feat_rows = [["Rank", "Feature", "SHAP Importance"]]
            for i, row in imp.iterrows():
                feat_rows.append([str(i+1), row["feature"], f"{row['importance']:.4f}"])
            t = Table(feat_rows, colWidths=[1.5*cm, 7*cm, 4*cm], repeatRows=1)
            t.setStyle(TableStyle([
                ("BACKGROUND",    (0, 0), (-1, 0),  SURFACE),
                ("TEXTCOLOR",     (0, 0), (-1, 0),  ACCENT_GRN),
                ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
                ("FONTSIZE",      (0, 0), (-1, -1), 8),
                ("TEXTCOLOR",     (0, 1), (-1, -1), TEXT_MAIN),
                ("BACKGROUND",    (0, 1), (-1, -1), DARK_BG),
                ("GRID",          (0, 0), (-1, -1), 0.3, BORDER),
                ("ALIGN",         (0, 0), (0, -1),  "CENTER"),
                ("ALIGN",         (2, 0), (2, -1),  "CENTER"),
                ("TOPPADDING",    (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]))
            self.story.append(t)
            self.story.append(Spacer(1, 0.4*cm))

            # Embed plots as images
            for plot_key, plot_name in [
                ("global_importance", "Global Feature Importance"),
                ("group_comparison",  "Feature Importance by Demographic Group"),
                ("disparate_features","Bias-Causing Features"),
            ]:
                fig = plots.get(plot_key)
                if fig is None:
                    continue
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=100, bbox_inches="tight",
                            facecolor="#0f111a")
                buf.seek(0)
                img = Image(buf, width=15*cm, height=7*cm)
                self.story.append(img)
                self.story.append(Paragraph(
                    f"Figure: {plot_name} — {meta['name']}",
                    self.styles["caption"]
                ))
                self.story.append(Spacer(1, 0.3*cm))

    # ── Gemini narrative
    def _add_gemini_narrative(self, gemini_summary: str):
        self.story.append(PageBreak())
        self._section("🤖 AI-Generated Audit Summary (Google Gemini)")

        self.story.append(Paragraph(
            "The following audit summary was generated by Google Gemini based on "
            "the quantitative findings above.",
            self.styles["body"]
        ))
        self.story.append(Spacer(1, 0.3*cm))

        # Split by sections and render
        for line in gemini_summary.split("\n"):
            line = line.strip()
            if not line:
                self.story.append(Spacer(1, 0.15*cm))
                continue

            # Section headers
            if any(line.startswith(h) for h in [
                "EXECUTIVE", "KEY FINDINGS", "CLINICAL", "RECOMMENDATIONS", "CONCLUSION"
            ]):
                self.story.append(Paragraph(line, self.styles["sub_header"]))
            elif line.startswith(("•", "-", "*", "1.", "2.", "3.", "4.", "5.")):
                self.story.append(Paragraph(f"  {line}", self.styles["bullet"]))
            else:
                self.story.append(Paragraph(line, self.styles["gemini_text"]))

    # ── Build full PDF
    def build(
        self,
        fairness_results:  Dict,
        discovery_results: Dict,
        explain_results:   Dict,
        gemini_summary:    str,
    ) -> str:
        """Build the complete PDF report. Returns output file path."""

        print(f"\n  Building PDF report: {self.output_path}")

        doc = SimpleDocTemplate(
            self.output_path,
            pagesize=A4,
            leftMargin=2*cm, rightMargin=2*cm,
            topMargin=2*cm,  bottomMargin=2*cm,
        )

        self._add_cover()
        self._add_fairness_scores(fairness_results)
        self.story.append(PageBreak())
        self._add_detailed_metrics(fairness_results)
        self.story.append(PageBreak())
        self._add_bias_discovery(discovery_results)
        self.story.append(PageBreak())
        self._add_shap_plots(explain_results)
        self._add_gemini_narrative(gemini_summary)

        doc.build(self.story)
        print(f"  ✅ PDF saved: {self.output_path}")
        return self.output_path


# ═══════════════════════════════════════════════════════════
#  Main Runner
# ═══════════════════════════════════════════════════════════
def generate_full_report(
    fairness_results:  Dict,
    discovery_results: Dict,
    explain_results:   Dict,
    gemini_api_key:    str,
    output_path:       str = "ClinicalFairness_Audit_Report.pdf",
) -> str:
    """
    Generate complete audit report with Gemini summary + PDF.
    Returns path to generated PDF.
    """
    print("\n" + "="*50)
    print("GENERATING AUDIT REPORT")
    print("="*50)

    # ── Step 1: Gemini summary
    print("\nPart 1 — Gemini LLM Summary")
    writer  = GeminiReportWriter(api_key=gemini_api_key)
    summary = writer.generate_summary(
        fairness_results, discovery_results, explain_results
    )

    # ── Step 2: PDF generation
    print("\nPart 2 — PDF Report Generation")
    pdf_gen = PDFReportGenerator(output_path=output_path)
    path    = pdf_gen.build(
        fairness_results, discovery_results, explain_results, summary
    )

    return path


# ═══════════════════════════════════════════════════════════
#  Quick Test (without Gemini API key)
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Report generator ready.")
    print("To test: call generate_full_report() with your Gemini API key.")
    print("Get free API key at: https://aistudio.google.com/")
