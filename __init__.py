"""
ClinicalFairness: A Bias Discovery Engine for Medical AI Diagnosis Systems
===========================================================================
Project Structure
-----------------
app.py                  → Main Gradio application (Step 9)
bias_detector.py        → HDBSCAN + UMAP Bias Discovery Engine (Step 5)
fairness_metrics.py     → Demographic Parity, Equalized Odds (Step 4)
explainability.py       → SHAP + Attention visualization (Step 6)
report_generator.py     → PDF + Gemini LLM summary (Steps 7 & 8)
preprocessing.py        → Data cleaning + embedding pipeline (Step 3)
requirements.txt        → All dependencies
README.md               → GitHub documentation

Author  : Your Name
Dataset : Kaggle Medical Datasets (Diabetes, Heart Disease, Skin Cancer)
Deploy  : HuggingFace Spaces (Gradio)
LLM     : Google Gemini API (free tier)
"""
