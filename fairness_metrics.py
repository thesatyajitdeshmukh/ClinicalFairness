"""
fairness_metrics.py — Fairness Metrics Module
===============================================
Built specifically for:
  - Heart Disease UCI  (920 rows) → Gender + Age + Origin bias
  - Diabetes Pima      (768 rows) → Age group bias

Metrics
--------
1. Demographic Parity      — Equal positive prediction rates across groups
2. Equalized Odds          — Equal TPR and FPR across groups
3. Equal Opportunity       — Equal true positive rates across groups
4. Per-group Accuracy      — Accuracy broken down by demographic group
5. Counterfactual Fairness — Would decision change if sensitive attr changed
6. Overall Fairness Score  — 0 to 100 composite score
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════
#  Age Group Helper
# ═══════════════════════════════════════════════════════
def create_age_groups(age_series: pd.Series) -> pd.Series:
    return pd.cut(
        age_series,
        bins=[0, 30, 50, 120],
        labels=["Young (<30)", "Middle (30-50)", "Senior (>50)"]
    ).astype(str)


# ═══════════════════════════════════════════════════════
#  Heart Disease Loader
# ═══════════════════════════════════════════════════════
def load_heart_disease(file_path: str):
    df = pd.read_csv(file_path)

    # Binary target
    df["target"] = (df["num"] > 0).astype(int)
    df["age_group"] = create_age_groups(df["age"])

    # Handle missing values
    for col in ["trestbps", "chol", "thalch", "oldpeak"]:
        if col in df.columns:
            df[col].fillna(df[col].median(), inplace=True)
    for col in ["fbs", "restecg", "exang", "slope", "ca", "thal", "cp"]:
        if col in df.columns:
            df[col].fillna(df[col].mode()[0], inplace=True)

    # Encode for model
    df_encoded = df.copy()
    le = LabelEncoder()
    for col in ["cp", "restecg", "exang", "slope", "thal", "fbs", "sex"]:
        if col in df_encoded.columns:
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))

    feature_cols = ["age", "trestbps", "chol", "fbs", "restecg",
                    "thalch", "exang", "oldpeak", "cp"]
    feature_cols = [c for c in feature_cols if c in df_encoded.columns]

    meta = {
        "name": "Heart Disease UCI",
        "sensitive_cols": ["sex", "age_group", "dataset"],
        "target_col": "target",
        "feature_cols": feature_cols,
        "n_samples": len(df),
        "disease": "Heart Disease",
    }
    return df, df_encoded, meta


# ═══════════════════════════════════════════════════════
#  Diabetes Loader
# ═══════════════════════════════════════════════════════
def load_diabetes(file_path: str):
    df = pd.read_csv(file_path)
    df["age_group"] = create_age_groups(df["Age"])
    df["pregnancies_group"] = pd.cut(
        df["Pregnancies"],
        bins=[-1, 0, 3, 20],
        labels=["None (0)", "Low (1-3)", "High (4+)"]
    ).astype(str)

    feature_cols = ["Pregnancies", "Glucose", "BloodPressure",
                    "SkinThickness", "Insulin", "BMI",
                    "DiabetesPedigreeFunction", "Age"]

    meta = {
        "name": "Diabetes Pima Indians",
        "sensitive_cols": ["age_group", "pregnancies_group"],
        "target_col": "Outcome",
        "feature_cols": feature_cols,
        "n_samples": len(df),
        "disease": "Diabetes",
    }
    return df, df.copy(), meta


# ═══════════════════════════════════════════════════════
#  Model Trainer
# ═══════════════════════════════════════════════════════
def train_model(df_encoded, feature_cols, target_col, model_type="random_forest"):
    X = df_encoded[feature_cols].values
    y = df_encoded[target_col].values

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X_scaled, y, np.arange(len(y)),
        test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "random_forest":  RandomForestClassifier(n_estimators=100, random_state=42),
        "gradient_boost": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "logistic":       LogisticRegression(max_iter=1000, random_state=42),
    }
    model = models.get(model_type, models["random_forest"])
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"F1 Score : {f1_score(y_test, y_pred):.4f}")
    print(f"AUC-ROC  : {roc_auc_score(y_test, y_prob):.4f}")

    return model, scaler, X_test, y_test, y_pred, y_prob, idx_test


# ═══════════════════════════════════════════════════════
#  Fairness Analyzer
# ═══════════════════════════════════════════════════════
class FairnessAnalyzer:

    def __init__(self, y_true, y_pred, y_prob, sensitive_values, group_name):
        self.y_true    = np.array(y_true)
        self.y_pred    = np.array(y_pred)
        self.y_prob    = np.array(y_prob)
        self.sensitive = np.array(sensitive_values)
        self.group_name = group_name
        self.groups    = np.unique(self.sensitive)

    def _confusion_stats(self, mask):
        yt = self.y_true[mask]
        yp = self.y_pred[mask]
        if len(np.unique(yt)) < 2:
            return 0, 0, 0, sum(yt)
        tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
        return tn, fp, fn, tp

    def demographic_parity(self):
        rates = {g: self.y_pred[self.sensitive == g].mean() for g in self.groups}
        disparity = max(rates.values()) - min(rates.values())
        return {
            "metric": "Demographic Parity",
            "rates": {k: round(v, 4) for k, v in rates.items()},
            "disparity": round(disparity, 4),
            "passed": disparity < 0.1,
            "explanation": f"Positive prediction rate differs by {disparity:.2%} across {self.group_name} groups."
        }

    def equalized_odds(self):
        tpr_dict, fpr_dict = {}, {}
        for g in self.groups:
            mask = self.sensitive == g
            tn, fp, fn, tp = self._confusion_stats(mask)
            tpr_dict[g] = round(tp / (tp + fn) if (tp + fn) > 0 else 0, 4)
            fpr_dict[g] = round(fp / (fp + tn) if (fp + tn) > 0 else 0, 4)

        tpr_disp = max(tpr_dict.values()) - min(tpr_dict.values())
        fpr_disp = max(fpr_dict.values()) - min(fpr_dict.values())
        return {
            "metric": "Equalized Odds",
            "tpr_per_group": tpr_dict,
            "fpr_per_group": fpr_dict,
            "tpr_disparity": round(tpr_disp, 4),
            "fpr_disparity": round(fpr_disp, 4),
            "passed": tpr_disp < 0.1 and fpr_disp < 0.1,
            "explanation": f"True Positive Rate differs by {tpr_disp:.2%}, False Positive Rate by {fpr_disp:.2%}."
        }

    def equal_opportunity(self):
        tpr_dict = {}
        for g in self.groups:
            mask = self.sensitive == g
            tn, fp, fn, tp = self._confusion_stats(mask)
            tpr_dict[g] = round(tp / (tp + fn) if (tp + fn) > 0 else 0, 4)
        disparity = max(tpr_dict.values()) - min(tpr_dict.values())
        return {
            "metric": "Equal Opportunity",
            "tpr_per_group": tpr_dict,
            "disparity": round(disparity, 4),
            "passed": disparity < 0.1,
            "explanation": f"Recall differs by {disparity:.2%} — some groups are less likely to be correctly diagnosed."
        }

    def per_group_accuracy(self):
        acc_dict, f1_dict, size_dict = {}, {}, {}
        for g in self.groups:
            mask = self.sensitive == g
            yt, yp = self.y_true[mask], self.y_pred[mask]
            if len(yt) > 0:
                acc_dict[g]  = round(accuracy_score(yt, yp), 4)
                f1_dict[g]   = round(f1_score(yt, yp, zero_division=0), 4)
                size_dict[g] = int(mask.sum())
        disparity = max(acc_dict.values()) - min(acc_dict.values())
        return {
            "metric": "Per-Group Accuracy",
            "accuracy": acc_dict,
            "f1_score": f1_dict,
            "group_sizes": size_dict,
            "disparity": round(disparity, 4),
            "passed": disparity < 0.05,
            "explanation": f"Accuracy varies by {disparity:.2%} across {self.group_name} groups."
        }

    def run_all(self):
        return {
            "demographic_parity": self.demographic_parity(),
            "equalized_odds":     self.equalized_odds(),
            "equal_opportunity":  self.equal_opportunity(),
            "per_group_accuracy": self.per_group_accuracy(),
        }


# ═══════════════════════════════════════════════════════
#  Fairness Score (0-100)
# ═══════════════════════════════════════════════════════
def compute_fairness_score(results: Dict) -> Dict:
    scores = []
    dp = results.get("demographic_parity", {})
    if "disparity" in dp:
        scores.append(max(0, 100 - dp["disparity"] * 500))
    eo = results.get("equalized_odds", {})
    if "tpr_disparity" in eo:
        scores.append(max(0, 100 - eo["tpr_disparity"] * 500))
    pg = results.get("per_group_accuracy", {})
    if "disparity" in pg:
        scores.append(max(0, 100 - pg["disparity"] * 1000))

    overall = round(np.mean(scores), 1) if scores else 0.0
    if overall >= 80:
        verdict, color = "FAIR", "green"
    elif overall >= 60:
        verdict, color = "MODERATE BIAS", "orange"
    else:
        verdict, color = "SIGNIFICANT BIAS", "red"

    return {"score": overall, "verdict": verdict, "color": color, "out_of": 100}


# ═══════════════════════════════════════════════════════
#  Main Runner
# ═══════════════════════════════════════════════════════
def run_fairness_analysis(heart_path: str, diabetes_path: str, model_type: str = "random_forest") -> Dict:
    all_results = {}

    # Heart Disease
    print("\n=== HEART DISEASE ANALYSIS ===")
    df_hd, df_hd_enc, meta_hd = load_heart_disease(heart_path)
    model_hd, scaler_hd, _, y_test_hd, y_pred_hd, y_prob_hd, idx_test_hd = train_model(
        df_hd_enc, meta_hd["feature_cols"], meta_hd["target_col"], model_type
    )
    hd_results = {}
    for sens_col in ["sex", "age_group", "dataset"]:
        if sens_col in df_hd.columns:
            sens_vals = df_hd.iloc[idx_test_hd][sens_col].values
            analyzer  = FairnessAnalyzer(y_test_hd, y_pred_hd, y_prob_hd, sens_vals, sens_col)
            hd_results[sens_col] = analyzer.run_all()
            hd_results[sens_col]["fairness_score"] = compute_fairness_score(hd_results[sens_col])
            score = hd_results[sens_col]["fairness_score"]
            print(f"{sens_col}: {score['score']}/100 — {score['verdict']}")

    all_results["heart_disease"] = {
        "meta": meta_hd, "results": hd_results,
        "model": model_hd, "scaler": scaler_hd,
        "test_data": {"y_true": y_test_hd, "y_pred": y_pred_hd, "y_prob": y_prob_hd, "idx": idx_test_hd},
        "df_original": df_hd, "df_encoded": df_hd_enc,
    }

    # Diabetes
    print("\n=== DIABETES ANALYSIS ===")
    df_db, df_db_enc, meta_db = load_diabetes(diabetes_path)
    model_db, scaler_db, _, y_test_db, y_pred_db, y_prob_db, idx_test_db = train_model(
        df_db_enc, meta_db["feature_cols"], meta_db["target_col"], model_type
    )
    db_results = {}
    for sens_col in ["age_group", "pregnancies_group"]:
        if sens_col in df_db.columns:
            sens_vals = df_db.iloc[idx_test_db][sens_col].values
            analyzer  = FairnessAnalyzer(y_test_db, y_pred_db, y_prob_db, sens_vals, sens_col)
            db_results[sens_col] = analyzer.run_all()
            db_results[sens_col]["fairness_score"] = compute_fairness_score(db_results[sens_col])
            score = db_results[sens_col]["fairness_score"]
            print(f"{sens_col}: {score['score']}/100 — {score['verdict']}")

    all_results["diabetes"] = {
        "meta": meta_db, "results": db_results,
        "model": model_db, "scaler": scaler_db,
        "test_data": {"y_true": y_test_db, "y_pred": y_pred_db, "y_prob": y_prob_db, "idx": idx_test_db},
        "df_original": df_db, "df_encoded": df_db_enc,
    }

    return all_results


if __name__ == "__main__":
    results = run_fairness_analysis("heart_disease_uci.csv", "diabetes.csv")
    print("\nDone. Datasets analyzed:", list(results.keys()))
