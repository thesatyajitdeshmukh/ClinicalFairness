"""
preprocessing.py — Data Preprocessing Pipeline
================================================
Handles any medical dataset uploaded by user.
- Auto detects sensitive attributes (gender, race, age)
- Handles missing values
- Encodes categorical variables
- Generates SBERT embeddings for text columns
- Prepares data for all bias detection modules
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sentence_transformers import SentenceTransformer
from typing import Tuple, List, Dict, Optional
import warnings
warnings.filterwarnings("ignore")


# ── Sensitive attribute keywords to auto-detect
SENSITIVE_KEYWORDS = {
    "gender":    ["gender", "sex", "male", "female"],
    "race":      ["race", "ethnicity", "ethnic", "nationality"],
    "age":       ["age", "dob", "birth"],
    "income":    ["income", "salary", "socioeconomic"],
    "insurance": ["insurance", "payer", "coverage"],
}

# ── Target/outcome column keywords
TARGET_KEYWORDS = [
    "diagnosis", "outcome", "result", "label",
    "target", "disease", "condition", "prediction",
    "readmission", "mortality", "survival"
]


class MedicalDataPreprocessor:
    """
    Preprocessing pipeline for medical datasets.
    Supports CSV, XLSX uploads from Gradio.
    """

    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self.scaler          = StandardScaler()
        self.label_encoders  = {}
        self.embedding_model = None
        self.embedding_name  = embedding_model
        self.sensitive_cols  = []
        self.target_col      = None
        self.feature_cols    = []
        self.text_cols       = []

    def _load_embedding_model(self):
        if self.embedding_model is None:
            print("Loading SBERT model...")
            self.embedding_model = SentenceTransformer(self.embedding_name)

    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load CSV or XLSX file."""
        if file_path.endswith(".csv"):
            df = pd.read_csv(file_path)
        elif file_path.endswith((".xlsx", ".xls")):
            df = pd.read_excel(file_path)
        else:
            raise ValueError("Only CSV and XLSX files supported.")
        print(f"Loaded dataset: {df.shape[0]} rows × {df.shape[1]} columns")
        return df

    def auto_detect_sensitive(self, df: pd.DataFrame) -> List[str]:
        """Auto detect sensitive attribute columns."""
        detected = []
        for col in df.columns:
            col_lower = col.lower()
            for attr, keywords in SENSITIVE_KEYWORDS.items():
                if any(kw in col_lower for kw in keywords):
                    detected.append(col)
                    break
        return detected

    def auto_detect_target(self, df: pd.DataFrame) -> Optional[str]:
        """Auto detect target/outcome column."""
        for col in df.columns:
            col_lower = col.lower()
            if any(kw in col_lower for kw in TARGET_KEYWORDS):
                return col
        # Fallback — last column
        return df.columns[-1]

    def handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values per column type."""
        for col in df.columns:
            if df[col].dtype in ["float64", "int64"]:
                df[col].fillna(df[col].median(), inplace=True)
            else:
                df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown", inplace=True)
        return df

    def encode_categoricals(self, df: pd.DataFrame, exclude: List[str] = []) -> pd.DataFrame:
        """Label encode categorical columns."""
        for col in df.columns:
            if col in exclude:
                continue
            if df[col].dtype == "object":
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
        return df

    def detect_text_columns(self, df: pd.DataFrame) -> List[str]:
        """Detect free text columns (long strings)."""
        text_cols = []
        for col in df.columns:
            if df[col].dtype == "object":
                avg_len = df[col].dropna().astype(str).apply(len).mean()
                if avg_len > 50:
                    text_cols.append(col)
        return text_cols

    def generate_embeddings(self, df: pd.DataFrame, text_cols: List[str]) -> np.ndarray:
        """Generate SBERT embeddings for text columns."""
        self._load_embedding_model()
        all_embeddings = []
        for col in text_cols:
            texts = df[col].fillna("").astype(str).tolist()
            embeddings = self.embedding_model.encode(
                texts, batch_size=32, show_progress_bar=True
            )
            all_embeddings.append(embeddings)
        if all_embeddings:
            return np.hstack(all_embeddings)
        return np.array([])

    def preprocess(
        self,
        file_path: str,
        sensitive_cols: Optional[List[str]] = None,
        target_col:     Optional[str] = None,
    ) -> Dict:
        """
        Full preprocessing pipeline.
        Returns dict with processed data and metadata.
        """
        # ── Load
        df = self.load_data(file_path)
        df_original = df.copy()

        # ── Auto detect columns
        self.sensitive_cols = sensitive_cols or self.auto_detect_sensitive(df)
        self.target_col     = target_col     or self.auto_detect_target(df)
        self.text_cols      = self.detect_text_columns(df)

        print(f"Sensitive attributes: {self.sensitive_cols}")
        print(f"Target column: {self.target_col}")
        print(f"Text columns: {self.text_cols}")

        # ── Handle missing
        df = self.handle_missing(df)

        # ── Feature columns (exclude sensitive + target)
        self.feature_cols = [
            c for c in df.columns
            if c not in self.sensitive_cols
            and c != self.target_col
            and c not in self.text_cols
        ]

        # ── Encode categoricals
        df_encoded = self.encode_categoricals(df.copy())

        # ── Scale numerical features
        X = df_encoded[self.feature_cols].values
        X_scaled = self.scaler.fit_transform(X)

        # ── Generate embeddings for text columns
        embeddings = np.array([])
        if self.text_cols:
            embeddings = self.generate_embeddings(df, self.text_cols)

        # ── Combine features
        if embeddings.size > 0:
            X_combined = np.hstack([X_scaled, embeddings])
        else:
            X_combined = X_scaled

        # ── Sensitive attributes (keep original for fairness analysis)
        sensitive_data = {}
        for col in self.sensitive_cols:
            sensitive_data[col] = df_original[col].values

        # ── Target
        y = df_encoded[self.target_col].values if self.target_col in df_encoded.columns else None

        return {
            "df_original":    df_original,
            "df_encoded":     df_encoded,
            "X":              X_combined,
            "X_scaled":       X_scaled,
            "y":              y,
            "sensitive_data": sensitive_data,
            "sensitive_cols": self.sensitive_cols,
            "target_col":     self.target_col,
            "feature_cols":   self.feature_cols,
            "text_cols":      self.text_cols,
            "embeddings":     embeddings,
            "n_samples":      len(df),
            "n_features":     X_combined.shape[1],
        }


def get_dataset_summary(data: Dict) -> str:
    """Generate a text summary of the dataset for Gemini."""
    df = data["df_original"]
    summary = f"""
Dataset Summary:
- Total samples: {data['n_samples']}
- Features: {len(data['feature_cols'])}
- Target column: {data['target_col']}
- Sensitive attributes detected: {', '.join(data['sensitive_cols']) if data['sensitive_cols'] else 'None auto-detected'}
- Missing values handled: Yes
- Text columns: {', '.join(data['text_cols']) if data['text_cols'] else 'None'}

Target distribution:
{df[data['target_col']].value_counts().to_string() if data['target_col'] in df.columns else 'N/A'}
    """
    return summary.strip()
