from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from projeto_integrador.config import (  # noqa: E402
    COLS_BINARIAS,
    COLS_NORMALIZAR,
    FEATURE_COLS,
    K_CLUSTERS,
)
from projeto_integrador.extract import consolidar  # noqa: E402
from projeto_integrador.features import engenharia_features  # noqa: E402


def _linha(i: int) -> dict:
    contratos = ["Month-to-month", "One year", "Two year"]
    internet = ["Fiber optic", "DSL", "No"]
    pagamento = [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    return {
        "customerID": f"id-{i:04d}",
        "gender": "Female" if i % 2 == 0 else "Male",
        "SeniorCitizen": int(i % 5 == 0),
        "Partner": "Yes" if i % 3 == 0 else "No",
        "Dependents": "No",
        "tenure": (i * 7) % 72,
        "PhoneService": "Yes",
        "MultipleLines": "No" if i % 4 else "Yes",
        "InternetService": internet[i % 3],
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes" if i % 2 else "No",
        "DeviceProtection": "No",
        "TechSupport": "No" if i % 2 else "Yes",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": contratos[i % 3],
        "PaperlessBilling": "Yes",
        "PaymentMethod": pagamento[i % 4],
        "MonthlyCharges": 30 + (i % 10) * 8,
        "TotalCharges": str(100 + i * 15),
        "Churn": "Yes" if i % 4 == 0 else "No",
    }


@pytest.fixture
def bruto() -> pd.DataFrame:
    return pd.DataFrame([_linha(i) for i in range(24)])


@pytest.fixture
def consolidada(bruto: pd.DataFrame) -> pd.DataFrame:
    return consolidar(bruto)


class _SempreRisco:
    def predict_proba(self, X):
        p = np.full(len(X), 0.91)
        return np.column_stack([1 - p, p])


@pytest.fixture
def pasta_modelos(tmp_path: Path, consolidada: pd.DataFrame) -> Path:
    df = engenharia_features(consolidada)
    encoders = {}
    for col in COLS_BINARIAS:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    df = pd.get_dummies(
        df, columns=["InternetService", "Contract", "PaymentMethod", "tenure_grupo"]
    )
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0
    X = df[FEATURE_COLS].copy()
    cols_norm = [c for c in COLS_NORMALIZAR if c in X.columns]
    scaler = StandardScaler()
    X[cols_norm] = scaler.fit_transform(X[cols_norm].astype(float))

    kmeans = KMeans(n_clusters=K_CLUSTERS, random_state=42, n_init=10)
    kmeans.fit(X)

    pasta = tmp_path / "models"
    pasta.mkdir()
    joblib.dump(
        {
            "modelo": _SempreRisco(),
            "threshold": 0.795,
            "feature_cols": FEATURE_COLS,
            "nome_modelo": "dummy",
        },
        pasta / "modelo_churn.pkl",
    )
    joblib.dump({"modelo": kmeans, "n_clusters": K_CLUSTERS}, pasta / "kmeans_segmentacao.pkl")
    joblib.dump(encoders, pasta / "encoders_binarios.pkl")
    joblib.dump(scaler, pasta / "scaler.pkl")
    return pasta
