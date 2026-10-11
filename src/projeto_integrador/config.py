"""Constantes e caminhos do pipeline operacional."""

from __future__ import annotations

import os
from pathlib import Path

RANDOM_STATE = 42
K_CLUSTERS = 3
THRESHOLD_DEFAULT = 0.795
PESO_F1 = 0.60
PESO_ACURACIA = 0.40
TEST_SIZE = 0.20

# Receita operacional do XGBoost escolhido no laboratório (Optuna fica no NB04).
XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "eval_metric": "logloss",
    "tree_method": "hist",
    "n_jobs": 1,
    "random_state": RANDOM_STATE,
}

RAW_CSV_NAME = "WA_Fn-UseC_-Telco-Customer-Churn.csv"

RAW_COLUMNS = [
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
]

COLS_SERVICOS_LIMPEZA = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "MultipleLines",
]

SERVICOS_ADICIONAIS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

COLS_BINARIAS = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "PaperlessBilling",
]

COLS_OHE = ["InternetService", "Contract", "PaymentMethod", "tenure_grupo"]

COLS_NORMALIZAR = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "charges_por_mes",
    "charges_desvio_contrato",
    "qtd_servicos",
    "qtd_servicos_protecao",
    "score_risco_estendido",
    "tenure_x_contrato_mensal",
    "charges_x_contrato_mensal",
]

FEATURE_COLS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "PaperlessBilling",
    "MonthlyCharges",
    "TotalCharges",
    "is_novo_cliente",
    "charges_por_mes",
    "charges_desvio_contrato",
    "qtd_servicos",
    "sem_servicos_adicionais",
    "fibra_sem_suporte",
    "qtd_servicos_protecao",
    "sem_protecao",
    "score_risco_estendido",
    "contrato_mensal_novo",
    "contrato_mensal_fibra",
    "tenure_x_contrato_mensal",
    "charges_x_contrato_mensal",
    "boleto_x_contrato_mensal",
    "fibra_x_sem_protecao",
    "tenure_curto_fibra",
    "alta_cobranca_relativa",
    "InternetService_DSL",
    "InternetService_Fiber optic",
    "InternetService_No",
    "Contract_Month-to-month",
    "Contract_One year",
    "Contract_Two year",
    "PaymentMethod_Bank transfer (automatic)",
    "PaymentMethod_Credit card (automatic)",
    "PaymentMethod_Electronic check",
    "PaymentMethod_Mailed check",
    "tenure_grupo_0-12m",
    "tenure_grupo_13-24m",
    "tenure_grupo_25-48m",
    "tenure_grupo_49-72m",
]


def project_root() -> Path:
    env = os.environ.get("PROJETO_ROOT")
    if env:
        return Path(env).resolve()
    return Path(__file__).resolve().parents[2]


def paths(root: Path | None = None) -> dict[str, Path]:
    base = root or project_root()
    return {
        "root": base,
        "raw": base / "data" / "raw",
        "interm": base / "data" / "interm",
        "processed": base / "data" / "processed",
        "models": base / "models",
        "raw_csv": base / "data" / "raw" / RAW_CSV_NAME,
        "base_consolidada": base / "data" / "interm" / "base_consolidada.parquet",
        "clientes_em_risco": base / "data" / "processed" / "clientes_em_risco.parquet",
        "clientes_segmentados": base / "data" / "processed" / "clientes_segmentados.parquet",
        "relatorio_nucleos": base / "data" / "processed" / "relatorio_nucleos.parquet",
        "modelo_churn": base / "models" / "modelo_churn.pkl",
        "kmeans": base / "models" / "kmeans_segmentacao.pkl",
        "scaler": base / "models" / "scaler.pkl",
        "encoders": base / "models" / "encoders_binarios.pkl",
        "feature_cols": base / "models" / "feature_cols.pkl",
        "contract_means": base / "models" / "contract_means.json",
        "production": base / "models" / "production.json",
    }
