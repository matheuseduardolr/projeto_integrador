"""Engenharia de atributos e preparação para scoring — NB03 / NB04."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from projeto_integrador.config import (
    COLS_BINARIAS,
    COLS_NORMALIZAR,
    COLS_OHE,
    FEATURE_COLS,
    RANDOM_STATE,
    SERVICOS_ADICIONAIS,
    TEST_SIZE,
)
from projeto_integrador.schemas import validar_matriz_features


def engenharia_features(
    df: pd.DataFrame,
    contract_means: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Replica a engenharia do Notebook 03.

    Se ``contract_means`` vier do treino, o desvio de fatura não depende
    da composição do lote atual — comportamento desejado em produção.
    Sem o artefato, cai no groupby do lote (como nos notebooks).
    """
    df = df.copy()
    df["tenure_grupo"] = pd.cut(
        df["tenure"],
        bins=[0, 12, 24, 48, 72],
        labels=["0-12m", "13-24m", "25-48m", "49-72m"],
        include_lowest=True,
    )
    df["is_novo_cliente"] = (df["tenure"] <= 12).astype(int)
    df["charges_por_mes"] = np.where(
        df["tenure"] > 0, df["TotalCharges"] / df["tenure"], df["MonthlyCharges"]
    )
    if contract_means:
        media_por_contrato = df["Contract"].map(contract_means).astype(float)
        media_por_contrato = media_por_contrato.fillna(df["MonthlyCharges"].mean())
    else:
        media_por_contrato = df.groupby("Contract")["MonthlyCharges"].transform("mean")
    df["charges_desvio_contrato"] = df["MonthlyCharges"] - media_por_contrato
    df["qtd_servicos"] = df[SERVICOS_ADICIONAIS].apply(
        lambda row: (row == "Yes").sum(), axis=1
    )
    df["sem_servicos_adicionais"] = (df["qtd_servicos"] == 0).astype(int)
    df["is_sem_suporte"] = (df["TechSupport"] == "No").astype(int)
    df["fibra_sem_suporte"] = (
        (df["InternetService"] == "Fiber optic") & (df["TechSupport"] == "No")
    ).astype(int)
    df["is_sem_seguranca_online"] = (df["OnlineSecurity"] == "No").astype(int)
    df["qtd_servicos_protecao"] = df[
        ["OnlineSecurity", "OnlineBackup", "DeviceProtection"]
    ].apply(lambda row: (row == "Yes").sum(), axis=1)
    df["sem_protecao"] = (df["qtd_servicos_protecao"] == 0).astype(int)
    df["is_contrato_mensal"] = (df["Contract"] == "Month-to-month").astype(int)
    df["is_fibra_optica"] = (df["InternetService"] == "Fiber optic").astype(int)
    df["is_boleto_eletronico"] = (df["PaymentMethod"] == "Electronic check").astype(int)
    df["is_senior"] = df["SeniorCitizen"].astype(int)
    df["score_risco_base"] = (
        df["is_contrato_mensal"]
        + df["is_fibra_optica"]
        + df["is_boleto_eletronico"]
        + df["is_senior"]
    )
    df["score_risco_estendido"] = (
        df["score_risco_base"] + df["is_sem_suporte"] + df["is_sem_seguranca_online"]
    )
    df["contrato_mensal_novo"] = df["is_contrato_mensal"] * df["is_novo_cliente"]
    df["contrato_mensal_fibra"] = df["is_contrato_mensal"] * df["is_fibra_optica"]
    df["tenure_x_contrato_mensal"] = df["tenure"] * df["is_contrato_mensal"]
    df["charges_x_contrato_mensal"] = df["MonthlyCharges"] * df["is_contrato_mensal"]
    df["boleto_x_contrato_mensal"] = (
        df["is_boleto_eletronico"] * df["is_contrato_mensal"]
    )
    df["fibra_x_sem_protecao"] = df["is_fibra_optica"] * df["sem_protecao"]
    df["tenure_curto_fibra"] = (
        (df["tenure"] <= 12) & (df["InternetService"] == "Fiber optic")
    ).astype(int)
    df["alta_cobranca_relativa"] = (df["charges_desvio_contrato"] > 0).astype(int)
    return df


def carregar_contract_means(caminho: Path | None) -> dict[str, float] | None:
    if caminho is None or not caminho.exists():
        return None
    return {str(k): float(v) for k, v in json.loads(caminho.read_text()).items()}


def preparar_features(
    df_raw: pd.DataFrame,
    encoders: dict,
    scaler,
    cols_features: list[str] | None = None,
    contract_means: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aplica encoding e scaler de treino. Devolve (X, meta)."""
    cols_features = cols_features or FEATURE_COLS
    meta = df_raw[
        [
            "customerID",
            "Churn",
            "MonthlyCharges",
            "TotalCharges",
            "tenure",
            "Contract",
            "InternetService",
            "PaymentMethod",
        ]
    ].copy()
    meta.columns = [
        "customerID",
        "Churn",
        "MonthlyCharges_orig",
        "TotalCharges_orig",
        "tenure_orig",
        "Contract",
        "InternetService",
        "PaymentMethod",
    ]

    df = engenharia_features(df_raw, contract_means=contract_means)
    for col in COLS_BINARIAS:
        df[col] = encoders[col].transform(df[col].astype(str))
    df = pd.get_dummies(df, columns=COLS_OHE, drop_first=False)
    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)

    for col in cols_features:
        if col not in df.columns:
            df[col] = 0

    X = df[cols_features].copy()
    cols_norm = [c for c in COLS_NORMALIZAR if c in X.columns]
    X[cols_norm] = scaler.transform(X[cols_norm].astype(float))
    validar_matriz_features(X, cols_features)
    return X, meta


def _transformar_binarias(series: pd.Series, encoder: LabelEncoder) -> pd.Series:
    valores = series.astype(str)
    conhecidos = set(encoder.classes_)
    fallback = encoder.classes_[0]
    return encoder.transform(valores.where(valores.isin(conhecidos), fallback))


def _para_matriz(eng: pd.DataFrame, cols_features: list[str]) -> pd.DataFrame:
    eng = pd.get_dummies(eng, columns=COLS_OHE, drop_first=False)
    bool_cols = eng.select_dtypes(include="bool").columns
    eng[bool_cols] = eng[bool_cols].astype(int)
    for col in cols_features:
        if col not in eng.columns:
            eng[col] = 0
    return eng[cols_features].copy()


def preparar_treino(
    consolidada: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> dict:
    """Split estratificado, médias de contrato no treino, encoders e scaler."""
    df = consolidada.reset_index(drop=True)
    if (df["Churn"] < 0).any() or not df["Churn"].isin([0, 1]).all():
        raise ValueError("Treino exige Churn rotulado em 0/1.")

    treino_raw, teste_raw = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df["Churn"]
    )
    treino_raw = treino_raw.reset_index(drop=True)
    teste_raw = teste_raw.reset_index(drop=True)
    contract_means = (
        treino_raw.groupby("Contract")["MonthlyCharges"].mean().astype(float).to_dict()
    )

    treino_eng = engenharia_features(treino_raw, contract_means=contract_means)
    teste_eng = engenharia_features(teste_raw, contract_means=contract_means)

    encoders: dict[str, LabelEncoder] = {}
    for col in COLS_BINARIAS:
        le = LabelEncoder()
        treino_eng[col] = le.fit_transform(treino_eng[col].astype(str))
        teste_eng[col] = _transformar_binarias(teste_eng[col], le)
        encoders[col] = le

    X_treino = _para_matriz(treino_eng, FEATURE_COLS)
    X_teste = _para_matriz(teste_eng, FEATURE_COLS)
    cols_norm = [c for c in COLS_NORMALIZAR if c in X_treino.columns]
    scaler = StandardScaler()
    X_treino = X_treino.copy()
    X_teste = X_teste.copy()
    X_treino[cols_norm] = scaler.fit_transform(X_treino[cols_norm].astype(float))
    X_teste[cols_norm] = scaler.transform(X_teste[cols_norm].astype(float))
    validar_matriz_features(X_treino)
    validar_matriz_features(X_teste)

    return {
        "X_treino": X_treino,
        "y_treino": treino_raw["Churn"].astype(int),
        "X_teste": X_teste,
        "y_teste": teste_raw["Churn"].astype(int),
        "treino_raw": treino_raw,
        "teste_raw": teste_raw,
        "encoders": encoders,
        "scaler": scaler,
        "contract_means": contract_means,
        "feature_cols": list(FEATURE_COLS),
    }
