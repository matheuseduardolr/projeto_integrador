"""Validação de schema e checagens de fumaça do pipeline."""

from __future__ import annotations

import pandas as pd

from projeto_integrador.config import FEATURE_COLS, K_CLUSTERS, RAW_COLUMNS


class SchemaError(ValueError):
    """Base consolidada ou bruta fora do contrato esperado."""


def validar_bruto(df: pd.DataFrame, exigir_churn: bool = False) -> None:
    obrigatorias = [c for c in RAW_COLUMNS if c != "Churn"]
    if exigir_churn:
        obrigatorias = list(RAW_COLUMNS)
    faltando = [c for c in obrigatorias if c not in df.columns]
    if faltando:
        raise SchemaError(f"Colunas ausentes na base bruta: {faltando}")
    if df.empty:
        raise SchemaError("Base bruta vazia.")
    if "customerID" in df.columns and df["customerID"].nunique() != len(df):
        raise SchemaError("customerID não é único.")


def validar_consolidada(df: pd.DataFrame) -> None:
    validar_bruto(df, exigir_churn=True)
    if df["TotalCharges"].isnull().any():
        raise SchemaError("TotalCharges contém nulos após a consolidação.")
    if not df["Churn"].isin([0, 1]).all():
        raise SchemaError("Churn deve ser 0 ou 1 na base consolidada.")


def validar_matriz_features(X: pd.DataFrame, cols: list[str] | None = None) -> None:
    esperadas = cols or FEATURE_COLS
    faltando = [c for c in esperadas if c not in X.columns]
    if faltando:
        raise SchemaError(f"Features ausentes na matriz de scoring: {faltando}")
    extra_nao_num = X[esperadas].select_dtypes(exclude="number").columns.tolist()
    if extra_nao_num:
        raise SchemaError(f"Features não numéricas: {extra_nao_num}")


def validar_saida_score(segmentados: pd.DataFrame, k: int = K_CLUSTERS) -> dict:
    """Checagens de fumaça da inferência encadeada. Não substitui monitoramento."""
    if segmentados.empty:
        raise SchemaError("Nenhum cliente sinalizado — verifique o limiar ou a base.")
    obrigatorias = [
        "customerID",
        "prob_churn",
        "em_risco",
        "cluster",
        "MonthlyCharges_orig",
    ]
    faltando = [c for c in obrigatorias if c not in segmentados.columns]
    if faltando:
        raise SchemaError(f"Saída de score incompleta: {faltando}")
    if not segmentados["em_risco"].all():
        raise SchemaError("A tabela de sinalizados contém clientes fora do limiar.")
    n_clusters = segmentados["cluster"].nunique()
    vazios = [i for i in range(k) if i not in set(segmentados["cluster"].unique())]
    return {
        "n_sinalizados": int(len(segmentados)),
        "n_clusters_presentes": int(n_clusters),
        "clusters_vazios": vazios,
        "prob_media": float(segmentados["prob_churn"].mean()),
        "receita_mensal_exposta": float(segmentados["MonthlyCharges_orig"].sum()),
    }
