"""Consolidação da base bruta — equivalente operacional do NB01."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from projeto_integrador.config import COLS_SERVICOS_LIMPEZA
from projeto_integrador.schemas import validar_bruto, validar_consolidada


def consolidar(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Converte tipos, trata nulos de tenure 0 e padroniza categorias de serviço."""
    validar_bruto(df_raw, exigir_churn=False)
    df = df_raw.copy()
    df["TotalCharges"] = (
        pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0).astype(float)
    )
    if "Churn" in df.columns:
        if df["Churn"].dtype == object or str(df["Churn"].dtype).startswith("str"):
            df["Churn"] = (df["Churn"] == "Yes").astype(int)
        else:
            df["Churn"] = df["Churn"].astype(int)
    else:
        df["Churn"] = -1
    for col in COLS_SERVICOS_LIMPEZA:
        df[col] = df[col].replace(
            {"No internet service": "No", "No phone service": "No"}
        )
    if (df["Churn"] >= 0).all():
        validar_consolidada(df)
    elif df["customerID"].nunique() != len(df) or df["TotalCharges"].isnull().any():
        raise ValueError("Falha na consolidação da base sem rótulo.")
    return df


def consolidar_csv(origem: Path, destino: Path | None = None) -> pd.DataFrame:
    df = consolidar(pd.read_csv(origem))
    if destino is not None:
        destino.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(destino, engine="pyarrow", index=False)
    return df
