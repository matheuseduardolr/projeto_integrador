"""Caracterização dos núcleos — saída de negócio do pipeline."""

from __future__ import annotations

import pandas as pd

from projeto_integrador.config import K_CLUSTERS


def caracterizar_nucleos(segmentados: pd.DataFrame, k: int = K_CLUSTERS) -> pd.DataFrame:
    if segmentados.empty or "cluster" not in segmentados.columns:
        return pd.DataFrame(
            {
                "cluster": list(range(k)),
                "n_clientes": [0] * k,
                "pct_clientes_risco": [0.0] * k,
                "churn_rate": [float("nan")] * k,
                "tenure_medio": [float("nan")] * k,
                "fatura_media": [float("nan")] * k,
                "receita_mensal_total": [0.0] * k,
                "receita_anual_projetada": [0.0] * k,
                "pct_receita_exposta": [0.0] * k,
            }
        )
    n_risco = len(segmentados)
    receita_total = float(segmentados["MonthlyCharges_orig"].sum())
    linhas = []
    for cluster_id in range(k):
        bloco = segmentados[segmentados["cluster"] == cluster_id]
        n = len(bloco)
        receita = float(bloco["MonthlyCharges_orig"].sum()) if n else 0.0
        churn = float(bloco["Churn"].mean() * 100) if n and "Churn" in bloco else float("nan")
        linhas.append(
            {
                "cluster": cluster_id,
                "n_clientes": n,
                "pct_clientes_risco": (n / n_risco * 100) if n_risco else 0.0,
                "churn_rate": churn,
                "tenure_medio": float(bloco["tenure_orig"].mean()) if n else float("nan"),
                "fatura_media": float(bloco["MonthlyCharges_orig"].mean()) if n else float("nan"),
                "receita_mensal_total": receita,
                "receita_anual_projetada": receita * 12,
                "pct_receita_exposta": (receita / receita_total * 100) if receita_total else 0.0,
            }
        )
    return pd.DataFrame(linhas)
