"""Inferência encadeada: classificador → filtro do limiar → K-Means."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from projeto_integrador.config import FEATURE_COLS, K_CLUSTERS, THRESHOLD_DEFAULT, paths
from projeto_integrador.features import carregar_contract_means, preparar_features
from projeto_integrador.registry import resolver_models_dir
from projeto_integrador.report import caracterizar_nucleos
from projeto_integrador.schemas import validar_saida_score
from projeto_integrador.tracking import iniciar_run, registrar_metrics, registrar_params


def carregar_artefatos(models_dir: Path, exigir_production: bool = False) -> dict:
    models_dir = resolver_models_dir(models_dir, exigir_production=exigir_production)
    churn = joblib.load(models_dir / "modelo_churn.pkl")
    kmeans = joblib.load(models_dir / "kmeans_segmentacao.pkl")
    return {
        "churn": churn,
        "kmeans": kmeans,
        "encoders": joblib.load(models_dir / "encoders_binarios.pkl"),
        "scaler": joblib.load(models_dir / "scaler.pkl"),
        "feature_cols": churn.get("feature_cols") or FEATURE_COLS,
        "threshold": float(churn.get("threshold", THRESHOLD_DEFAULT)),
        "modelo": churn["modelo"],
        "clusterer": kmeans["modelo"] if isinstance(kmeans, dict) else kmeans,
        "k": int(kmeans.get("n_clusters", K_CLUSTERS)) if isinstance(kmeans, dict) else K_CLUSTERS,
    }


def pontuar(
    consolidada: pd.DataFrame,
    artefatos: dict,
    contract_means: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    X, meta = preparar_features(
        consolidada,
        encoders=artefatos["encoders"],
        scaler=artefatos["scaler"],
        cols_features=artefatos["feature_cols"],
        contract_means=contract_means,
    )
    probas = artefatos["modelo"].predict_proba(X)[:, 1]
    scoring = meta.copy()
    scoring["prob_churn"] = probas
    scoring["em_risco"] = scoring["prob_churn"] >= artefatos["threshold"]
    for col in artefatos["feature_cols"]:
        scoring[col] = X[col].values

    sinalizados = scoring[scoring["em_risco"]].copy().reset_index(drop=True)
    if sinalizados.empty:
        relatorio = caracterizar_nucleos(sinalizados, k=artefatos["k"])
        return scoring, relatorio, {"n_sinalizados": 0, "n_clusters_presentes": 0}

    X_risco = sinalizados[artefatos["feature_cols"]]
    sinalizados["cluster"] = artefatos["clusterer"].predict(X_risco)
    fumaça = validar_saida_score(sinalizados, k=artefatos["k"])
    relatorio = caracterizar_nucleos(sinalizados, k=artefatos["k"])
    fumaça["receita_mensal_exposta"] = float(relatorio["receita_mensal_total"].sum())
    return sinalizados, relatorio, fumaça


def executar_lote(
    entrada: Path,
    saida_segmentados: Path,
    saida_relatorio: Path,
    models_dir: Path | None = None,
    registrar: bool = True,
) -> dict:
    dirs = paths()
    models_dir = models_dir or dirs["models"]
    consolidada = pd.read_parquet(entrada)
    artefatos = carregar_artefatos(models_dir, exigir_production=True)
    contract_means = carregar_contract_means(models_dir / "contract_means.json")

    with iniciar_run(
        "score_lote",
        tags={"pipeline": "churn_kmeans", "etapa": "inferencia"},
    ):
        if registrar:
            registrar_params(
                {
                    "threshold": artefatos["threshold"],
                    "k": artefatos["k"],
                    "n_features": len(artefatos["feature_cols"]),
                    "n_base": len(consolidada),
                }
            )
        segmentados, relatorio, fumaca = pontuar(
            consolidada, artefatos, contract_means=contract_means
        )
        if registrar:
            registrar_metrics(fumaca)

    saida_segmentados.parent.mkdir(parents=True, exist_ok=True)
    segmentados.to_parquet(saida_segmentados, index=False)
    relatorio.to_parquet(saida_relatorio, index=False)
    return fumaca
