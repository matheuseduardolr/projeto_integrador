"""Retreino operacional do XGBoost e do K-Means já escolhidos no laboratório."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)

from projeto_integrador.config import (
    FEATURE_COLS,
    K_CLUSTERS,
    PESO_ACURACIA,
    PESO_F1,
    RANDOM_STATE,
    XGB_PARAMS,
)
from projeto_integrador.features import preparar_features, preparar_treino
from projeto_integrador.registry import escrever_production
from projeto_integrador.tracking import (
    iniciar_run,
    registrar_artefato,
    registrar_metrics,
    registrar_params,
)


def score_composto(f1: float, acuracia: float) -> float:
    return PESO_F1 * f1 + PESO_ACURACIA * acuracia


def encontrar_melhor_threshold(y_true, probas) -> dict:
    melhor = None
    y_true = np.asarray(y_true)
    for t in np.arange(0.15, 0.85, 0.005):
        pred = (probas >= t).astype(int)
        fb = fbeta_score(y_true, pred, beta=0.5, zero_division=0)
        f1 = f1_score(y_true, pred, zero_division=0)
        acc = accuracy_score(y_true, pred)
        combo = score_composto(fb, acc)
        if melhor is None or combo > melhor["score_composto"]:
            melhor = {
                "threshold": float(t),
                "f1": float(f1),
                "fbeta": float(fb),
                "accuracy": float(acc),
                "precision": float(precision_score(y_true, pred, zero_division=0)),
                "recall": float(recall_score(y_true, pred, zero_division=0)),
                "score_composto": float(combo),
            }
    return melhor


def metricas_holdout(y_true, y_pred, probas) -> dict:
    return {
        "auc_roc": float(roc_auc_score(y_true, probas)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def treinar_churn(prep: dict) -> dict:
    y_treino = prep["y_treino"]
    n_pos = int((y_treino == 1).sum())
    n_neg = int((y_treino == 0).sum())
    params = dict(XGB_PARAMS)
    params["scale_pos_weight"] = (n_neg / n_pos) if n_pos else 1.0
    modelo = xgb.XGBClassifier(**params)
    modelo.fit(prep["X_treino"], y_treino)

    probas_treino = modelo.predict_proba(prep["X_treino"])[:, 1]
    corte = encontrar_melhor_threshold(y_treino, probas_treino)
    probas_teste = modelo.predict_proba(prep["X_teste"])[:, 1]
    pred_teste = (probas_teste >= corte["threshold"]).astype(int)
    holdout = metricas_holdout(prep["y_teste"], pred_teste, probas_teste)
    return {
        "modelo": modelo,
        "threshold": corte["threshold"],
        "corte_treino": corte,
        "metricas_teste": holdout,
        "hiperparametros": params,
        "feature_cols": prep["feature_cols"],
        "nome_modelo": "XGBoost",
    }


def treinar_kmeans(
    consolidada: pd.DataFrame,
    churn: dict,
    prep: dict,
) -> dict:
    X, meta = preparar_features(
        consolidada,
        encoders=prep["encoders"],
        scaler=prep["scaler"],
        cols_features=churn["feature_cols"],
        contract_means=prep["contract_means"],
    )
    scoring = meta.copy()
    scoring["prob_churn"] = churn["modelo"].predict_proba(X)[:, 1]
    scoring["em_risco"] = scoring["prob_churn"] >= churn["threshold"]
    for col in churn["feature_cols"]:
        scoring[col] = X[col].values
    flagged = scoring[scoring["em_risco"]].copy()
    if len(flagged) < K_CLUSTERS:
        raise ValueError(
            f"Apenas {len(flagged)} sinalizados — insuficiente para K={K_CLUSTERS}."
        )
    X_risco = flagged[churn["feature_cols"]]
    kmeans = KMeans(n_clusters=K_CLUSTERS, random_state=RANDOM_STATE, n_init=10)
    labels = kmeans.fit_predict(X_risco)
    silhueta = float(silhouette_score(X_risco, labels))
    return {
        "modelo": kmeans,
        "n_clusters": K_CLUSTERS,
        "feature_cols": churn["feature_cols"],
        "random_state": RANDOM_STATE,
        "silhouette_k3": silhueta,
        "n_sinalizados": int(len(flagged)),
    }


def persistir_artefatos(models_dir: Path, prep: dict, churn: dict, kmeans: dict) -> dict:
    models_dir.mkdir(parents=True, exist_ok=True)
    artefato_churn = {
        "modelo": churn["modelo"],
        "nome_modelo": churn["nome_modelo"],
        "threshold": churn["threshold"],
        "feature_cols": churn["feature_cols"],
        "metricas_teste": churn["metricas_teste"],
        "hiperparametros": churn["hiperparametros"],
    }
    joblib.dump(artefato_churn, models_dir / "modelo_churn.pkl")
    joblib.dump(kmeans, models_dir / "kmeans_segmentacao.pkl")
    joblib.dump(prep["scaler"], models_dir / "scaler.pkl")
    joblib.dump(prep["encoders"], models_dir / "encoders_binarios.pkl")
    joblib.dump(prep["feature_cols"], models_dir / "feature_cols.pkl")
    (models_dir / "contract_means.json").write_text(
        json.dumps(prep["contract_means"], indent=2), encoding="utf-8"
    )
    manifesto = escrever_production(
        models_dir,
        {
            "threshold": churn["threshold"],
            "k": kmeans["n_clusters"],
            "n_sinalizados": kmeans["n_sinalizados"],
            "silhouette": kmeans["silhouette_k3"],
            "auc_holdout": churn["metricas_teste"]["auc_roc"],
        },
    )
    return {"production": manifesto}


def executar_treino(entrada: Path, models_dir: Path) -> dict:
    consolidada = pd.read_parquet(entrada) if entrada.suffix == ".parquet" else pd.read_csv(entrada)
    if entrada.suffix == ".csv":
        from projeto_integrador.extract import consolidar

        consolidada = consolidar(consolidada)

    with iniciar_run("train_encadeado", tags={"pipeline": "churn_kmeans", "etapa": "treino"}):
        prep = preparar_treino(consolidada)
        churn = treinar_churn(prep)
        kmeans = treinar_kmeans(consolidada, churn, prep)
        persistir_artefatos(models_dir, prep, churn, kmeans)
        registrar_params(
            {
                "threshold": churn["threshold"],
                "k": kmeans["n_clusters"],
                "n_features": len(FEATURE_COLS),
                "n_treino": len(prep["X_treino"]),
                "n_teste": len(prep["X_teste"]),
            }
        )
        registrar_metrics(
            {
                **{f"holdout_{k}": v for k, v in churn["metricas_teste"].items()},
                "n_sinalizados": kmeans["n_sinalizados"],
                "silhouette": kmeans["silhouette_k3"],
            }
        )
        for nome in (
            "modelo_churn.pkl",
            "kmeans_segmentacao.pkl",
            "production.json",
        ):
            registrar_artefato(models_dir / nome, "production")

    return {
        "threshold": churn["threshold"],
        "metricas_teste": churn["metricas_teste"],
        "n_sinalizados": kmeans["n_sinalizados"],
        "silhouette": kmeans["silhouette_k3"],
        "k": kmeans["n_clusters"],
        "models_dir": str(models_dir),
    }
