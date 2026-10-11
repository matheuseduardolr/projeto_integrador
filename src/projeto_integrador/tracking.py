"""Registro opcional de parâmetros e métricas no MLflow."""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator


def tracking_ativo() -> bool:
    flag = os.environ.get("PROJETO_MLOPS_TRACKING", "").strip().lower()
    if flag in {"0", "false", "off", "no"}:
        return False
    if flag in {"1", "true", "on", "yes"}:
        return True
    return bool(os.environ.get("MLFLOW_TRACKING_URI"))


@contextmanager
def iniciar_run(nome: str, tags: dict | None = None) -> Iterator[object | None]:
    if not tracking_ativo():
        yield None
        return
    import mlflow

    with mlflow.start_run(run_name=nome) as run:
        if tags:
            mlflow.set_tags(tags)
        yield run


def registrar_params(params: dict) -> None:
    if not tracking_ativo():
        return
    import mlflow

    mlflow.log_params({k: str(v) for k, v in params.items()})


def registrar_metrics(metrics: dict) -> None:
    if not tracking_ativo():
        return
    import mlflow

    numericos = {k: float(v) for k, v in metrics.items() if v is not None and v == v}
    if numericos:
        mlflow.log_metrics(numericos)


def registrar_artefato(caminho, pasta: str = "artefatos") -> None:
    if not tracking_ativo():
        return
    from pathlib import Path

    import mlflow

    alvo = Path(caminho)
    if alvo.exists():
        mlflow.log_artifact(str(alvo), artifact_path=pasta)
