"""Tracking e Model Registry no MLflow — obrigatórios no ciclo operacional."""

from __future__ import annotations

import numbers
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


def _carregar_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(override=False)


_carregar_dotenv()


def tracking_ativo() -> bool:
    return bool(os.environ.get("MLFLOW_TRACKING_URI", "").strip())


def exigir_tracking() -> None:
    if not tracking_ativo():
        raise RuntimeError(
            "MLflow é obrigatório. Copie .env.example para .env e suba "
            "`docker compose up -d` (MLFLOW_TRACKING_URI)."
        )


def nome_experimento() -> str:
    return os.environ.get("MLFLOW_EXPERIMENT_NAME", "projeto-integrador")


def nome_modelo_dev() -> str:
    return os.environ.get("PROJETO_MLOPS_DEV_MODEL", "dev.churn-pipeline")


def nome_modelo_prod() -> str:
    return os.environ.get("PROJETO_MLOPS_PROD_MODEL", "prod.churn-pipeline")


def _preparar_mlflow():
    import mlflow

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(nome_experimento())
    return mlflow


@contextmanager
def iniciar_run(nome: str, tags: dict | None = None) -> Iterator[object]:
    exigir_tracking()
    mlflow = _preparar_mlflow()
    with mlflow.start_run(run_name=nome) as run:
        if tags:
            mlflow.set_tags(tags)
        yield run


def registrar_params(params: dict) -> None:
    exigir_tracking()
    import mlflow

    mlflow.log_params({k: str(v) for k, v in params.items()})


def registrar_metrics(metrics: dict) -> None:
    exigir_tracking()
    import mlflow

    numericos = {
        k: float(v) for k, v in metrics.items() if isinstance(v, numbers.Real) and v == v
    }
    if numericos:
        mlflow.log_metrics(numericos)


def registrar_artefato(caminho, pasta: str = "artefatos") -> None:
    exigir_tracking()
    import mlflow

    alvo = Path(caminho)
    if alvo.exists():
        mlflow.log_artifact(str(alvo), artifact_path=pasta)


def _garantir_modelo(client, nome: str, descricao: str) -> None:
    try:
        client.get_registered_model(nome)
    except Exception:
        client.create_registered_model(nome, description=descricao)


def registrar_no_registry(models_dir: Path | None = None) -> dict:
    exigir_tracking()
    import mlflow
    from mlflow.tracking import MlflowClient

    run = mlflow.active_run()
    if run is None:
        raise RuntimeError("registrar_no_registry exige um run MLflow ativo.")
    run_id = run.info.run_id
    if models_dir is not None:
        for nome in (
            "modelo_churn.pkl",
            "kmeans_segmentacao.pkl",
            "scaler.pkl",
            "encoders_binarios.pkl",
            "contract_means.json",
            "production.json",
        ):
            registrar_artefato(Path(models_dir) / nome, "production")

    nome = nome_modelo_dev()
    client = MlflowClient()
    _garantir_modelo(client, nome, "Pipeline encadeado XGBoost + K-Means (candidato).")
    versao = client.create_model_version(
        name=nome,
        source=f"runs:/{run_id}/production",
        run_id=run_id,
    )
    client.set_registered_model_alias(nome, "candidate", versao.version)
    return {
        "dev_model": nome,
        "version": versao.version,
        "run_id": run_id,
        "alias": "candidate",
    }


def promover_champion() -> dict:
    exigir_tracking()
    from mlflow.tracking import MlflowClient

    _preparar_mlflow()
    client = MlflowClient()
    dev = nome_modelo_dev()
    prod = nome_modelo_prod()
    try:
        candidato = client.get_model_version_by_alias(dev, "candidate")
    except Exception as exc:
        raise RuntimeError(
            f"Nenhum candidate em {dev}. Rode train antes do promote."
        ) from exc

    _garantir_modelo(client, prod, "Pipeline encadeado em Production (champion).")
    promovido = client.create_model_version(
        name=prod,
        source=candidato.source,
        run_id=candidato.run_id,
    )
    client.set_registered_model_alias(prod, "champion", promovido.version)
    return {
        "dev_model": dev,
        "prod_model": prod,
        "dev_version": candidato.version,
        "prod_version": promovido.version,
        "run_id": candidato.run_id,
        "alias": "champion",
    }


def consultar_champion() -> dict:
    exigir_tracking()
    from mlflow.tracking import MlflowClient

    _preparar_mlflow()
    client = MlflowClient()
    try:
        mv = client.get_model_version_by_alias(nome_modelo_prod(), "champion")
    except Exception as exc:
        raise RuntimeError(
            f"Nenhum champion em {nome_modelo_prod()}. Rode train e promote."
        ) from exc
    return {
        "prod_model": nome_modelo_prod(),
        "version": mv.version,
        "run_id": mv.run_id,
        "alias": "champion",
    }
