"""API operacional: extração via Kaggle e o ciclo train → promote → score."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException

from projeto_integrador.config import paths
from projeto_integrador.extract import extrair_kaggle
from projeto_integrador.registry import atualizar_production
from projeto_integrador.schemas import validar_consolidada
from projeto_integrador.score import executar_lote
from projeto_integrador.tracking import promover_champion
from projeto_integrador.train import executar_treino

app = FastAPI(
    title="Telecom Churn Pipeline",
    description="Ciclo operacional: Kaggle → train → champion → núcleos acionáveis.",
)


def _erro(exc: Exception) -> HTTPException:
    return HTTPException(status_code=400, detail=str(exc))


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/extract")
def extract() -> dict:
    try:
        dirs = paths()
        df = extrair_kaggle(dirs["base_consolidada"])
    except Exception as exc:
        raise _erro(exc) from exc
    return {"linhas": int(len(df)), "saida": str(dirs["base_consolidada"])}


@app.post("/validate")
def validate() -> dict:
    import pandas as pd

    try:
        origem = paths()["base_consolidada"]
        df = pd.read_parquet(origem)
        validar_consolidada(df)
    except Exception as exc:
        raise _erro(exc) from exc
    return {"schema": "ok", "linhas": int(len(df)), "origem": str(origem)}


@app.post("/train")
def train() -> dict:
    try:
        dirs = paths()
        return executar_treino(dirs["base_consolidada"], dirs["models"])
    except Exception as exc:
        raise _erro(exc) from exc


@app.post("/promote")
def promote() -> dict:
    try:
        resultado = promover_champion()
        atualizar_production(paths()["models"], {"mlflow": resultado})
        return resultado
    except Exception as exc:
        raise _erro(exc) from exc


@app.post("/score")
def score() -> dict:
    try:
        dirs = paths()
        return executar_lote(
            dirs["base_consolidada"],
            dirs["clientes_segmentados"],
            dirs["relatorio_nucleos"],
            models_dir=dirs["models"],
        )
    except Exception as exc:
        raise _erro(exc) from exc
