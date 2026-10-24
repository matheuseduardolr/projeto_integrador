"""Manifesto do estágio Production — o score lê só o que foi promovido."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

ARQUIVOS_PRODUCAO = (
    "modelo_churn",
    "kmeans",
    "scaler",
    "encoders",
    "contract_means",
)


def caminho_production(models_dir: Path) -> Path:
    return models_dir / "production.json"


def escrever_production(models_dir: Path, info: dict) -> Path:
    models_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "stage": "Production",
        "promovido_em": datetime.now(timezone.utc).isoformat(),
        "modelo_churn": "modelo_churn.pkl",
        "kmeans": "kmeans_segmentacao.pkl",
        "scaler": "scaler.pkl",
        "encoders": "encoders_binarios.pkl",
        "feature_cols": "feature_cols.pkl",
        "contract_means": "contract_means.json",
        **info,
    }
    destino = caminho_production(models_dir)
    destino.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return destino


def atualizar_production(models_dir: Path, extra: dict) -> Path | None:
    """Acrescenta campos (ex.: run do MLflow) ao manifesto já escrito."""
    atual = ler_production(models_dir)
    if atual is None:
        return None
    atual.update(extra)
    destino = caminho_production(models_dir)
    destino.write_text(json.dumps(atual, indent=2, ensure_ascii=False), encoding="utf-8")
    return destino


def ler_production(models_dir: Path) -> dict | None:
    destino = caminho_production(models_dir)
    if not destino.exists():
        return None
    return json.loads(destino.read_text(encoding="utf-8"))


def resolver_models_dir(models_dir: Path, exigir_production: bool = False) -> Path:
    """Valida o manifesto Production, se existir. Sem manifesto, usa a pasta crua."""
    manifesto = ler_production(models_dir)
    if manifesto is None:
        if exigir_production:
            raise FileNotFoundError(
                f"Nenhum modelo em Production em {models_dir}. Rode `projeto-integrador train`."
            )
        return models_dir
    for chave in ARQUIVOS_PRODUCAO:
        nome = manifesto.get(chave)
        if not nome:
            continue
        caminho = models_dir / nome
        if not caminho.exists():
            raise FileNotFoundError(
                f"Artefato de Production ausente ({chave}): {caminho}"
            )
    return models_dir
