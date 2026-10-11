"""Interface de linha de comando do pipeline operacional."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from projeto_integrador.config import paths
from projeto_integrador.extract import consolidar_csv
from projeto_integrador.schemas import validar_consolidada
from projeto_integrador.score import executar_lote
from projeto_integrador.train import executar_treino


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="projeto-integrador",
        description="Pipeline operacional mensal: extract, train, score e nucleos.",
    )
    sub = parser.add_subparsers(dest="comando", required=True)

    p_extract = sub.add_parser("extract", help="Consolida o CSV bruto (NB01).")
    p_extract.add_argument("--input", type=Path, default=None)
    p_extract.add_argument("--output", type=Path, default=None)

    p_val = sub.add_parser("validate", help="Valida schema da base consolidada.")
    p_val.add_argument("--input", type=Path, default=None)

    p_train = sub.add_parser(
        "train",
        help="Retreina XGBoost e K-Means e promove os artefatos a Production.",
    )
    p_train.add_argument("--input", type=Path, default=None)
    p_train.add_argument("--models", type=Path, default=None)

    p_score = sub.add_parser(
        "score",
        help="Classifica, filtra pelo limiar e agrupa os sinalizados (Production).",
    )
    p_score.add_argument("--input", type=Path, default=None)
    p_score.add_argument("--output", type=Path, default=None)
    p_score.add_argument("--relatorio", type=Path, default=None)
    p_score.add_argument("--models", type=Path, default=None)

    args = parser.parse_args(argv)
    dirs = paths()

    if args.comando == "extract":
        origem = args.input or dirs["raw_csv"]
        destino = args.output or dirs["base_consolidada"]
        df = consolidar_csv(origem, destino)
        print(f"Consolidada: {len(df):,} linhas → {destino}")
        return 0

    if args.comando == "validate":
        import pandas as pd

        origem = args.input or dirs["base_consolidada"]
        df = pd.read_parquet(origem)
        validar_consolidada(df)
        print(f"Schema OK: {origem} ({len(df):,} linhas)")
        return 0

    if args.comando == "train":
        resumo = executar_treino(
            entrada=args.input or dirs["base_consolidada"],
            models_dir=args.models or dirs["models"],
        )
        print(json.dumps(resumo, indent=2, ensure_ascii=False))
        return 0

    if args.comando == "score":
        fumaca = executar_lote(
            entrada=args.input or dirs["base_consolidada"],
            saida_segmentados=args.output or dirs["clientes_segmentados"],
            saida_relatorio=args.relatorio or dirs["relatorio_nucleos"],
            models_dir=args.models or dirs["models"],
        )
        print(json.dumps(fumaca, indent=2, ensure_ascii=False))
        return 0

    return 1
