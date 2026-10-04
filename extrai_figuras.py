"""Extrai as figuras já renderizadas nos notebooks para o diretório do artigo."""

import base64
import json
from pathlib import Path

DESTINO = Path("article/artigo/figuras")

# (notebook, índice da célula, índice da imagem na célula) -> nome do arquivo
MAPA = {
    ("02_eda", 2, 0): "eda_perfil_demografico",
    ("02_eda", 2, 1): "eda_contrato",
    ("02_eda", 2, 2): "eda_pagamento",
    ("02_eda", 3, 0): "eda_financeiro",
    ("02_eda", 3, 1): "eda_tenure",
    ("02_eda", 4, 0): "eda_servicos",
    ("02_eda", 5, 0): "eda_correlacao",
    ("04_churn", 13, 0): "avaliacao_modelo",
    ("04_churn", 15, 1): "shap_beeswarm",
    ("04_churn", 15, 2): "shap_waterfall",
    ("05_segmentacao", 6, 0): "selecao_k",
    ("05_segmentacao", 12, 0): "pca_clusters",
    ("05_segmentacao", 14, 0): "perfil_clusters",
    ("05_segmentacao", 14, 1): "boxplots_clusters",
    ("06_relatorio", 5, 0): "metricas_modelo",
    ("06_relatorio", 9, 0): "impacto_clusters",
    ("06_relatorio", 9, 1): "receita_participacao",
}


def main() -> None:
    DESTINO.mkdir(parents=True, exist_ok=True)
    for caminho in sorted(Path("notebooks").glob("*.ipynb")):
        nb_nome = caminho.stem
        nb = json.loads(caminho.read_text(encoding="utf-8"))
        for idx_celula, celula in enumerate(nb["cells"]):
            idx_imagem = 0
            for saida in celula.get("outputs", []):
                dados = saida.get("data", {})
                if "image/png" not in dados:
                    continue
                nome = MAPA.get((nb_nome, idx_celula, idx_imagem))
                if nome:
                    png = dados["image/png"]
                    if isinstance(png, list):
                        png = "".join(png)
                    (DESTINO / f"{nome}.png").write_bytes(base64.b64decode(png))
                    print(f"{nome}.png  <-  {nb_nome} celula {idx_celula} imagem {idx_imagem}")
                idx_imagem += 1


if __name__ == "__main__":
    main()
