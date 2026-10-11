from projeto_integrador.config import K_CLUSTERS
from projeto_integrador.registry import ler_production
from projeto_integrador.score import carregar_artefatos, executar_lote, pontuar
from projeto_integrador.train import executar_treino


def test_train_promove_production_e_score_republica_nucleos(bruto, tmp_path):
    entrada = tmp_path / "base.parquet"
    bruto_ext = __import__("pandas").concat(
        [bruto, bruto.assign(customerID=lambda d: "b-" + d["customerID"])],
        ignore_index=True,
    )
    from projeto_integrador.extract import consolidar

    consolidada = consolidar(bruto_ext)
    consolidada.to_parquet(entrada, index=False)

    models = tmp_path / "models"
    resumo = executar_treino(entrada, models)

    manifesto = ler_production(models)
    assert manifesto is not None
    assert manifesto["stage"] == "Production"
    assert (models / "modelo_churn.pkl").exists()
    assert (models / "kmeans_segmentacao.pkl").exists()
    assert (models / "contract_means.json").exists()
    assert resumo["k"] == K_CLUSTERS
    assert resumo["n_sinalizados"] >= K_CLUSTERS

    artefatos = carregar_artefatos(models, exigir_production=True)
    segmentados, relatorio, fumaca = pontuar(consolidada, artefatos)
    assert fumaca["n_sinalizados"] == resumo["n_sinalizados"]
    assert list(relatorio["cluster"]) == list(range(K_CLUSTERS))
    assert abs(
        relatorio["receita_mensal_total"].sum()
        - segmentados["MonthlyCharges_orig"].sum()
    ) < 1e-6

    saida = tmp_path / "seg.parquet"
    rel = tmp_path / "nucleos.parquet"
    lote = executar_lote(entrada, saida, rel, models_dir=models, registrar=False)
    assert lote["n_sinalizados"] == resumo["n_sinalizados"]
    assert saida.exists() and rel.exists()
