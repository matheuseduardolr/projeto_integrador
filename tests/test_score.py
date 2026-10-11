from projeto_integrador.config import FEATURE_COLS, K_CLUSTERS
from projeto_integrador.features import engenharia_features
from projeto_integrador.score import carregar_artefatos, pontuar
from projeto_integrador.schemas import validar_matriz_features, validar_saida_score


def test_engenharia_gera_scores_de_risco(consolidada):
    df = engenharia_features(consolidada)
    assert df["score_risco_estendido"].between(0, 6).all()
    assert "charges_x_contrato_mensal" in df.columns


def test_score_encadeado_produz_nucleos(consolidada, pasta_modelos):
    artefatos = carregar_artefatos(pasta_modelos)
    segmentados, relatorio, fumaca = pontuar(consolidada, artefatos)

    assert len(segmentados) == len(consolidada)
    assert segmentados["em_risco"].all()
    assert set(segmentados["cluster"].unique()).issubset(set(range(K_CLUSTERS)))
    assert list(relatorio["cluster"]) == list(range(K_CLUSTERS))
    assert fumaca["n_sinalizados"] == len(consolidada)
    validar_saida_score(segmentados, k=K_CLUSTERS)
    validar_matriz_features(segmentados[FEATURE_COLS])


def test_relatorio_soma_receita(consolidada, pasta_modelos):
    artefatos = carregar_artefatos(pasta_modelos)
    segmentados, relatorio, _ = pontuar(consolidada, artefatos)
    assert abs(
        relatorio["receita_mensal_total"].sum()
        - segmentados["MonthlyCharges_orig"].sum()
    ) < 1e-6
