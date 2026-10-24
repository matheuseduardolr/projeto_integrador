import pytest

from projeto_integrador.tracking import exigir_tracking, promover_champion, tracking_ativo


def test_tracking_ativo_com_uri_dos_testes():
    assert tracking_ativo() is True
    exigir_tracking()


def test_promote_sem_candidate_falha(monkeypatch, tmp_path):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    monkeypatch.setenv("PROJETO_MLOPS_DEV_MODEL", "dev.vazio")
    monkeypatch.setenv("PROJETO_MLOPS_PROD_MODEL", "prod.vazio")
    with pytest.raises(RuntimeError, match="candidate"):
        promover_champion()


def test_exigir_tracking_sem_uri(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    with pytest.raises(RuntimeError, match="obrigatório"):
        exigir_tracking()
