from fastapi.testclient import TestClient

from projeto_integrador.api import app


def test_health():
    client = TestClient(app)
    assert client.get("/health").json() == {"status": "ok"}


def test_extract_chama_kaggle(monkeypatch, tmp_path, consolidada):
    monkeypatch.setenv("PROJETO_ROOT", str(tmp_path))
    (tmp_path / "data" / "interm").mkdir(parents=True)

    def fake_kaggle(destino):
        consolidada.to_parquet(destino, index=False)
        return consolidada

    monkeypatch.setattr("projeto_integrador.api.extrair_kaggle", fake_kaggle)
    client = TestClient(app)
    resp = client.post("/extract")
    assert resp.status_code == 200
    corpo = resp.json()
    assert corpo["linhas"] == len(consolidada)
