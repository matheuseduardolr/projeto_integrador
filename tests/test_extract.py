from projeto_integrador.extract import consolidar
from projeto_integrador.schemas import SchemaError, validar_consolidada


def test_consolidar_converte_churn_e_total(bruto):
    df = consolidar(bruto)
    assert df["Churn"].isin([0, 1]).all()
    assert df["TotalCharges"].dtype.kind == "f"
    assert df["TotalCharges"].isnull().sum() == 0
    validar_consolidada(df)


def test_consolidar_padroniza_servico_sem_internet(bruto):
    bruto = bruto.copy()
    bruto.loc[0, "InternetService"] = "No"
    bruto.loc[0, "OnlineSecurity"] = "No internet service"
    df = consolidar(bruto)
    assert df.loc[0, "OnlineSecurity"] == "No"


def test_schema_rejeita_id_duplicado(bruto):
    bruto = bruto.copy()
    bruto.loc[1, "customerID"] = bruto.loc[0, "customerID"]
    try:
        consolidar(bruto)
        raise AssertionError("deveria falhar")
    except SchemaError:
        pass
