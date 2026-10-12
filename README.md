# Telecom Churn Pipeline

Pipeline operacional para predição de churn e segmentação de clientes em risco em telecomunicações.

## Overview
Projeto Integrador (UTFPR, Ciência de Dados) que implementa um **pipeline recorrente**: a cada ciclo a carteira entra, um classificador sinaliza quem está em risco e o **K-Means** agrupa **somente esses sinalizados** em **núcleos acionáveis** (volume, fatura e receita exposta). A operação desenha a ação por núcleo, em vez de tratar toda a lista do mesmo jeito.

O laboratório são os notebooks (Optuna, SHAP, artigo SBC). A via operacional é o pacote `projeto_integrador` (`extract` → `validate` → `train` → `score`). A base é a pública **IBM Telco Customer Churn** (7.043 clientes). Na corrida de referência do laboratório, o **XGBoost** obteve AUC-ROC 0,847, acurácia 80,6% e precisão 66,4%; 1.556 clientes foram sinalizados e agrupados com **K = 3**, expondo R$ 123,4 mil/mês.

![Fluxo](images/fluxo-projeto-integrador.svg)

## Tech Stack
* **Linguagem:** Python 3.12
* **Gerenciamento de Pacotes:** uv
* **Data Tools:** pandas, NumPy, PyArrow (Parquet)
* **Machine Learning:** scikit-learn, XGBoost, LightGBM, Random Forest
* **Clustering:** K-Means, DBSCAN, PCA
* **Otimização:** Optuna
* **Interpretabilidade:** SHAP
* **Estatística:** SciPy (Mann-Whitney, Spearman)
* **Visualização:** Matplotlib, Seaborn
* **MLOps:** pytest, GitHub Actions, Docker, MLflow (opt-in)
* **Documentação:** LaTeX (formato SBC)

## Estrutura do Projeto
```text
.
├── article/                 # Artigo científico (SBC) e slides
├── estudo_piloto/           # Primeiro recorte (1.000 clientes, 2 variáveis)
├── data/                    # raw / interm / processed (não versionado)
├── models/                  # pkl + production.json (nascem no train)
├── notebooks/               # Laboratório (ETL → EDA → features → churn → clustering → relatório)
├── src/projeto_integrador/  # Pipeline operacional (extract, train, score)
├── tests/                   # Schema, consolidação e inferência encadeada
├── Dockerfile
└── pyproject.toml           # Dependências (uv)
```

## Processo
* **Extract:** ingestão da IBM Telco, limpeza (tipos, nulos de tenure 0, categorias de serviço) e persistência em Parquet.
* **EDA:** contrato mensal como principal discriminante (42,7% vs 2,8% no bienal); Mann-Whitney e Spearman.
* **Features:** 21 colunas → 47 atributos (tempo, financeiro, serviços, risco/interação); split 80/20 estratificado; scaler só no treino.
* **Classificação:** RF × XGBoost × LightGBM, Optuna, limiar calibrado (0,795); SHAP no laboratório.
* **Segmentação:** K-Means só nos sinalizados (K = 3); DBSCAN como comparativo.
* **Núcleos:** volume, fatura média e receita exposta por grupo — saída de cada corrida do `score`. Campanha no notebook é exemplo; a ação fica com a operação.

## Pipeline operacional
```text
carteira do ciclo → extract / validate → train → Production → score → relatorio_nucleos.parquet
```

```bash
uv sync --group dev
uv run projeto-integrador extract
uv run projeto-integrador validate
uv run projeto-integrador train
uv run projeto-integrador score
```

O CSV deve estar em `data/raw/` (`extract` consolida; o download via Kaggle fica no NB01). Sem `models/production.json`, o `score` não pontua. Optuna e SHAP ficam nos notebooks; o `train` refaz a receita já escolhida. MLflow é opt-in (`uv sync --extra mlops` e `PROJETO_MLOPS_TRACKING=1`).
