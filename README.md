# Pipeline para Predição de Churn e Segmentação de Clientes em Risco no Setor de Telecomunicações

**Pipeline operacional (MLOps) para telecomunicações: classifica risco de churn, agrupa só os sinalizados e devolve núcleos acionáveis a cada ciclo — para a empresa não tratar a carteira toda igual.**

`Data Science` · `MLOps` · `Churn Prediction` · `Customer Segmentation` · `Python` · `XGBoost` · `K-Means` · `MLflow`

---

## Overview

Projeto Integrador (UTFPR) de **Ciência de Dados**: um **pipeline recorrente** para operadoras de telecomunicações. A cada ciclo (por exemplo, todo mês) a carteira entra, o classificador sinaliza quem está em risco e o K-Means agrupa **somente esses sinalizados** em núcleos com volume, fatura e receita exposta. A empresa desenha a ação por núcleo. Os notebooks são o laboratório (Optuna, SHAP, artigo); o pacote `projeto_integrador` é a via operacional (`extract` → `train` → `score`). Sobre a base pública IBM Telco Customer Churn (7.043 clientes), a **corrida de referência do laboratório** com **XGBoost** obteve **AUC-ROC de 0,847, acurácia de 80,6% e precisão de 66,4%**, sinalizou 1.556 clientes (22,1%) e, com **K-Means (K=3)**, expôs **R$ 123,4 mil/mês** — 59% num único núcleo. O `train` operacional refaz essa receita (sem redesenhar o Optuna) e não reproduz automaticamente aqueles totais.

## Tech Stack

- **Linguagem:** Python 3.12
- **Manipulação e Análise de Dados:** pandas, NumPy, PyArrow (formato Parquet)
- **Machine Learning:** scikit-learn, XGBoost, LightGBM, Random Forest, imbalanced-learn
- **Aprendizado Não Supervisionado:** K-Means, DBSCAN, PCA (Principal Component Analysis)
- **Otimização de Hiperparâmetros:** Optuna (Bayesian Optimization, TPE Sampler)
- **Interpretabilidade de Modelos (Explainable AI):** SHAP (SHapley Additive exPlanations)
- **Estatística e Testes de Hipótese:** SciPy (teste de Mann-Whitney, correlação de Spearman)
- **Visualização de Dados:** Matplotlib, Seaborn
- **Ambiente e Versionamento:** uv, Jupyter Notebook, Git, GitHub
- **MLOps:** pacote `projeto_integrador` (`extract` → `validate` → `train` → `score`), manifesto Production, pytest, GitHub Actions, Docker; MLflow opt-in
- **Documentação Técnica:** LaTeX (formato SBC)



## Estrutura do Projeto

```
projeto_integrador/
├── article/
│   ├── artigo/                      # artigo científico em LaTeX (formato SBC)
│   │   ├── article.tex
│   │   ├── referencias.bib
│   │   ├── roteiro_slides.txt       # briefing para slides (NotebookLM)
│   │   └── figuras/                 # figuras extraídas das saídas dos notebooks
│   └── modelo artigo/               # template da disciplina
├── estudo_piloto/                   # primeiro recorte (1.000 clientes, 2 variáveis)
├── data/
│   ├── raw/                         # dataset original (IBM Telco Customer Churn)
│   ├── interm/                      # base consolidada após limpeza (Parquet)
│   └── processed/                   # treino/teste, sinalizados; NB06 → relatorio_final.parquet; score → relatorio_nucleos.parquet
├── models/                          # pkl + production.json (não versionados; nascem no `train`)
├── notebooks/
│   ├── 01_extracao.ipynb            # ETL e consolidação
│   ├── 02_eda.ipynb                 # análise exploratória
│   ├── 03_feature_store.ipynb       # feature engineering e pré-processamento
│   ├── 04_churn.ipynb               # modelagem preditiva e interpretabilidade
│   ├── 05_segmentacao.ipynb         # clustering dos clientes em risco
│   └── 06_relatorio.ipynb           # métricas de negócio + exemplo ilustrativo de leitura comercial
├── src/projeto_integrador/          # pipeline operacional (extract, train, score, núcleos)
├── tests/                           # testes de schema, consolidação e inferência encadeada
├── extrai_figuras.py                # exporta as figuras dos notebooks para o artigo
├── Dockerfile
├── pyproject.toml                   # dependências gerenciadas com uv
└── uv.lock
```

Cada notebook é autônomo: carrega seus insumos do disco e persiste seus resultados em Parquet, permitindo reexecutar qualquer etapa isoladamente.

## Ciclo mensal (entregável)

O produto não é uma fotografia única da base IBM. É o **mesmo código** rodando de novo quando chegar a carteira do mês:

```text
carteira do ciclo
  → extract / validate
  → train          (retreina XGBoost + K-Means e promove a Production)
  → score          (lê só o manifesto Production)
  → relatorio_nucleos.parquet   ← o que a operação usa
```

```bash
uv sync --group dev
uv run pytest
uv run projeto-integrador extract
uv run projeto-integrador validate
uv run projeto-integrador train
uv run projeto-integrador score
```

Dados e modelos **não estão no Git**: `data/` e `models/` só têm `.gitkeep`. Depois do clone:

1. Coloque o CSV da IBM Telco em `data/raw/` (o NB01 baixa via API do Kaggle; o comando `extract` **não** baixa — só consolida o arquivo local `WA_Fn-UseC_-Telco-Customer-Churn.csv`).
2. Rode `extract` → `validate` → `train`. Sem `models/production.json`, o `score` recusa a pasta.
3. Só então rode `score`.

`train` grava os pkl e o manifesto Production. A saída de negócio do ciclo é `data/processed/relatorio_nucleos.parquet` (volume, fatura, receita por núcleo). O laboratório (NB06) grava o registro científico equivalente em `data/processed/relatorio_final.parquet`. Campanha continua com a operação.

Optuna e SHAP ficam nos notebooks — o `train` operacional refaz a receita já escolhida (XGBoost + limiar + K=3).

Tracking no MLflow é opt-in:

```bash
uv sync --extra mlops
set PROJETO_MLOPS_TRACKING=1
uv run projeto-integrador train
uv run projeto-integrador score
```

## Processo



### 1. Extração e Consolidação (ETL)

Ingestão do dataset via **API do Kaggle** e tratamento de qualidade de dados: conversão de tipos (a coluna de fatura acumulada vinha como texto), identificação e tratamento de **valores ausentes** — 11 registros vazios, todos de clientes recém-ativados sem fatura gerada — e padronização de categorias redundantes em sete colunas de serviços. Validação de integridade com asserções de unicidade de chave e ausência de nulos. Saída persistida em **Parquet** para otimizar leitura nas etapas seguintes.

### 2. Análise Exploratória de Dados (EDA)

Investigação em três lentes: perfil de churn, comportamento financeiro e uso de serviços. A análise identificou o **tipo de contrato** como principal discriminante (42,7% de cancelamento em contratos mensais contra 2,8% em contratos de dois anos) e evidenciou concentração de churn nos primeiros meses de relacionamento. As diferenças entre grupos foram validadas estatisticamente com o **teste de Mann-Whitney** (não paramétrico, dada a assimetria das distribuições) e a associação com a variável alvo medida por **correlação de Spearman**.

### 3. Feature Engineering e Pré-processamento

Construção de uma **feature store** com 47 atributos derivados das 21 colunas originais, organizados em quatro grupos: temporais (faixas de tempo de relacionamento), financeiros (fatura média por mês de permanência, desvio em relação aos pares contratuais), de serviços (contagens e indicadores de ausência) e de risco (escore composto e termos de interação sugeridos pela EDA). Codificação com **Label Encoding** e **One-Hot Encoding**, **divisão estratificada** treino/teste em 80/20 preservando a proporção de classes, e **normalização com StandardScaler ajustado exclusivamente no conjunto de treino**.

### 4. Modelagem Preditiva e Interpretabilidade

Comparação de três algoritmos de ensemble — **Random Forest, XGBoost e LightGBM** — com **otimização bayesiana de hiperparâmetros via Optuna** (40 trials por algoritmo) avaliada por **validação cruzada estratificada de 5 folds**. O **desbalanceamento de classes** (26,5% de churn) foi tratado com ponderação (`class_weight`, `scale_pos_weight`). A função objetivo combina F-medida e acurácia, com o **threshold de decisão otimizado dentro de cada avaliação** em vez do corte padrão de 0,5 — escolha orientada pelo custo operacional do falso positivo, já que cada cliente sinalizado gera uma ação comercial. Avaliação final em *hold-out* com **AUC-ROC, precisão, recall, F1-score e matriz de confusão**, seguida de **interpretabilidade com SHAP** (importância global e decomposição de predições individuais).

### 5. Segmentação de Clientes em Risco (Clustering)

Aplicação de **K-Means** exclusivamente sobre os 1.556 clientes sinalizados pelo modelo, com o número de clusters definido por **Elbow Method** e **Silhouette Score**, e **DBSCAN** avaliado como comparativo. Projeção por **PCA** para inspeção visual da separação entre grupos. Os três segmentos resultantes foram caracterizados por tempo de relacionamento, patamar de fatura, tipo de acesso e serviços contratados, gerando perfis interpretáveis pela área comercial.

### 6. Relatório de Impacto

Tradução dos segmentos em **métricas de negócio**: volume de clientes, taxa real de cancelamento, receita mensal exposta e projeção anual por perfil. O resultado central é o contraste entre volume e valor — um segmento concentra 15,5% dos clientes sinalizados mas apenas 8,3% da receita exposta, enquanto outro reúne 57,1% dos clientes e 59,0% da receita. Essa leitura é a **saída de cada corrida** do pipeline: três núcleos para a empresa tratar de forma diferenciada no ciclo. Oferta, canal e KPI no notebook são só exemplo — quem define a campanha é a área comercial. O laboratório e o ciclo operacional estão documentados no artigo SBC, precedido por um estudo piloto.