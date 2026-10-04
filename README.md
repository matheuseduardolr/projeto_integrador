# Predição de Churn e Segmentação de Clientes em Telecomunicações

**Pipeline end-to-end de Machine Learning que prevê o cancelamento de clientes, segmenta a carteira em risco em perfis acionáveis e quantifica a receita mensal exposta.**

`Data Science` · `Machine Learning` · `Churn Prediction` · `Customer Segmentation` · `Python` · `XGBoost` · `K-Means` · `SHAP`

---

## Overview

Projeto de **Ciência de Dados** aplicado ao problema de **customer churn** no setor de telecomunicações, estruturado como um pipeline encadeado de duas etapas: um modelo de **classificação supervisionada** identifica os clientes com maior propensão a cancelar o serviço e, em seguida, um algoritmo de **clustering** segmenta exclusivamente essa população sinalizada em perfis comportamentais. A diferença em relação a um modelo de churn convencional é que a entrega não é apenas uma lista de risco, mas **núcleos acionáveis** com volume, fatura e receita exposta — a área comercial é quem desenha as campanhas sobre cada perfil. Sobre a base pública IBM Telco Customer Churn (7.043 clientes, 21 variáveis), o modelo **XGBoost** selecionado alcançou **AUC-ROC de 0,847, acurácia de 80,6% e precisão de 66,4%** em conjunto de teste *hold-out*, o equivalente a aproximadamente **2,5x a taxa base** de cancelamento, sinalizando 1.556 clientes (22,1% da carteira). A segmentação com **K-Means (K=3)** sobre esses clientes revelou três perfis distintos que expõem **R$ 123,4 mil de receita mensal** — R$ 1,48 milhão em projeção anual — com 59% concentrados em um único segmento. A interpretabilidade foi tratada com **SHAP values**, confirmando que as variáveis criadas na etapa de *feature engineering* dominam a predição. Desenvolvido como Projeto Integrador da Especialização em Ciência de Dados da **UTFPR**.

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
│   └── processed/                   # conjuntos de treino/teste, clientes em risco e segmentados
├── models/                          # modelos, encoders e scaler serializados (joblib)
├── notebooks/
│   ├── 01_extracao.ipynb            # ETL e consolidação
│   ├── 02_eda.ipynb                 # análise exploratória
│   ├── 03_feature_store.ipynb       # feature engineering e pré-processamento
│   ├── 04_churn.ipynb               # modelagem preditiva e interpretabilidade
│   ├── 05_segmentacao.ipynb         # clustering dos clientes em risco
│   └── 06_relatorio.ipynb           # métricas de negócio + exemplo ilustrativo de leitura comercial
├── src/projeto_integrador/
├── extrai_figuras.py                # exporta as figuras dos notebooks para o artigo
├── pyproject.toml                   # dependências gerenciadas com uv
└── uv.lock
```

Cada notebook é autônomo: carrega seus insumos do disco e persiste seus resultados em Parquet, permitindo reexecutar qualquer etapa isoladamente.

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

Tradução dos segmentos em **métricas de negócio**: volume de clientes, taxa real de cancelamento, receita mensal exposta e projeção anual por perfil. O resultado central é o contraste entre volume e valor — um segmento concentra 15,5% dos clientes sinalizados mas apenas 8,3% da receita exposta, enquanto outro reúne 57,1% dos clientes e 59,0% da receita. Essa leitura é a **saída principal** do pipeline: três núcleos distintos para a empresa tratar de forma diferenciada. Oferta, canal e KPI eventuais no notebook são só um exemplo de como a operação *pode* ler cada perfil — quem define a campanha é a área comercial. Os resultados foram documentados em artigo científico no formato SBC, precedido por um estudo piloto.