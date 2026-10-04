"""Estudo piloto: primeiro recorte, amostra pequena e poucas variáveis."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
FIG_DIR = Path(__file__).resolve().parent / "figuras"
FIG_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=1.05)
plt.rcParams["figure.dpi"] = 140

N_PILOTO = 1000
RANDOM_STATE = 42
# Primeiro recorte da EDA: só as variáveis que saltam à vista.
FEATURE_COLS = ["tenure", "Contract"]

df = pd.read_csv(CSV_PATH)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
df["Churn"] = (df["Churn"] == "Yes").astype(int)

_, piloto = train_test_split(
    df, test_size=N_PILOTO, random_state=RANDOM_STATE, stratify=df["Churn"]
)
piloto = piloto.copy()
print(f"Amostra piloto: {len(piloto)} clientes")
print(f"Taxa de churn na amostra: {piloto['Churn'].mean():.3f}")
print(f"Taxa de churn na base completa: {df['Churn'].mean():.3f}")
print(f"Features do modelo: {FEATURE_COLS}")

id_col = "customerID"
target = "Churn"
X = piloto[FEATURE_COLS]
y = piloto[target]

num_cols = X.select_dtypes(include=np.number).columns.tolist()
cat_cols = X.select_dtypes(exclude=np.number).columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.40, random_state=RANDOM_STATE, stratify=y
)
print(f"Treino: {len(X_train)} | Teste: {len(X_test)}")

preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ]
)

modelos = {
    "Regressão Logística": LogisticRegression(max_iter=500, random_state=RANDOM_STATE),
    "Árvore de Decisão": DecisionTreeClassifier(max_depth=3, random_state=RANDOM_STATE),
}

resultados = {}
for nome, clf in modelos.items():
    pipe = Pipeline([("prep", preprocess), ("clf", clf)])
    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    metricas = {
        "auc": roc_auc_score(y_test, proba),
        "acc": accuracy_score(y_test, pred),
        "prec": precision_score(y_test, pred, zero_division=0),
        "rec": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "pipe": pipe,
        "proba": proba,
        "pred": pred,
    }
    resultados[nome] = metricas
    print(f"\n=== {nome} (limiar 0,5) ===")
    print(
        f"AUC={metricas['auc']:.3f}  Acc={metricas['acc']:.3f}  "
        f"Prec={metricas['prec']:.3f}  Rec={metricas['rec']:.3f}  F1={metricas['f1']:.3f}"
    )
    print(classification_report(y_test, pred, digits=3))
    print("Matriz:\n", confusion_matrix(y_test, pred))

melhor_nome = max(resultados, key=lambda n: resultados[n]["auc"])
melhor = resultados[melhor_nome]
print(f"\nModelo selecionado no piloto: {melhor_nome}")

fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))
cm = confusion_matrix(y_test, melhor["pred"])
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0], cbar=False)
axes[0].set_title(f"Matriz de confusão — {melhor_nome}")
axes[0].set_xlabel("Predito")
axes[0].set_ylabel("Real")
axes[0].set_xticklabels(["Não churn", "Churn"])
axes[0].set_yticklabels(["Não churn", "Churn"])

fpr, tpr, _ = roc_curve(y_test, melhor["proba"])
axes[1].plot(fpr, tpr, label=f"AUC = {melhor['auc']:.3f}")
axes[1].plot([0, 1], [0, 1], "--", color="gray")
axes[1].set_title("Curva ROC (hold-out do piloto)")
axes[1].set_xlabel("Falso positivo")
axes[1].set_ylabel("Verdadeiro positivo")
axes[1].legend(loc="lower right")

nomes = list(resultados)
vals = {
    "Acurácia": [resultados[n]["acc"] for n in nomes],
    "Precisão": [resultados[n]["prec"] for n in nomes],
    "Recall": [resultados[n]["rec"] for n in nomes],
    "F1": [resultados[n]["f1"] for n in nomes],
}
x = np.arange(len(nomes))
width = 0.18
for i, (lab, serie) in enumerate(vals.items()):
    axes[2].bar(x + i * width, serie, width, label=lab)
axes[2].set_xticks(x + 1.5 * width)
axes[2].set_xticklabels(nomes, rotation=10)
axes[2].set_ylim(0, 1)
axes[2].set_title("Métricas no teste (limiar 0,5)")
axes[2].legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIG_DIR / "avaliacao_piloto.png", bbox_inches="tight")
plt.close()

pipe = melhor["pipe"]
piloto["proba"] = pipe.predict_proba(piloto[FEATURE_COLS])[:, 1]
piloto["sinalizado"] = (piloto["proba"] >= 0.5).astype(int)
risco = piloto[piloto["sinalizado"] == 1].copy()
print(f"\nSinalizados na amostra piloto: {len(risco)} ({len(risco)/len(piloto):.1%})")
if len(risco) == 0:
    raise SystemExit("Nenhum cliente sinalizado; ajuste o recorte.")
print(f"Taxa real de churn entre sinalizados: {risco['Churn'].mean():.3f}")
print(f"Receita mensal exposta (sinalizados): R$ {risco['MonthlyCharges'].sum():,.2f}")

feat_cluster = pd.get_dummies(
    risco[["tenure", "MonthlyCharges", "InternetService", "Contract"]],
    drop_first=True,
)
feat_cluster = StandardScaler().fit_transform(feat_cluster)
risco["segmento"] = KMeans(n_clusters=2, random_state=RANDOM_STATE, n_init=10).fit_predict(
    feat_cluster
)
print(f"Silhueta K=2: {silhouette_score(feat_cluster, risco['segmento']):.3f}")

perfil = (
    risco.groupby("segmento")
    .agg(
        clientes=("Churn", "size"),
        tenure_medio=("tenure", "mean"),
        fatura_media=("MonthlyCharges", "mean"),
        receita=("MonthlyCharges", "sum"),
        taxa_churn=("Churn", "mean"),
        proba_media=("proba", "mean"),
        fibra=("InternetService", lambda s: (s == "Fiber optic").mean()),
        mensal=("Contract", lambda s: (s == "Month-to-month").mean()),
    )
    .round(3)
)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
print("\nPerfil dos segmentos:\n", perfil)
print("\nReceita por segmento:\n", risco.groupby("segmento")["MonthlyCharges"].agg(["size", "sum", "mean"]))
print("\nTaxa churn por segmento:\n", risco.groupby("segmento")["Churn"].mean())

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
sns.scatterplot(
    data=risco,
    x="tenure",
    y="MonthlyCharges",
    hue="segmento",
    palette="Set2",
    ax=axes[0],
    alpha=0.75,
)
axes[0].set_title("Segmentos piloto (K = 2) — tenure × fatura")
axes[0].set_xlabel("Tempo de relacionamento (meses)")
axes[0].set_ylabel("Fatura mensal")

impacto = risco.groupby("segmento")["MonthlyCharges"].sum()
axes[1].bar([f"Segmento {i}" for i in impacto.index], impacto.values, color=["#66c2a5", "#fc8d62"])
axes[1].set_title("Receita mensal exposta por segmento")
axes[1].set_ylabel("R$")
for i, v in enumerate(impacto.values):
    axes[1].text(i, v, f"R$ {v:,.0f}", ha="center", va="bottom", fontsize=9)
fig.tight_layout()
fig.savefig(FIG_DIR / "segmentos_piloto.png", bbox_inches="tight")
plt.close()

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
contrato = piloto.groupby("Contract")["Churn"].mean().reindex(
    ["Month-to-month", "One year", "Two year"]
)
axes[0].bar(["Mensal", "Anual", "Bienal"], contrato.values, color="#4c72b0")
axes[0].set_ylabel("Taxa de churn")
axes[0].set_title("Churn por tipo de contrato (amostra piloto)")
for i, v in enumerate(contrato.values):
    axes[0].text(i, v, f"{v:.1%}", ha="center", va="bottom")

sns.kdeplot(data=piloto, x="tenure", hue="Churn", common_norm=False, ax=axes[1], fill=True, alpha=0.35)
axes[1].set_title("Tempo de relacionamento por churn")
axes[1].set_xlabel("Meses")
fig.tight_layout()
fig.savefig(FIG_DIR / "eda_piloto.png", bbox_inches="tight")
plt.close()

print("\nFiguras salvas em", FIG_DIR)
