"""
Treinamento do modelo K-means - Boston Housing
Execute: python train.py
"""
import os
import json
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "HousingData.csv")
MODEL_PATH = os.path.join(BASE_DIR, "modelo_kmeans_boston.joblib")

df = pd.read_csv(DATA_PATH)

# CHAS e RAD são tratadas como categóricas; as demais variáveis explicativas são quantitativas.
# MEDV é o valor/preço da residência (variável-alvo do conjunto original), por isso não entra
# como atributo de entrada do agrupamento.
categorical_cols = ["CHAS", "RAD"]
numeric_cols = [c for c in df.columns if c not in categorical_cols + ["MEDV"]]
feature_cols = numeric_cols + categorical_cols

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])
preprocessor = ColumnTransformer([
    ("num", numeric_pipe, numeric_cols),
    ("cat", categorical_pipe, categorical_cols)
], verbose_feature_names_out=False)

X = df[feature_cols]
X_scaled = preprocessor.fit_transform(X)

# Método do cotovelo: mede a distância perpendicular de cada ponto à reta
# que liga o primeiro e o último valor de inércia.
ks = list(range(1, 11))
inertias = []
for k in ks:
    candidate = KMeans(n_clusters=k, random_state=42, n_init=20)
    candidate.fit(X_scaled)
    inertias.append(float(candidate.inertia_))

points = np.column_stack((np.array(ks, dtype=float), np.array(inertias)))
first, last = points[0], points[-1]
distances = np.abs(np.cross(last - first, points - first)) / np.linalg.norm(last - first)
optimal_k = int(ks[int(np.argmax(distances))])

model = KMeans(n_clusters=optimal_k, random_state=42, n_init=20)
labels = model.fit_predict(X_scaled)
silhouette = float(silhouette_score(X_scaled, labels)) if optimal_k > 1 else None

artifact = {
    "preprocessor": preprocessor,
    "model": model,
    "feature_cols": feature_cols,
    "numeric_cols": numeric_cols,
    "categorical_cols": categorical_cols,
    "optimal_k": optimal_k,
    "elbow_k": ks,
    "inertias": inertias
}
joblib.dump(artifact, MODEL_PATH)

plt.figure(figsize=(8, 5))
plt.plot(ks, inertias, marker="o")
plt.axvline(optimal_k, linestyle="--", label=f"K escolhido = {optimal_k}")
plt.title("Método do cotovelo - Boston Housing")
plt.xlabel("Número de clusters (K)")
plt.ylabel("Inércia")
plt.xticks(ks)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "cotovelo.png"), dpi=150)
print(f"Modelo salvo em: {MODEL_PATH}")
print(f"Número de clusters escolhido: {optimal_k}")
print(f"Silhouette (referência): {silhouette:.4f}" if silhouette is not None else "")
