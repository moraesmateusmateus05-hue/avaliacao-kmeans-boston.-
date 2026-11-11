"""
Inferência de uma nova residência.
Execute: python predict.py
O programa solicita os 13 atributos explicativos e informa o cluster e seu centroide
convertido novamente para as unidades/categorias originais.
"""
import os
import joblib
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "modelo_kmeans_boston.joblib")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Modelo não encontrado. Execute primeiro: python train.py")

artifact = joblib.load(MODEL_PATH)
preprocessor = artifact["preprocessor"]
model = artifact["model"]
feature_cols = artifact["feature_cols"]
numeric_cols = artifact["numeric_cols"]
categorical_cols = artifact["categorical_cols"]

# Entrada amigável: um valor por atributo. CHAS e RAD são categorias numéricas.
descriptions = {
    "CRIM": "taxa de criminalidade per capita",
    "ZN": "proporção de terrenos residenciais para lotes grandes",
    "INDUS": "proporção de acres comerciais não varejistas",
    "CHAS": "proximidade ao rio Charles (0 = não, 1 = sim)",
    "NOX": "concentração de óxidos de nitrogênio",
    "RM": "número médio de cômodos",
    "AGE": "proporção de unidades ocupadas construídas antes de 1940",
    "DIS": "distâncias ponderadas a cinco centros de emprego",
    "RAD": "índice de acessibilidade a rodovias radiais",
    "TAX": "taxa de imposto sobre propriedade",
    "PTRATIO": "relação aluno-professor",
    "B": "índice demográfico B",
    "LSTAT": "percentual da população de menor status"
}

print("\\n=== Dados da nova residência ===")
values = {}
for col in feature_cols:
    while True:
        try:
            raw = input(f"{col} ({descriptions.get(col, col)}): ").strip().replace(",", ".")
            values[col] = float(raw)
            break
        except ValueError:
            print("Digite um número válido.")

new_home = pd.DataFrame([values], columns=feature_cols)
new_scaled = preprocessor.transform(new_home)
cluster = int(model.predict(new_scaled)[0])

# Converte o centroide do espaço transformado de volta às unidades originais.
center = model.cluster_centers_[cluster]
num_count = len(numeric_cols)
num_pipeline = preprocessor.named_transformers_["num"]
imputer = num_pipeline.named_steps["imputer"]
scaler = num_pipeline.named_steps["scaler"]

numeric_center_scaled = center[:num_count].reshape(1, -1)
numeric_center = scaler.inverse_transform(numeric_center_scaled)[0]

# Para cada variável categórica, escolhe a categoria com maior valor médio no centroide.
cat_pipeline = preprocessor.named_transformers_["cat"]
encoder = cat_pipeline.named_steps["onehot"]
cat_center = center[num_count:]
categories = {}
offset = 0
for col, category_values in zip(categorical_cols, encoder.categories_):
    width = len(category_values)
    category_scores = cat_center[offset:offset + width]
    categories[col] = category_values[int(np.argmax(category_scores))]
    offset += width

print("\\n====================================")
print(f"Residência classificada no CLUSTER {cluster}")
print("CENTROIDE DO GRUPO (valores originais)")
print("====================================")
for col, value in zip(numeric_cols, numeric_center):
    print(f"{col}: {value:.4f}")
for col in categorical_cols:
    print(f"{col}: {categories[col]}")
