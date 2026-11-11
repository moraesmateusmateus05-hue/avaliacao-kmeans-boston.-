# Avaliação do 2º Bimestre — Clusterização com K-means (Boston Housing)

## Objetivo
Agrupar residências/regiões da base Boston Housing com K-means e permitir classificar uma nova residência, exibindo o centroide do grupo em valores originais.

## Estrutura
- `HousingData.csv`: base fornecida para a avaliação.
- `train.py`: preparação dos dados, método do cotovelo, treinamento e salvamento.
- `predict.py`: recebe os dados de uma nova residência, prevê o cluster e apresenta o centroide desnormalizado.
- `modelo_kmeans_boston.joblib`: modelo e pré-processador treinados.
- `cotovelo.png`: gráfico utilizado para escolher K.

## Tratamento dos dados
- `CHAS` e `RAD` são tratadas como variáveis categóricas e codificadas com One-Hot Encoding.
- As demais variáveis explicativas quantitativas são preenchidas pela mediana quando necessário e normalizadas com `StandardScaler`.
- Valores ausentes categóricos são preenchidos pela categoria mais frequente.
- `MEDV` foi excluída dos atributos de agrupamento por ser a variável-alvo/valor da residência no conjunto Boston Housing original. O modelo utiliza os 13 atributos explicativos.

## Como executar
Requer Python 3.10 ou superior.

1. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```
2. Treine o modelo:
   ```bash
   python train.py
   ```
3. Faça a inferência:
   ```bash
   python predict.py
   ```
   Informe os 13 valores solicitados no terminal. Para `CHAS`, use 0 ou 1. Para `RAD`, informe uma das categorias presentes na base.

## Resultado do treinamento já executado
- K escolhido pelo método do cotovelo: **3**
- Silhouette score (referência): **0.2516**
- O artefato `modelo_kmeans_boston.joblib` já está incluído. É possível reproduzir o treinamento executando `train.py`.

## Observação
O centroide é calculado no espaço transformado pelo pré-processamento. Na inferência, as variáveis quantitativas são convertidas de volta às unidades originais e as variáveis categóricas são apresentadas pela categoria de maior representação no centroide.
