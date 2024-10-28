import os
import cv2
import numpy as np
import json
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

ANNOTATION_PATH = os.getenv('ANNOTATION_PATH')
IMAGE_PATH = os.getenv('IMAGE_PATH')
OUTPUT_PATH = os.getenv('OUTPUT_PATH')

# Função para calcular o histograma de uma imagem
def calc_histogram(image_path):
  image = cv2.imread(image_path)
  hist = cv2.calcHist([image], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
  hist = cv2.normalize(hist, hist).flatten()
  return hist

# Carregar o arquivo JSON
with open(ANNOTATION_PATH, 'r') as file:
  data = json.load(file)

# Listas para armazenar os resultados
gesture_names = []
histogram_similarities = []

# Calcular a similaridade de histograma para cada gesto
for gesture in data['gestures']:
  human_image_path = f"{IMAGE_PATH}/{gesture['human_image']}"
  gpt_image_path = f"{IMAGE_PATH}/{gesture['gpt_image']}"
  
  human_hist = calc_histogram(human_image_path)
  gpt_hist = calc_histogram(gpt_image_path)
  
  similarity = cosine_similarity([human_hist], [gpt_hist])[0][0]
  
  gesture_names.append(gesture['name'])
  histogram_similarities.append(similarity)

# Criar um DataFrame com os resultados
df = pd.DataFrame({
  'Gesture': gesture_names,
  'Histogram Similarity': histogram_similarities
})

# Salvar os resultados em um arquivo CSV
df.to_csv(f'{OUTPUT_PATH}/histogram_similarity_results.csv', index=False)

# Gerar um gráfico de barras
plt.figure(figsize=(10, 6))
plt.bar(gesture_names, histogram_similarities, color='skyblue')
plt.xlabel('Gestos')
plt.ylabel('Similaridade de Histograma')
plt.title('Similaridade de Histograma entre Imagens Humanas e GPT')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Salvar o gráfico como imagem
plt.savefig(f'{OUTPUT_PATH}/histogram_similarity_chart.png')

# Exibir o gráfico
plt.show()
