import json
import os
import cv2
from matplotlib import pyplot as plt
import numpy as np

ANNOTATION_PATH = os.getenv('ANNOTATION_PATH')
IMAGE_PATH = os.getenv('IMAGE_PATH')
OUTPUT_PATH = os.getenv('OUTPUT_PATH')

# Carregar dados JSON
with open(ANNOTATION_PATH, 'r') as file:
  data = json.load(file)

# Função para calcular MSE e RMSE
def calcular_mse_rmse(imageA, imageB):
  if imageA.shape != imageB.shape:
    raise ValueError("As imagens devem ter o mesmo tamanho e forma para calcular MSE e RMSE.")
  mse = np.mean((imageA - imageB) ** 2)
  rmse = np.sqrt(mse)
  return mse, rmse

resultados_rmse = []

img_dir = IMAGE_PATH

# Loop pelas anotações e imagens
for gesture in data['gestures']:
  original_path = os.path.join(img_dir, gesture['original_image'])
  human_path = os.path.join(img_dir, gesture['human_image'])
  gpt_path = os.path.join(img_dir, gesture['gpt_image'])
  
  # Carregar imagens e converter para escala de cinza
  img_original = cv2.imread(original_path, cv2.IMREAD_GRAYSCALE)
  img_human = cv2.imread(human_path, cv2.IMREAD_GRAYSCALE)
  img_gpt = cv2.imread(gpt_path, cv2.IMREAD_GRAYSCALE)
  
  # Redimensionar imagens para o mesmo tamanho
  img_human = cv2.resize(img_human, (img_original.shape[1], img_original.shape[0]))
  img_gpt = cv2.resize(img_gpt, (img_original.shape[1], img_original.shape[0]))
  
  # Calcular MSE e RMSE
  mse_human, rmse_human = calcular_mse_rmse(img_original, img_human)
  mse_gpt, rmse_gpt = calcular_mse_rmse(img_original, img_gpt)
  
  resultados_rmse.append({
    'Gesto': gesture['name'],
    'MSE Humano': mse_human,
    'RMSE Humano': rmse_human,
    'MSE GPT': mse_gpt,
    'RMSE GPT': rmse_gpt
  })

# Plotagem dos resultados
gestos = [res['Gesto'] for res in resultados_rmse]
rmse_human = [res['RMSE Humano'] for res in resultados_rmse]
rmse_gpt = [res['RMSE GPT'] for res in resultados_rmse]

x = np.arange(len(gestos))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width/2, rmse_human, width, label='RMSE - Anotação Humana')
bars2 = ax.bar(x + width/2, rmse_gpt, width, label='RMSE - Anotação GPT')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('Gestos')
ax.set_ylabel('Raiz do Erro Quadrático Médio (RMSE)')
ax.set_title('Comparação de RMSE entre Imagens')
ax.set_xticks(x)
ax.set_xticklabels(gestos, rotation=45)
ax.legend()

# Set y-axis limit to add extra space above the tallest bar
max_height = max(max(rmse_human), max(rmse_gpt))
ax.set_ylim(0, max_height * 1.2)

# Attach a text label above each bar in *bars*, displaying its height.
def autolabel(bars):
  """Attach a text label above each bar in *bars*, displaying its height."""
  for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height:.2f}',
          xy=(bar.get_x() + bar.get_width() / 2, height),
          xytext=(0, 3),  # 3 points vertical offset
          textcoords="offset points",
          ha='center', va='bottom')

autolabel(bars1)
autolabel(bars2)

fig.tight_layout()
plt.grid()
plt.show()

# Imprimir resultados
for res in resultados_rmse:
  print(f"Gesto: {res['Gesto']}, RMSE Humano: {res['RMSE Humano']:.4f}, RMSE GPT: {res['RMSE GPT']:.4f}")
