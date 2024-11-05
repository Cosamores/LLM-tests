import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from skimage.metrics import structural_similarity as ssim
from keras.applications import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing import image
from keras.models import Model

ANNOTATION_PATH = os.getenv('ANNOTATION_PATH')
IMAGE_PATH = os.getenv('IMAGE_PATH')
OUTPUT_PATH = os.getenv('OUTPUT_PATH')

# Carregar dados JSON
with open(ANNOTATION_PATH, 'r') as file:
  data = json.load(file)

# Modelo pré-treinado para extrair embeddings
base_model = ResNet50(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)

resultados_ssim = []
resultados_cosine = []

# Função para calcular SSIM entre imagens
def calcular_ssim(imageA, imageB):
  return ssim(imageA, imageB, win_size=7, channel_axis=2)

# Função para calcular embeddings da imagem
def extrair_embedding(img_path):
  img = image.load_img(img_path, target_size=(224, 224))
  img_data = image.img_to_array(img)
  img_data = np.expand_dims(img_data, axis=0)
  img_data = preprocess_input(img_data)
  embedding = model.predict(img_data)
  return embedding.flatten()

# Diretório com as imagens
img_dir = IMAGE_PATH

# Loop pelas anotações e imagens
for gesture in data['gestures']:
  original_path = os.path.join(img_dir, gesture['original_image'])
  human_path = os.path.join(img_dir, gesture['human_image'])
  gpt_path = os.path.join(img_dir, gesture['gpt_image'])
  
  # Carregar imagens
  img_original = cv2.imread(original_path)
  img_human = cv2.imread(human_path)
  img_gpt = cv2.imread(gpt_path)
  
  # Verificar se as imagens foram carregadas corretamente
  if img_original is None:
    print(f"Erro ao carregar a imagem original: {original_path}")
    continue
  if img_human is None:
    print(f"Erro ao carregar a imagem humana: {human_path}")
    continue
  if img_gpt is None:
    print(f"Erro ao carregar a imagem GPT: {gpt_path}")
    continue

  # Redimensionar imagens para ter as mesmas dimensões
  img_human = cv2.resize(img_human, (img_original.shape[1], img_original.shape[0]))
  img_gpt = cv2.resize(img_gpt, (img_original.shape[1], img_original.shape[0]))

  # Calcular SSIM
  ssim_human = calcular_ssim(img_original, img_human)
  ssim_gpt = calcular_ssim(img_original, img_gpt)
  resultados_ssim.append({
    'Gesto': gesture['name'],
    'SSIM Humano': ssim_human,
    'SSIM GPT': ssim_gpt
  })
  
  # Extrair embeddings e calcular Cosine Similarity
  emb_original = extrair_embedding(original_path)
  emb_human = extrair_embedding(human_path)
  emb_gpt = extrair_embedding(gpt_path)
  
  cosine_human = cosine_similarity([emb_original], [emb_human])[0][0]
  cosine_gpt = cosine_similarity([emb_original], [emb_gpt])[0][0]
  resultados_cosine.append({
    'Gesto': gesture['name'],
    'Cosine Humano': cosine_human,
    'Cosine GPT': cosine_gpt
  })

# Organizar dados para plotagem
gestos = [res['Gesto'] for res in resultados_ssim]
ssim_human = [res['SSIM Humano'] for res in resultados_ssim]
ssim_gpt = [res['SSIM GPT'] for res in resultados_ssim]
cosine_human = [res['Cosine Humano'] for res in resultados_cosine]
cosine_gpt = [res['Cosine GPT'] for res in resultados_cosine]

# Plotagem dos resultados - SSIM
plt.figure(figsize=(10, 6))
bar_width = 0.35
index = np.arange(len(gestos))

plt.bar(index, ssim_human, bar_width, label='SSIM - Anotação Humana', color='blue')
plt.bar(index + bar_width, ssim_gpt, bar_width, label='SSIM - Anotação GPT', color='green')

plt.xlabel('Gestos')
plt.ylabel('Similaridade Estrutural (SSIM)')
plt.title('Comparação de SSIM - Imagens Originais vs Anotações')
plt.xticks(index + bar_width / 2, gestos, rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# Plotagem dos resultados - Cosine Similarity
plt.figure(figsize=(10, 6))

plt.bar(index, cosine_human, bar_width, label='Cosine Similarity - Anotação Humana', color='blue')
plt.bar(index + bar_width, cosine_gpt, bar_width, label='Cosine Similarity - Anotação GPT', color='green')

plt.xlabel('Gestos')
plt.ylabel('Similaridade Cosine')
plt.title('Comparação de Similaridade Cosine - Imagens Originais vs Anotações')
plt.xticks(index + bar_width / 2, gestos, rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

save_path = OUTPUT_PATH

# Salvar resultados como CSV
df_ssim = pd.DataFrame(resultados_ssim)
df_cosine = pd.DataFrame(resultados_cosine)
df_ssim.to_csv(f'{save_path}/ssim_results.csv', index=False)
df_cosine.to_csv(f'{save_path}/cosine_results.csv', index=False)

# Imprimir resultados
for res_ssim, res_cosine in zip(resultados_ssim, resultados_cosine):
  print(f"Gesto: {res_ssim['Gesto']}, SSIM Humano: {res_ssim['SSIM Humano']:.4f}, SSIM GPT: {res_ssim['SSIM GPT']:.4f}, "
      f"Cosine Humano: {res_cosine['Cosine Humano']:.4f}, Cosine GPT: {res_cosine['Cosine GPT']:.4f}")
