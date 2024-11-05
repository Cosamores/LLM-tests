import os
import json
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from skimage.metrics import structural_similarity as ssim

ANNOTATION_PATH = os.getenv('ANNOTATION_PATH')
IMAGE_PATH = os.getenv('IMAGE_PATH')
OUTPUT_PATH = os.getenv('OUTPUT_PATH')

# Carregar dados JSON
with open(ANNOTATION_PATH, 'r') as file:
  data = json.load(file)

resultados_ssim = []

# Função para calcular SSIM entre imagens
def calcular_ssim(imageA, imageB):
  return ssim(imageA, imageB, win_size=7, channel_axis=2)

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

# Plotagem dos resultados
gestos = [res['Gesto'] for res in resultados_ssim]
ssim_human = [res['SSIM Humano'] for res in resultados_ssim]
ssim_gpt = [res['SSIM GPT'] for res in resultados_ssim]

x = range(len(gestos))

plt.figure(figsize=(10, 6))
plt.bar(x, ssim_human, width=0.4, label='SSIM - Anotação Humana', align='center')
plt.bar(x, ssim_gpt, width=0.4, label='SSIM - Anotação GPT', align='edge')
plt.xlabel('Gestos')
plt.ylabel('Similaridade Estrutural (SSIM)')
plt.title('Comparação de Similaridade Estrutural entre Imagens')
plt.legend()
plt.xticks(x, gestos, rotation=45)
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# Salvar resultados em um arquivo CSV
df = pd.DataFrame(resultados_ssim)
df.to_csv(f'{OUTPUT_PATH}/ssim_results.csv', index=False)


# Imprimir resultados
for res in resultados_ssim:
  print(f"Gesto: {res['Gesto']}, SSIM Humano: {res['SSIM Humano']:.4f}, SSIM GPT: {res['SSIM GPT']:.4f}")
