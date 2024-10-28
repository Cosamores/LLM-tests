import os
import json
import pandas as pd
from skimage.metrics import structural_similarity as ssim
import cv2
import matplotlib.pyplot as plt

ANNOTATION_PATH = os.getenv('ANNOTATION_PATH')
IMAGE_PATH = os.getenv('IMAGE_PATH')
OUTPUT_PATH = os.getenv('OUTPUT_PATH')

# Função para carregar e converter a imagem para um formato comparável
def load_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Warning: Unable to load image at {image_path}")
    return image

# Função para redimensionar a imagem para um tamanho fixo
def resize_image(image, size=(256, 256)):
    return cv2.resize(image, size)

# Carregar o arquivo JSON
with open(ANNOTATION_PATH, 'r') as file:
    data = json.load(file)

# Extrair os caminhos das imagens e os nomes dos gestos
gestures = data['gestures']
human_image_paths = [gesture['human_image'] for gesture in gestures]
gpt_image_paths = [gesture['gpt_image'] for gesture in gestures]
gesture_names = [gesture['name'] for gesture in gestures]

# Carregar as imagens geradas por humanos e GPT
human_images = [load_image(f'{IMAGE_PATH}/{image}') for image in human_image_paths]
gpt_images = [load_image(f'{IMAGE_PATH}/{image}') for image in gpt_image_paths]

# Filtrar pares de imagens válidos e os nomes dos gestos correspondentes
valid_pairs = []
valid_gesture_names = []
for human, gpt, name in zip(human_images, gpt_images, gesture_names):
    if human is not None and gpt is not None:
        valid_pairs.append((human, gpt))
        valid_gesture_names.append(name)

# Redimensionar as imagens para o mesmo tamanho
resized_pairs = [(resize_image(human), resize_image(gpt)) for human, gpt in valid_pairs]

# Calcular o SSIM para cada par de imagens válidas
ssim_scores = [ssim(human, gpt) for human, gpt in resized_pairs]

# Exibir os resultados
print("SSIM Scores", ssim_scores)

# Adicionar ao DataFrame e salvar como CSV
df = pd.DataFrame({'Gesture': valid_gesture_names, 'SSIM Score': ssim_scores})
df.to_csv(f'{OUTPUT_PATH}/ssim_results.csv', index=False)

# Criar um gráfico de barras para exibir os resultados
plt.figure(figsize=(10, 6))
bars = plt.bar(valid_gesture_names, ssim_scores, color='blue')
plt.xlabel('Gestos')
plt.ylabel('Pontuação SSIM')
plt.title('Comparação de Imagens Humanas e GPT por Pontuação SSIM')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Adicionar rótulos com os nomes dos gestos
for bar, score in zip(bars, ssim_scores):
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval, round(score, 2), ha='center', va='bottom')

plt.savefig(f'{OUTPUT_PATH}/ssim_comparison_chart.png')
plt.show()