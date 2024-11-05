import os
import json
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model

ANNOTATION_PATH = os.getenv('ANNOTATION_PATH')
IMAGE_PATH = os.getenv('IMAGE_PATH')
OUTPUT_PATH = os.getenv('OUTPUT_PATH')

# Carregar dados JSON
with open(ANNOTATION_PATH, 'r') as file:
  data = json.load(file)

# Carregar o modelo VGG16
model = VGG16(weights='imagenet', include_top=False)
model = Model(inputs=model.inputs, outputs=model.layers[-1].output)

# Função para extrair características
def extrair_features(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    features = model.predict(img_data)
    return features.flatten()

resultados_cnn = []

img_dir = IMAGE_PATH

# Loop pelas anotações e imagens
for gesture in data['gestures']:
    original_path = os.path.join(img_dir, gesture['original_image'])
    human_path = os.path.join(img_dir, gesture['human_image'])
    gpt_path = os.path.join(img_dir, gesture['gpt_image'])
    
    # Extrair características
    feat_original = extrair_features(original_path)
    feat_human = extrair_features(human_path)
    feat_gpt = extrair_features(gpt_path)
    
    # Calcular similaridade euclidiana
    dist_human = np.linalg.norm(feat_original - feat_human)
    dist_gpt = np.linalg.norm(feat_original - feat_gpt)
    
    resultados_cnn.append({
        'Gesto': gesture['name'],
        'Distância Euclidiana Humano': dist_human,
        'Distância Euclidiana GPT': dist_gpt
    })

# Plotagem dos resultados
dist_human = [res['Distância Euclidiana Humano'] for res in resultados_cnn]
dist_gpt = [res['Distância Euclidiana GPT'] for res in resultados_cnn]
gestos = [res['Gesto'] for res in resultados_cnn]

x = np.arange(len(gestos))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, dist_human, width, label='Distância Euclidiana - Anotação Humana')
rects2 = ax.bar(x + width/2, dist_gpt, width, label='Distância Euclidiana - Anotação GPT')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('Gestos')
ax.set_ylabel('Distância Euclidiana')
ax.set_title('Similaridade de Características Extraídas com CNN')
ax.set_xticks(x)
ax.set_xticklabels(gestos, rotation=45)
ax.legend()

fig.tight_layout()
plt.grid()
plt.show()

# Salvar os resultados em um arquivo CSV
df = pd.DataFrame(resultados_cnn)
df.to_csv(f'{OUTPUT_PATH}/cnn_similarity_results.csv', index=False)

# Imprimir resultados
for res in resultados_cnn:
    print(f"Gesto: {res['Gesto']}, Distância Humano: {res['Distância Euclidiana Humano']:.4f}, Distância GPT: {res['Distância Euclidiana GPT']:.4f}")
