import os
import json
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import matplotlib.pyplot as plt

ANNOTATION_PATH = os.getenv('ANNOTATION_PATH')
IMAGE_PATH = os.getenv('IMAGE_PATH')
OUTPUT_PATH = os.getenv('OUTPUT_PATH')

# Load JSON data
json_path = ANNOTATION_PATH
with open(json_path, 'r') as file:
  data = json.load(file)

# Function to load and convert an image to a pixel vector
def load_image_as_vector(image_path):
  image = Image.open(image_path).convert('L')  # Convert to grayscale
  image = image.resize((256, 256))  # Resize to a standard size
  image_array = np.array(image).flatten()  # Convert the image to a 1D vector
  return image_array

# Lists to store results
gesture_names = []
image_similarities = []

# Process each gesture
for gesture in data['gestures']:
  human_image_path = f"{IMAGE_PATH}/{gesture['human_image']}"
  gpt_image_path = f"{IMAGE_PATH}/{gesture['gpt_image']}"
  
  human_image_vector = load_image_as_vector(human_image_path)
  gpt_image_vector = load_image_as_vector(gpt_image_path)
  
  similarity = cosine_similarity([human_image_vector], [gpt_image_vector])[0][0]
  
  gesture_names.append(gesture['name'])
  image_similarities.append(similarity)

# Create a DataFrame and save to CSV
df = pd.DataFrame({
  'Gesture': gesture_names,
  'Image Similarity': image_similarities
})
csv_path = f'{OUTPUT_PATH}/image_similarity_results.csv'
df.to_csv(csv_path, index=False)

# Plot the results in a bar chart
plt.figure(figsize=(10, 6))
plt.bar(gesture_names, image_similarities, color='skyblue')
plt.xlabel('Gesto')  # Translated to Portuguese
plt.ylabel('Similaridade de Imagem')  # Translated to Portuguese
plt.title('Similaridade de Imagem entre Imagens Humanas e Geradas pelo GPT')  # Translated to Portuguese
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(f'{OUTPUT_PATH}/image_similarity_results_new.png')
plt.show()
