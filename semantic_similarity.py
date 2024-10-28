import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

ANNOTATION_PATH = os.getenv('ANNOTATION_PATH')
OUTPUT_PATH = os.getenv('OUTPUT_PATH')

# Load JSON data
with open(ANNOTATION_PATH, 'r') as file:
  data = json.load(file)

# Extract gesture names, human and GPT annotations
gesture_names = [gesture['name'] for gesture in data['gestures']]
human_annotations = [gesture['human_annotation'] for gesture in data['gestures']]
gpt_annotations = [gesture['gpt_annotation'] for gesture in data['gestures']]

# Load the SBERT model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Generate embeddings for annotations
human_embeddings = model.encode(human_annotations)
gpt_embeddings = model.encode(gpt_annotations)

# Compute cosine similarity for each gesture
similarities = [cosine_similarity([human], [gpt])[0][0] for human, gpt in zip(human_embeddings, gpt_embeddings)]
print("Similarities", similarities)

# Create a DataFrame for easy visualization
df = pd.DataFrame({'Gesture': gesture_names, 'Similarity': similarities})

# Save the DataFrame to a CSV file
df.to_csv(f'{OUTPUT_PATH}/similarity_results.csv', index=False)

# Plot the similarity as a bar chart
plt.figure(figsize=(10, 8))
plt.bar(df['Gesture'], df['Similarity'], color='skyblue')
plt.xticks(rotation=45)
plt.title('Similaridade do cosseno entre anotações humanas e do GPT-4o')
plt.xlabel('Gestos')
plt.ylabel('Similaridade do cosseno')

# Save the plot as an image file
plt.savefig(f'{OUTPUT_PATH}/similarity_chart_new.png')

# Show the plot
plt.show()
