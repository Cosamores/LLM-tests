import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from rouge_score import rouge_scorer

ANNOTATION_PATH = os.getenv('ANNOTATION_PATH')
IMAGE_PATH = os.getenv('IMAGE_PATH')
OUTPUT_PATH = os.getenv('OUTPUT_PATH')

# Load the JSON file
with open(ANNOTATION_PATH, 'r') as file:
  data = json.load(file)

# Extract human and GPT annotations
human_annotations = [gesture['human_annotation'] for gesture in data['gestures']]
gpt_annotations = [gesture['gpt_annotation'] for gesture in data['gestures']]

# Create the ROUGE scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

# Calculate the ROUGE score for each gesture
rouge_scores = [scorer.score(human, gpt) for human, gpt in zip(human_annotations, gpt_annotations)]

# Extract the rouge1 f-measure for each gesture
rouge1_scores = [score['rouge1'].fmeasure for score in rouge_scores]

# Create a DataFrame with the results
df = pd.DataFrame({
  'Gesto': [gesture['name'] for gesture in data['gestures']],
  'Pontuação ROUGE': rouge1_scores
})

# Save the results to a CSV file
df.to_csv('{OUTPUT_PATH}/similarity_results_rouge.csv', index=False)

# Generate a chart with the results
plt.figure(figsize=(10, 6))
plt.bar(df['Gesto'], df['Pontuação ROUGE'], color='skyblue')
plt.title('Pontuações ROUGE-1 F-measure para Cada Gesto')
plt.xlabel('Gesto')
plt.ylabel('Pontuação ROUGE-1 F-measure')
plt.grid(True)
plt.savefig('{OUTPUT_PATH}/similarity_results_rouge_chart.png')
plt.show()
