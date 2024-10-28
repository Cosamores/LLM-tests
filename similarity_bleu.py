import os
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import pandas as pd
import json
import matplotlib.pyplot as plt

ANNOTATION_PATH = os.getenv('ANNOTATION_PATH')
OUTPUT_PATH = os.getenv('OUTPUT_PATH')

# Carregar os dados JSON
with open(ANNOTATION_PATH, 'r') as file:
  data = json.load(file)

# Extrair os nomes dos gestos, anotações humanas e anotações GPT
gesture_names = [gesture['name'] for gesture in data['gestures']]
human_annotations = [gesture['human_annotation'] for gesture in data['gestures']]
gpt_annotations = [gesture['gpt_annotation'] for gesture in data['gestures']]

# Criar um DataFrame para visualização fácil
df = pd.DataFrame({'Gesto': gesture_names, 'Anotação humana': human_annotations, 'Anotação do GPT-4o': gpt_annotations})

# Calcular o BLEU score para cada gesto com SmoothingFunction
smoothing_function = SmoothingFunction().method1
bleu_scores = [sentence_bleu([human.split()], gpt.split(), smoothing_function=smoothing_function) for human, gpt in zip(human_annotations, gpt_annotations)]

# Exibir os resultados
print("Pontuação BLEU", bleu_scores)

# Adicionar ao DataFrame e salvar como CSV
df['Pontuação BLEU'] = bleu_scores
df.to_csv(f'{OUTPUT_PATH}/similarity_results_bleu.csv', index=False)

# Visualizar os resultados em um gráfico de barras
plt.figure(figsize=(10, 6))
plt.bar(df['Gesto'], df['Pontuação BLEU'], color='skyblue')
plt.xlabel('Gesto')
plt.ylabel('Pontuação BLEU')
plt.title('Pontuação BLEU para anotações humanas vs anotações do GPT-4o')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(f'{OUTPUT_PATH}/similarity_results_bleu_chart_new.png')
plt.show()
