"""
Esplorazione Dataset Financial News
Analizza la struttura e il contenuto del dataset Kaggle
"""

import pandas as pd
import os

print("\n" + "=" * 70)
print("📊 ESPLORAZIONE DATASET - FINANCIAL NEWS SENTIMENT")
print("=" * 70 + "\n")

# Verifica esistenza file
dataset_path = 'data/all-data.csv'

if not os.path.exists(dataset_path):
    print("❌ ERRORE: Dataset non trovato!")
    print(f"   Percorso cercato: {dataset_path}")
    print()
    print("📥 SOLUZIONE:")
    print("   1. Scarica dataset da:")
    print("      https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news")
    print("   2. Metti il file 'all-data.csv' nella cartella 'data/'")
    print()
    exit(1)

print(f"✅ Dataset trovato: {dataset_path}\n")

# Carica dataset
print("📂 Caricamento dataset in corso...")
try:
    df = pd.read_csv(dataset_path, encoding='latin-1', names=['sentiment', 'text'])
    print("✅ Dataset caricato con successo!\n")
except Exception as e:
    print(f"❌ Errore nel caricamento: {e}")
    exit(1)

# Informazioni generali
print("=" * 70)
print("📋 INFORMAZIONI GENERALI")
print("=" * 70)
print(f"Numero totale di news:  {len(df):,}")
print(f"Numero di colonne:      {len(df.columns)}")
print(f"Nomi colonne:           {list(df.columns)}")
print(f"Memoria utilizzata:     {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
print()

# Dimensioni dataset
print("=" * 70)
print("📐 DIMENSIONI DATASET")
print("=" * 70)
print(f"Righe (news):    {df.shape[0]:,}")
print(f"Colonne:         {df.shape[1]}")
print()

# Prime righe
print("=" * 70)
print("👀 PRIME 5 NEWS DEL DATASET")
print("=" * 70)
print(df.head())
print()

# Distribuzione sentiment
print("=" * 70)
print("📊 DISTRIBUZIONE SENTIMENT")
print("=" * 70)
sentiment_counts = df['sentiment'].value_counts()
print(sentiment_counts)
print()

# Percentuali
print("Percentuali:")
for sentiment, count in sentiment_counts.items():
    percentage = (count / len(df)) * 100
    print(f"  {sentiment:10} {count:6,} ({percentage:5.1f}%)")
print()

# Valori mancanti
print("=" * 70)
print("🔍 VALORI MANCANTI")
print("=" * 70)
missing = df.isnull().sum()
if missing.sum() == 0:
    print("✅ Nessun valore mancante!")
else:
    print(missing)
print()

# Statistiche testo
print("=" * 70)
print("📝 STATISTICHE LUNGHEZZA TESTO")
print("=" * 70)
df['text_length'] = df['text'].str.len()
print(f"Lunghezza media:     {df['text_length'].mean():.0f} caratteri")
print(f"Lunghezza minima:    {df['text_length'].min()} caratteri")
print(f"Lunghezza massima:   {df['text_length'].max()} caratteri")
print(f"Mediana:             {df['text_length'].median():.0f} caratteri")
print()

# Esempi per categoria
print("=" * 70)
print("💬 ESEMPI DI NEWS PER SENTIMENT")
print("=" * 70)
for sentiment in df['sentiment'].unique():
    print(f"\n🏷️  {sentiment.upper()}:")
    print("-" * 70)
    # Prendi 3 esempi casuali
    examples = df[df['sentiment'] == sentiment].sample(min(3, len(df[df['sentiment'] == sentiment])))
    for idx, (_, row) in enumerate(examples.iterrows(), 1):
        print(f"{idx}. {row['text'][:200]}...")
        print()

# Parole più comuni (analisi veloce)
print("=" * 70)
print("🔤 TOP 10 PAROLE PIÙ COMUNI (length > 4 caratteri)")
print("=" * 70)
from collections import Counter
import re

# Estrai tutte le parole
all_text = ' '.join(df['text'].str.lower())
words = re.findall(r'\b[a-z]{5,}\b', all_text)  # Solo parole > 4 caratteri

# Conta frequenze
word_freq = Counter(words)
top_words = word_freq.most_common(10)

for idx, (word, count) in enumerate(top_words, 1):
    print(f"{idx:2}. {word:15} → {count:5,} volte")
print()

# Riepilogo finale
print("=" * 70)
print("✅ ESPLORAZIONE COMPLETATA")
print("=" * 70)
print()
print("📌 PROSSIMI STEP:")
print("   ✅ GIORNO 1 COMPLETATO!")
print("   📅 Domani (Giorno 2): Analisi completa del dataset")
print()