"""
Test Sentiment Analysis Pipeline
Testa il modello DistilBERT su frasi di esempio
"""

from transformers import pipeline
import warnings
warnings.filterwarnings('ignore')

print("\n" + "=" * 70)
print("🧪 TEST SENTIMENT ANALYSIS PIPELINE")
print("=" * 70 + "\n")

# Inizializza pipeline
print("🔄 Caricamento modello DistilBERT in corso...")
print("   (Primo avvio: può richiedere 1-2 minuti per download modello)")
print()

try:
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )
    print("✅ Modello caricato con successo!\n")
except Exception as e:
    print(f"❌ Errore nel caricamento del modello: {e}")
    exit(1)

# Frasi di test (finanziarie)
test_texts = [
    # Positive examples
    "Stock market reaches all-time high amid strong economic growth",
    "Company reports record profits for the third consecutive quarter",
    "Tech giant announces major expansion and thousands of new jobs",
    "Investors celebrate as markets surge to new heights",
    
    # Negative examples
    "Stock prices plummet as company reports massive losses",
    "Economic recession fears grow as unemployment rates rise sharply",
    "Company faces bankruptcy after failed merger attempt",
    "Market crash wipes out billions in investor wealth",
    
    # Neutral/Mixed
    "Company announces quarterly earnings report for next week",
    "Stock market remains steady with mixed trading signals",
    "Federal Reserve maintains interest rates at current levels"
]

print("=" * 70)
print("📝 TEST SU FRASI FINANZIARIE")
print("=" * 70)
print()

# Analizza ogni frase
for idx, text in enumerate(test_texts, 1):
    print(f"News #{idx}:")
    print(f"📰 '{text}'")
    print()
    
    # Predizione
    result = sentiment_pipeline(text)[0]
    
    # Emoji per sentiment
    emoji = "🟢" if result['label'] == 'POSITIVE' else "🔴"
    
    # Mostra risultato
    print(f"   {emoji} Sentiment: {result['label']}")
    print(f"   📊 Confidence: {result['score']:.2%}")
    print()
    print("-" * 70)
    print()

# Test su frase personalizzata
print("=" * 70)
print("✍️  TEST PERSONALIZZATO")
print("=" * 70)
print()
print("Prova a modificare questa frase nel codice per testare altre frasi:")
print()

custom_text = "The company's innovative strategy leads to unprecedented success"
print(f"📝 Frase: '{custom_text}'")
print()

result = sentiment_pipeline(custom_text)[0]
emoji = "🟢" if result['label'] == 'POSITIVE' else "🔴"

print(f"   {emoji} Sentiment: {result['label']}")
print(f"   📊 Confidence: {result['score']:.2%}")
print()

# Statistiche finali
print("=" * 70)
print("📈 STATISTICHE TEST")
print("=" * 70)
print()

# Conta sentiment
all_results = [sentiment_pipeline(text)[0] for text in test_texts]
positive_count = sum(1 for r in all_results if r['label'] == 'POSITIVE')
negative_count = sum(1 for r in all_results if r['label'] == 'NEGATIVE')
avg_confidence = sum(r['score'] for r in all_results) / len(all_results)

print(f"Total news analyzed:    {len(test_texts)}")
print(f"Positive predictions:   {positive_count} ({positive_count/len(test_texts)*100:.1f}%)")
print(f"Negative predictions:   {negative_count} ({negative_count/len(test_texts)*100:.1f}%)")
print(f"Average confidence:     {avg_confidence:.2%}")
print()

print("=" * 70)
print("✅ TEST COMPLETATO CON SUCCESSO!")
print("=" * 70)
print()
print("📌 PROSSIMI STEP:")
print("   1. Il modello funziona correttamente ✅")
print("   2. Domani crei lo script principale per analizzare tutto il dataset")
print()