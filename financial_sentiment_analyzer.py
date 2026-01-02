"""
Financial Sentiment Analyzer
Analizza il sentiment di news finanziarie usando DistilBERT
"""

import pandas as pd
from transformers import pipeline
import warnings
import os
from datetime import datetime

warnings.filterwarnings('ignore')

class FinancialSentimentAnalyzer:
    """
    Classe principale per l'analisi del sentiment di news finanziarie
    """
    
    def __init__(self, data_path):
        """
        Inizializza l'analyzer
        
        Args:
            data_path (str): Percorso del file CSV con le news
        """
        print("\n" + "=" * 70)
        print("🚀 FINANCIAL SENTIMENT ANALYZER")
        print("=" * 70)
        print()
        print("📊 Inizializzazione in corso...")
        
        self.data_path = data_path
        self.df = None
        
        # Inizializza pipeline sentiment
        print("🤖 Caricamento modello DistilBERT...")
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )
        print("✅ Modello caricato!\n")
    
    def load_data(self, sample_size=None):
        """
        Carica il dataset
        
        Args:
            sample_size (int, optional): Numero di news da campionare per test veloce
        
        Returns:
            pd.DataFrame: Dataset caricato
        """
        print("=" * 70)
        print("📂 CARICAMENTO DATASET")
        print("=" * 70)
        
        # Verifica esistenza file
        if not os.path.exists(self.data_path):
            print(f"❌ ERRORE: File non trovato: {self.data_path}")
            exit(1)
        
        # Carica CSV
        print(f"📥 Lettura da: {self.data_path}")
        self.df = pd.read_csv(
            self.data_path, 
            encoding='latin-1', 
            names=['original_sentiment', 'text']
        )
        
        # Sample se richiesto
        if sample_size:
            print(f"🎲 Campionamento casuale di {sample_size} news...")
            self.df = self.df.sample(sample_size, random_state=42)
            print(f"✅ Sample di {len(self.df)} news selezionato")
        else:
            print(f"✅ Dataset completo caricato: {len(self.df):,} news")
        
        print()
        return self.df
    
    def analyze_sentiment(self):
        """
        Analizza il sentiment di tutte le news nel dataset
        
        Returns:
            pd.DataFrame: Dataset con sentiment predetti
        """
        print("=" * 70)
        print("🔍 ANALISI SENTIMENT IN CORSO")
        print("=" * 70)
        print()
        
        total = len(self.df)
        sentiments = []
        scores = []
        
        print(f"📊 Analisi di {total:,} news...")
        print("⏳ Questo potrebbe richiedere alcuni minuti...\n")
        
        # Analizza ogni news
        for idx, text in enumerate(self.df['text'], 1):
            # Progress indicator ogni 100 news
            if idx % 100 == 0 or idx == 1:
                percentage = (idx / total) * 100
                print(f"   Progresso: {idx:,}/{total:,} ({percentage:.1f}%)")
            
            try:
                # Tronca a 512 caratteri (limite modello)
                text_truncated = text[:512]
                
                # Predizione
                result = self.sentiment_pipeline(text_truncated)[0]
                
                sentiments.append(result['label'])
                scores.append(result['score'])
                
            except Exception as e:
                # In caso di errore, assegna valore neutro
                print(f"   ⚠️  Errore su news #{idx}: {str(e)[:50]}")
                sentiments.append('NEUTRAL')
                scores.append(0.5)
        
        # Aggiungi risultati al dataframe
        self.df['predicted_sentiment'] = sentiments
        self.df['confidence'] = scores
        
        print()
        print("✅ Analisi completata!\n")
        
        return self.df
    
    def generate_statistics(self):
        """
        Genera statistiche descrittive sui risultati
        """
        print("=" * 70)
        print("📈 STATISTICHE ANALISI")
        print("=" * 70)
        print()
        
        # 1. Distribuzione sentiment predetti
        print("1️⃣  DISTRIBUZIONE SENTIMENT PREDETTI")
        print("-" * 70)
        sentiment_dist = self.df['predicted_sentiment'].value_counts()
        for sentiment, count in sentiment_dist.items():
            percentage = (count / len(self.df)) * 100
            bar = "█" * int(percentage / 2)
            print(f"   {sentiment:10} {count:5,} ({percentage:5.1f}%)  {bar}")
        print()
        
        # 2. Confidence statistics
        print("2️⃣  CONFIDENCE SCORES")
        print("-" * 70)
        print(f"   Media:    {self.df['confidence'].mean():.2%}")
        print(f"   Mediana:  {self.df['confidence'].median():.2%}")
        print(f"   Min:      {self.df['confidence'].min():.2%}")
        print(f"   Max:      {self.df['confidence'].max():.2%}")
        print()
        
        # 3. Confronto con sentiment originale
        print("3️⃣  CONFRONTO CON SENTIMENT ORIGINALE")
        print("-" * 70)
        
        # Mappa sentiment per confronto (case-insensitive)
        self.df['original_lower'] = self.df['original_sentiment'].str.lower()
        self.df['predicted_lower'] = self.df['predicted_sentiment'].str.lower()
        
        # Calcola accuracy
        matches = (self.df['original_lower'] == self.df['predicted_lower']).sum()
        accuracy = (matches / len(self.df)) * 100
        
        print(f"   Match:      {matches:,}/{len(self.df):,} ({accuracy:.1f}%)")
        print(f"   Mismatch:   {len(self.df) - matches:,} ({100-accuracy:.1f}%)")
        print()
        
        # 4. Top confident predictions
        print("4️⃣  TOP 3 PREDIZIONI PIÙ SICURE")
        print("-" * 70)
        top_confident = self.df.nlargest(3, 'confidence')
        for idx, (_, row) in enumerate(top_confident.iterrows(), 1):
            print(f"{idx}. {row['predicted_sentiment']} ({row['confidence']:.2%})")
            print(f"   '{row['text'][:80]}...'")
            print()
        
        # 5. Esempi per sentiment
        print("5️⃣  ESEMPI PER SENTIMENT")
        print("-" * 70)
        for sentiment in self.df['predicted_sentiment'].unique():
            examples = self.df[self.df['predicted_sentiment'] == sentiment].head(2)
            print(f"\n   📌 {sentiment}:")
            for _, row in examples.iterrows():
                print(f"      • {row['text'][:100]}...")
                print(f"        Confidence: {row['confidence']:.2%}")
        
        print()
        print("=" * 70)
        print()
    
    def save_results(self, output_dir='results'):
        """
        Salva i risultati in CSV e Excel
        
        Args:
            output_dir (str): Directory dove salvare i risultati
        """
        print("=" * 70)
        print("💾 SALVATAGGIO RISULTATI")
        print("=" * 70)
        print()
        
        # Crea directory se non esiste
        os.makedirs(output_dir, exist_ok=True)
        
        # Timestamp per i file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Salva CSV
        csv_path = f"{output_dir}/sentiment_results.csv"
        print(f"💾 Salvataggio CSV: {csv_path}")
        self.df.to_csv(csv_path, index=False)
        print(f"   ✅ CSV salvato ({len(self.df):,} righe)")
        print()
        
        # 2. Salva Excel con sheet multipli
        excel_path = f"{output_dir}/sentiment_results.xlsx"
        print(f"📊 Salvataggio Excel: {excel_path}")
        
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # Sheet 1: Risultati completi
            self.df.to_excel(writer, sheet_name='Results', index=False)
            
            # Sheet 2: Statistiche
            stats_data = {
                'Metric': [
                    'Total News',
                    'Positive',
                    'Negative', 
                    'Neutral',
                    'Avg Confidence',
                    'Min Confidence',
                    'Max Confidence'
                ],
                'Value': [
                    len(self.df),
                    (self.df['predicted_sentiment'] == 'POSITIVE').sum(),
                    (self.df['predicted_sentiment'] == 'NEGATIVE').sum(),
                    (self.df['predicted_sentiment'] == 'NEUTRAL').sum() 
                        if 'NEUTRAL' in self.df['predicted_sentiment'].values else 0,
                    f"{self.df['confidence'].mean():.2%}",
                    f"{self.df['confidence'].min():.2%}",
                    f"{self.df['confidence'].max():.2%}"
                ]
            }
            stats_df = pd.DataFrame(stats_data)
            stats_df.to_excel(writer, sheet_name='Statistics', index=False)
            
            # Sheet 3: Distribution
            dist_df = self.df['predicted_sentiment'].value_counts().reset_index()
            dist_df.columns = ['Sentiment', 'Count']
            dist_df['Percentage'] = (dist_df['Count'] / len(self.df) * 100).round(1)
            dist_df.to_excel(writer, sheet_name='Distribution', index=False)
        
        print(f"   ✅ Excel salvato con 3 sheet:")
        print(f"      - Results (dati completi)")
        print(f"      - Statistics (metriche)")
        print(f"      - Distribution (distribuzione)")
        print()
        
        # 3. Summary report
        summary_path = f"{output_dir}/analysis_summary.txt"
        print(f"📄 Salvataggio summary: {summary_path}")
        
        with open(summary_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("FINANCIAL NEWS SENTIMENT ANALYSIS - SUMMARY REPORT\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Dataset: {self.data_path}\n")
            f.write(f"Total News Analyzed: {len(self.df):,}\n\n")
            
            f.write("SENTIMENT DISTRIBUTION:\n")
            f.write("-" * 70 + "\n")
            for sentiment, count in self.df['predicted_sentiment'].value_counts().items():
                percentage = (count / len(self.df)) * 100
                f.write(f"{sentiment:10} {count:6,} ({percentage:5.1f}%)\n")
            
            f.write(f"\nAVERAGE CONFIDENCE: {self.df['confidence'].mean():.2%}\n")
            
            # Confronto con originale
            matches = (self.df['original_sentiment'].str.lower() == 
                      self.df['predicted_sentiment'].str.lower()).sum()
            accuracy = (matches / len(self.df)) * 100
            f.write(f"ACCURACY vs ORIGINAL: {accuracy:.1f}%\n")
        
        print(f"   ✅ Summary salvato")
        print()
        
        print("=" * 70)
        print("✅ TUTTI I RISULTATI SALVATI!")
        print("=" * 70)
        print()
        print(f"📁 File generati in '{output_dir}/':")
        print(f"   • sentiment_results.csv")
        print(f"   • sentiment_results.xlsx")
        print(f"   • analysis_summary.txt")
        print()


def main():
    """
    Funzione principale
    """
    print("\n" + "=" * 70)
    print("🏦 FINANCIAL NEWS SENTIMENT ANALYSIS")
    print("=" * 70)
    print()
    print("📅 Data:", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print("👨‍💻 Progetto: Data Mining & Text Analytics - IULM University")
    print()
    
    # Inizializza analyzer
    analyzer = FinancialSentimentAnalyzer('data/all-data.csv')
    
    # Carica dati
    # Per test veloce usa: sample_size=500
    # Per analisi completa usa: sample_size=None
    print("⚙️  CONFIGURAZIONE:")
    print("   • Test veloce: sample_size=500")
    print("   • Analisi completa: sample_size=None (commenta la riga)")
    print()
    
    # 🔧 MODIFICA QUI per test o analisi completa
    analyzer.load_data()  # ← Rimuovi questo per analisi completa
    # analyzer.load_data()  # ← Decommentare per analisi completa
    
    # Analizza sentiment
    analyzer.analyze_sentiment()
    
    # Genera statistiche
    analyzer.generate_statistics()
    
    # Salva risultati
    analyzer.save_results()
    
    print("=" * 70)
    print("🎉 ANALISI COMPLETATA CON SUCCESSO!")
    print("=" * 70)
    print()
    print("📌 PROSSIMI STEP:")
    print("   1. Controlla i file in results/")
    print("   2. Se va bene, esegui analisi completa (rimuovi sample_size)")
    print("   3. Domani (Giorno 3): Genera visualizzazioni!")
    print()


if __name__ == "__main__":
    main()