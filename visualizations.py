"""
Visualizations Generator
Genera grafici per i risultati dell'analisi sentiment
Versione ottimizzata - solo visualizzazioni coerenti con transformer-based models
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import warnings

warnings.filterwarnings('ignore')

# Configurazione stile grafici
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

class SentimentVisualizer:
    """
    Classe per generare visualizzazioni dei risultati sentiment analysis
    """
    
    def __init__(self, results_path='results/sentiment_results.csv'):
        """
        Inizializza visualizer
        
        Args:
            results_path (str): Percorso file CSV con risultati
        """
        print("\n" + "=" * 70)
        print("🎨 SENTIMENT ANALYSIS VISUALIZER")
        print("=" * 70)
        print()
        
        # Verifica esistenza file
        if not os.path.exists(results_path):
            print(f"❌ ERRORE: File non trovato: {results_path}")
            print("   Esegui prima: python financial_sentiment_analyzer.py")
            exit(1)
        
        # Carica risultati
        print(f"📂 Caricamento risultati da: {results_path}")
        self.df = pd.read_csv(results_path)
        print(f"✅ Caricati {len(self.df):,} risultati")
        print()
        
        # Crea directory visualizations
        os.makedirs('visualizations', exist_ok=True)
        
        # Colori personalizzati
        self.colors = {
            'POSITIVE': '#2ecc71',  # Verde
            'NEGATIVE': '#e74c3c',  # Rosso
            'NEUTRAL': '#95a5a6'    # Grigio
        }
    
    def plot_sentiment_distribution(self):
        """
        Grafico distribuzione sentiment (pie + bar chart)
        """
        print("=" * 70)
        print("📊 GENERAZIONE: Sentiment Distribution")
        print("=" * 70)
        print()
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        sentiment_counts = self.df['predicted_sentiment'].value_counts()
        colors_list = [self.colors.get(s, '#95a5a6') for s in sentiment_counts.index]
        
        # 1. Bar Chart
        ax1 = axes[0]
        bars = ax1.bar(sentiment_counts.index, sentiment_counts.values, 
                       color=colors_list, edgecolor='black', linewidth=1.5)
        ax1.set_title('Sentiment Distribution - Counts', fontweight='bold', fontsize=14)
        ax1.set_xlabel('Sentiment', fontweight='bold')
        ax1.set_ylabel('Number of News', fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        
        # Aggiungi valori sopra le barre
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height):,}',
                    ha='center', va='bottom', fontweight='bold')
        
        # 2. Pie Chart
        ax2 = axes[1]
        wedges, texts, autotexts = ax2.pie(
            sentiment_counts.values, 
            labels=sentiment_counts.index,
            autopct='%1.1f%%',
            colors=colors_list,
            startangle=90,
            textprops={'fontsize': 12, 'fontweight': 'bold'}
        )
        
        # Migliora leggibilità percentuali
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(13)
            autotext.set_fontweight('bold')
        
        ax2.set_title('Sentiment Distribution - Percentage', fontweight='bold', fontsize=14)
        
        plt.suptitle('📊 Financial News Sentiment Analysis - Distribution', 
                     fontsize=16, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        
        # Salva
        filepath = 'visualizations/sentiment_distribution.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"✅ Salvato: {filepath}")
        # plt.show()  # Commentato per non bloccare
        plt.close()
        print()
    
    def plot_confidence_distribution(self):
        """
        Grafico distribuzione confidence scores
        """
        print("=" * 70)
        print("📈 GENERAZIONE: Confidence Distribution")
        print("=" * 70)
        print()
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # 1. Histogram
        ax1 = axes[0]
        ax1.hist(self.df['confidence'], bins=30, color='#3498db', 
                edgecolor='black', alpha=0.7)
        ax1.axvline(self.df['confidence'].mean(), color='red', 
                   linestyle='--', linewidth=2,
                   label=f"Mean: {self.df['confidence'].mean():.2%}")
        ax1.axvline(self.df['confidence'].median(), color='green', 
                   linestyle='--', linewidth=2,
                   label=f"Median: {self.df['confidence'].median():.2%}")
        ax1.set_title('Confidence Score Distribution', fontweight='bold', fontsize=14)
        ax1.set_xlabel('Confidence Score', fontweight='bold')
        ax1.set_ylabel('Frequency', fontweight='bold')
        ax1.legend(loc='upper left', fontsize=11)
        ax1.grid(axis='y', alpha=0.3)
        
        # 2. Boxplot per sentiment
        ax2 = axes[1]
        sentiments = self.df['predicted_sentiment'].unique()
        colors_box = [self.colors.get(s, '#95a5a6') for s in sentiments]
        
        box_data = [self.df[self.df['predicted_sentiment']==s]['confidence'].values 
                    for s in sentiments]
        
        bp = ax2.boxplot(box_data, labels=sentiments, patch_artist=True,
                        showmeans=True, meanline=True)
        
        # Colora boxplot
        for patch, color in zip(bp['boxes'], colors_box):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax2.set_title('Confidence by Sentiment', fontweight='bold', fontsize=14)
        ax2.set_xlabel('Sentiment', fontweight='bold')
        ax2.set_ylabel('Confidence Score', fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        
        plt.suptitle('📈 Confidence Analysis', 
                     fontsize=16, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        
        # Salva
        filepath = 'visualizations/confidence_analysis.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"✅ Salvato: {filepath}")
        # plt.show()  # Commentato
        plt.close()
        print()
    
    def create_summary_dashboard(self):
        """
        Dashboard riassuntivo completo
        """
        print("=" * 70)
        print("🎨 GENERAZIONE: Summary Dashboard")
        print("=" * 70)
        print()
        
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Sentiment Distribution (Bar) - GRANDE
        ax1 = fig.add_subplot(gs[0, :2])
        sentiment_counts = self.df['predicted_sentiment'].value_counts()
        colors_list = [self.colors.get(s, '#95a5a6') for s in sentiment_counts.index]
        bars = ax1.bar(sentiment_counts.index, sentiment_counts.values, 
                      color=colors_list, edgecolor='black', linewidth=2)
        ax1.set_title('Sentiment Distribution', fontsize=13, fontweight='bold')
        ax1.set_ylabel('Count', fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height):,}',
                    ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        # 2. Key Metrics
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.axis('off')
        
        total = len(self.df)
        pos_count = (self.df['predicted_sentiment'] == 'POSITIVE').sum()
        neg_count = (self.df['predicted_sentiment'] == 'NEGATIVE').sum()
        avg_conf = self.df['confidence'].mean()
        
        metrics_text = f"""
        📊 KEY METRICS
        {'─'*25}
        
        Total News: {total:,}
        
        🟢 Positive: {pos_count:,}
           ({pos_count/total*100:.1f}%)
        
        🔴 Negative: {neg_count:,}
           ({neg_count/total*100:.1f}%)
        
        📈 Avg Confidence:
           {avg_conf:.1%}
        
        📉 Min: {self.df['confidence'].min():.1%}
        📈 Max: {self.df['confidence'].max():.1%}
        """
        
        ax2.text(0.05, 0.5, metrics_text, fontsize=10, 
                family='monospace', verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        # 3. Confidence Distribution Histogram
        ax3 = fig.add_subplot(gs[1, :2])
        ax3.hist(self.df['confidence'], bins=40, color='#3498db', 
                edgecolor='black', alpha=0.7)
        ax3.axvline(self.df['confidence'].mean(), color='red', 
                   linestyle='--', linewidth=2,
                   label=f"Mean: {self.df['confidence'].mean():.2%}")
        ax3.set_title('Confidence Score Distribution', fontsize=13, fontweight='bold')
        ax3.set_xlabel('Confidence Score', fontweight='bold')
        ax3.set_ylabel('Frequency', fontweight='bold')
        ax3.legend()
        ax3.grid(axis='y', alpha=0.3)
        
        # 4. Confidence by Sentiment (Violin Plot)
        ax4 = fig.add_subplot(gs[1, 2])
        sentiments = self.df['predicted_sentiment'].unique()
        colors_box = [self.colors.get(s, '#95a5a6') for s in sentiments]
        
        parts = ax4.violinplot(
            [self.df[self.df['predicted_sentiment']==s]['confidence'].values for s in sentiments],
            positions=range(len(sentiments)),
            showmeans=True,
            showmedians=True
        )
        
        for pc, color in zip(parts['bodies'], colors_box):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
        
        ax4.set_xticks(range(len(sentiments)))
        ax4.set_xticklabels(sentiments)
        ax4.set_title('Confidence by Sentiment', fontsize=13, fontweight='bold')
        ax4.set_ylabel('Confidence Score', fontweight='bold')
        ax4.grid(axis='y', alpha=0.3)
        
        # 5. Top Confident Predictions
        ax5 = fig.add_subplot(gs[2, :])
        ax5.axis('off')
        
        top_conf = self.df.nlargest(5, 'confidence')
        
        examples_text = "🏆 TOP 5 MOST CONFIDENT PREDICTIONS\n"
        examples_text += "─" * 80 + "\n\n"
        
        for idx, (_, row) in enumerate(top_conf.iterrows(), 1):
            emoji = "🟢" if row['predicted_sentiment'] == 'POSITIVE' else "🔴"
            text_preview = row['text'][:90] + "..." if len(row['text']) > 90 else row['text']
            examples_text += f"{idx}. {emoji} {row['predicted_sentiment']} ({row['confidence']:.2%})\n"
            examples_text += f"   \"{text_preview}\"\n\n"
        
        ax5.text(0.05, 0.95, examples_text, fontsize=9, 
                family='monospace', verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        
        # Titolo principale
        plt.suptitle('📊 FINANCIAL NEWS SENTIMENT ANALYSIS - DASHBOARD', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        # Salva
        filepath = 'visualizations/dashboard.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"✅ Salvato: {filepath}")
        # plt.show()  # Commentato
        plt.close()
        print()
    
    def generate_all_visualizations(self):
        """
        Genera tutte le visualizzazioni in sequenza
        """
        print("\n" + "=" * 70)
        print("🎨 GENERAZIONE TUTTE LE VISUALIZZAZIONI")
        print("=" * 70)
        print()
        print("ℹ️  Note: Word frequency charts excluded")
        print("   Reason: Inconsistent with transformer-based approach")
        print("   DistilBERT uses contextual analysis, not keyword frequency")
        print()
        
        # 1. Sentiment Distribution
        self.plot_sentiment_distribution()
        
        # 2. Confidence Analysis
        self.plot_confidence_distribution()
        
        # 3. Dashboard
        self.create_summary_dashboard()
        
        print("=" * 70)
        print("🎉 TUTTE LE VISUALIZZAZIONI GENERATE!")
        print("=" * 70)
        print()
        print("📁 File salvati in 'visualizations/':")
        print("   • sentiment_distribution.png")
        print("   • confidence_analysis.png")
        print("   • dashboard.png")
        print()
        print("📌 DESIGN CHOICE:")
        print("   Keyword frequency charts deliberately excluded")
        print("   to maintain consistency with transformer architecture.")
        print("   DistilBERT analyzes context, not individual word frequency.")
        print()
        print("📌 PROSSIMI STEP:")
        print("   ✅ Giorno 3 completato!")
        print("   📅 Domani (Giorno 4): Documentazione (README, LICENSE, etc.)")
        print()


def main():
    """
    Funzione principale
    """
    # Inizializza visualizer
    visualizer = SentimentVisualizer('results/sentiment_results.csv')
    
    # Genera tutte le visualizzazioni
    visualizer.generate_all_visualizations()


if __name__ == "__main__":
    main()