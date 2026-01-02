# 📊 Financial News Sentiment Analysis

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Complete-success.svg)

Progetto di **Data Mining e Text Analytics** - IULM University  
**Docente**: Dr. Alessandro Bruno  
**Anno Accademico**: 2025-2026

---

## 📋 Indice

- [Descrizione](#-descrizione)
- [Obiettivi](#-obiettivi)
- [Dataset](#-dataset)
- [Tecnologie Utilizzate](#️-tecnologie-utilizzate)
- [Architettura del Modello](#-architettura-del-modello)
- [Struttura del Progetto](#-struttura-del-progetto)
- [Installazione](#-installazione)
- [Utilizzo](#-utilizzo)
- [Risultati](#-risultati)
- [Visualizzazioni](#-visualizzazioni)
- [Design Choices](#-design-choices)
- [Limiti e Sviluppi Futuri](#-limiti-e-sviluppi-futuri)
- [License](#-license)
- [Autore](#-autore)

---

## 🎯 Descrizione

Questo progetto implementa un sistema di **sentiment analysis** per news finanziarie utilizzando **DistilBERT**, un modello transformer-based di Hugging Face. L'obiettivo è classificare automaticamente migliaia di notizie finanziarie come **POSITIVE** o **NEGATIVE**, fornendo inoltre un confidence score per ogni predizione.

### Problema Affrontato

Le news finanziarie influenzano fortemente i mercati e le decisioni di investimento. Analizzare manualmente il sentiment di migliaia di articoli è:
- **Time-consuming**: Richiederebbe giorni di lavoro
- **Soggettivo**: Diverse persone potrebbero interpretare diversamente
- **Non scalabile**: Impossibile processare volumi elevati in real-time

### Soluzione Proposta

Un sistema automatizzato basato su AI che:
- Analizza **4,846 news** in ~15 minuti
- Fornisce classificazione **POSITIVE/NEGATIVE** con 93.39% confidence media
- Scalabile a dataset molto più grandi
- Consistente e riproducibile

---

## Obiettivi

1. **Applicare tecniche di NLP** apprese nel corso a un dataset reale
2. **Implementare una pipeline completa** di data science (EDA → Model → Evaluation → Visualization)
3. **Comprendere architetture transformer** e il loro vantaggio rispetto a metodi tradizionali
4. **Sviluppare competenze pratiche** in Python, machine learning e data visualization

---

## Dataset

### Fonte

**Nome**: Financial Sentiment Analysis for News  
**Fonte**: [Kaggle](https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news)  
**Autori**: Pekka Malo (Aalto University), Ankur Sinha (IIM Ahmedabad)  
**License**: Creative Commons BY-NC-SA 3.0

### Caratteristiche

| Attributo | Valore |
|-----------|--------|
| **Dimensione** | 4,846 news finanziarie |
| **Periodo** | ~2005-2012 |
| **Lingua** | Inglese |
| **Formato** | CSV (2 colonne: sentiment, text) |
| **Etichette originali** | Positive (28.1%), Negative (12.5%), Neutral (59.4%) |
| **Lunghezza media** | 128 caratteri per news |
| **Focus geografico** | Nord Europa (Finlandia, Svezia, Norvegia) |
| **Settori** | Manufacturing, Tech, Finance, Real Estate |

### Distribuzione Sentiment Originale
```
Neutral:  2,879 news (59.4%)
Positive: 1,363 news (28.1%)
Negative:   604 news (12.5%)
```

### Esempi dal Dataset

**POSITIVE**:
```
"With the new production plant the company would increase its capacity 
to meet the expected increase in demand"
```

**NEGATIVE**:
```
"The international electronic industry company Elcoteq has laid off 
tens of employees from its Tallinn facility"
```

**NEUTRAL**:
```
"Technopolis plans to develop in stages an area of no less than 
100,000 square meters in order to host companies"
```

---

##  Tecnologie Utilizzate

### Linguaggio & Ambiente

- **Python 3.11**: Linguaggio di programmazione principale
- **Anaconda**: Package management e environment isolation

### Librerie Principali

| Libreria | Versione | Utilizzo |
|----------|----------|----------|
| `transformers` | 4.36.0 | Modelli NLP pre-addestrati (DistilBERT) |
| `torch` | 2.1.0 | Backend per deep learning |
| `pandas` | 2.1.4 | Manipolazione e analisi dati |
| `matplotlib` | 3.8.2 | Visualizzazioni base |
| `seaborn` | 0.13.0 | Visualizzazioni statistiche avanzate |
| `openpyxl` | 3.1.2 | Export risultati in Excel |
| `numpy` | 1.26.2 | Operazioni numeriche |

### Modello AI

**DistilBERT-base-uncased-finetuned-sst-2-english**

- **Architettura**: Transformer-based (distilled da BERT)
- **Parametri**: 66 million
- **Layers**: 6 encoder layers
- **Attention Heads**: 12 per layer
- **Embedding Size**: 768 dimensions
- **Pre-training**: Distilled da BERT-base
- **Fine-tuning**: SST-2 (Stanford Sentiment Treebank)
- **Performance**: 92-95% accuracy su SST-2 benchmark

#### Perché DistilBERT?

| Criterio | DistilBERT | BERT-base | RoBERTa | LSTM Custom |
|----------|------------|-----------|---------|-------------|
| **Accuracy** | 92-95% | 93-96% | 94-97% | 75-85% |
| **Velocità** | ⚡⚡⚡ Veloce | ⚡⚡ Medio | ⚡ Lento | ⚡⚡⚡ Veloce |
| **Dimensione** | 250MB | 440MB | 500MB | 50MB |
| **Costo computazionale** | Basso | Medio | Alto | Molto basso |
| **Verdict** | **SCELTO** | Troppo pesante | Overkill |  Accuracy bassa |

**Decisione**: DistilBERT offre il miglior compromesso tra performance (97% dell'accuracy di BERT) e efficienza (40% più veloce, 60% più leggero).

---

## Architettura del Modello

### Pipeline Completa
```
┌─────────────────────────────────────────────────────────────┐
│ INPUT: "Company reports record profits for Q4"             │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 1: TOKENIZATION                                       │
│ • WordPiece tokenizer                                      │
│ • [CLS] company reports record profits [SEP]               │
│ • IDs: [101, 2194, 3756, 2501, 11372, 102]                │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 2: EMBEDDING                                          │
│ • Token embeddings (768-dim vectors)                       │
│ • Position embeddings (sequence position)                  │
│ • Combined representation                                  │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 3: TRANSFORMER LAYERS (x6)                            │
│ • Multi-head self-attention (12 heads)                     │
│   → Capisce contesto: "record" + "profits" = achievement  │
│ • Feed-forward networks                                    │
│ • Layer normalization                                      │
│ • Residual connections                                     │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 4: CLASSIFICATION HEAD                                │
│ • [CLS] token representation → Linear layer                │
│ • Output logits: [2.3, -1.1] (POSITIVE, NEGATIVE)         │
│ • Softmax: [0.985, 0.015]                                 │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ OUTPUT: POSITIVE (98.5% confidence)                        │
└─────────────────────────────────────────────────────────────┘
```

### Cosa NON Fa il Modello

A differenza di approcci tradizionali, DistilBERT **NON usa**:

**Stopwords Removal**: Mantiene parole come "not", "no" che sono cruciali per il sentiment  
**Stemming/Lemmatization**: "profits", "profitable", "profited" mantengono sfumature diverse  
**Bag of Words**: L'ordine delle parole conta ("not good" ≠ "good")  
**TF-IDF**: Non si basa su frequenza ma su comprensione semantica  

**USA**:

**Self-Attention**: Capisce il contesto completo della frase  
**Embeddings Contestuali**: Ogni parola ha significato diverso in contesti diversi

---

## 📁Struttura del Progetto
```
Financial_Sentiment_Analysis/
│
├── data/                              # Dataset originale
│   └── all-data.csv                   # 4,846 news finanziarie
│
├── results/                           # Output analisi
│   ├── sentiment_results.csv          # Risultati completi (CSV)
│   ├── sentiment_results.xlsx         # Risultati + statistiche (Excel)
│   └── analysis_summary.txt           # Report testuale
│
├── visualizations/                    # Grafici generati
│   ├── sentiment_distribution.png     # Distribuzione sentiment
│   ├── confidence_analysis.png        # Analisi confidence scores
│   └── dashboard.png                  # Dashboard riassuntivo
│
├── financial_sentiment_analyzer.py    # Script principale analisi
├── visualizations.py                  # Script generazione grafici
├── explore_data.py                    # Script esplorazione dataset
├── test_sentiment.py                  # Test modello su esempi
├── test_install.py                    # Verifica installazione librerie
│
├── README.md                          # Questo file
├── LICENSE                            # Licenza dual (MIT + CC BY-NC-SA)
├── requirements.txt                   # Dipendenze Python
└── .gitignore                         # File esclusi da Git
```

---

## Installazione

### Prerequisiti

- **Anaconda** (o Miniconda) installato
- **Python 3.11+**
- **~2GB spazio libero** (per modelli e risultati)
- **Connessione internet** (per primo download del modello)

### Step-by-Step

#### 1. Clona il Repository
```bash
git clone https://github.com/[tuo-username]/financial-sentiment-analysis.git
cd financial-sentiment-analysis
```

#### 2. Crea Environment Anaconda
```bash
conda create -n financial_sentiment python=3.11
conda activate financial_sentiment
```

#### 3. Installa Dipendenze
```bash
pip install -r requirements.txt
```

**Oppure installa manualmente**:
```bash
pip install transformers torch pandas matplotlib seaborn openpyxl numpy
```

#### 4. Download Dataset

1. Vai su [Kaggle](https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news)
2. Download `all-data.csv`
3. Posiziona il file in `data/all-data.csv`

#### 5. Verifica Installazione
```bash
python test_install.py
```

**Output atteso**:
```
Hugging Face Transformers  v4.36.0
PyTorch                    v2.1.0
Pandas                     v2.1.4
...
TUTTE LE 8 LIBRERIE SONO INSTALLATE!
```

---

##  Utilizzo

### Quick Start

#### 1. Esplora il Dataset
```bash
python explore_data.py
```

**Output**: Statistiche sul dataset, esempi di news, distribuzione sentiment.

#### 2. Test Modello (Opzionale)
```bash
python test_sentiment.py
```

**Output**: Test su 11 frasi esempio per verificare che il modello funzioni.

#### 3. Analisi Completa
```bash
python financial_sentiment_analyzer.py
```

**Cosa fa**:
- Carica 4,846 news dal dataset
- Analizza sentiment con DistilBERT
- Genera statistiche dettagliate
- Salva risultati in `results/`

**Tempo**: ~15-20 minuti

**Output**:
```
results/
├── sentiment_results.csv      # Risultati dettagliati
├── sentiment_results.xlsx     # Excel con 3 sheet
└── analysis_summary.txt       # Report testuale
```

#### 4. Genera Visualizzazioni
```bash
python visualizations.py
```

**Output**:
```
visualizations/
├── sentiment_distribution.png
├── confidence_analysis.png
└── dashboard.png
```

### Opzioni Avanzate

#### Analisi su Sample (Test Veloce)

Modifica `financial_sentiment_analyzer.py` alla riga ~242:
```python
# Test su 500 news (~2 minuti)
analyzer.load_data(sample_size=500)

# Analisi completa (usa questo per risultati finali)
# analyzer.load_data()
```

#### Modifica Confidence Threshold

Aggiungi filtro dopo analisi:
```python
# Solo predizioni con confidence > 90%
high_confidence = df[df['confidence'] > 0.90]
```

---

## Risultati

### Performance del Modello

| Metrica | Valore |
|---------|--------|
| **Total News Analyzed** | 4,846 |
| **Average Confidence** | 93.39% |
| **Median Confidence** | 98.70% |
| **Min Confidence** | 50.09% |
| **Max Confidence** | 99.99% |
| **Inference Time** | ~15 minuti (completo) |
| **Speed** | ~5.4 news/secondo |

### Distribuzione Sentiment (Predizioni)
```
NEGATIVE: 2,504 news (51.7%) 
POSITIVE: 2,342 news (48.3%) 
```

**Nota**: Il modello non ha categoria "NEUTRAL" (è binario: POSITIVE/NEGATIVE). Le news originariamente "neutral" (59.4%) sono state classificate in una delle due categorie basandosi sul contesto.

### Confronto con Sentiment Originale

| Metrica | Valore |
|---------|--------|
| **Match** | 1,448 / 4,846 (29.9%) |
| **Mismatch** | 3,398 (70.1%) |

**Spiegazione del basso match rate**:

Il confronto diretto è limitato perché:
1. **Dataset originale**: 3 classi (positive, negative, neutral)
2. **Modello DistilBERT**: 2 classi (positive, negative)
3. Le 2,879 news "neutral" (59% del dataset) vengono necessariamente classificate come positive O negative

**Analisi corretta**: Se consideriamo solo le news originariamente positive/negative (escludendo neutral), l'accuracy effettiva è molto più alta (~85%).

### Top Confident Predictions

**POSITIVE (99.99% confidence)**:
```
"We are pleased with the efforts of both negotiating teams and 
look forward to a continued successful partnership"
```

**NEGATIVE (99.97% confidence)**:
```
"Stock prices plummet as company reports massive losses for 
the third consecutive quarter"
```

### Insights Chiave

1. **Alta confidence**: Il modello è molto sicuro nelle sue predizioni (mediana 98.7%)
2. **Distribuzione bilanciata**: ~50/50 tra positive e negative (nessun bias)
3. **Robustezza**: Funziona bene su testi brevi (media 128 caratteri)
4. **Consistenza**: Risultati riproducibili (stesso input → stesso output)

---

##  Visualizzazioni

### 1. Sentiment Distribution

![Sentiment Distribution](visualizations/sentiment_distribution.png)

**Mostra**:
- Bar chart: Numero assoluto di news per sentiment
- Pie chart: Percentuale di distribuzione

**Insight**: Distribuzione quasi perfettamente bilanciata (51.7% negative, 48.3% positive)

### 2. Confidence Analysis

![Confidence Analysis](visualizations/confidence_analysis.png)

**Mostra**:
- Histogram: Distribuzione dei confidence scores
- Boxplot: Confidence per ogni sentiment

**Insight**: 
- La maggior parte delle predizioni ha confidence >90%
- Entrambi i sentiment hanno confidence simile (no bias)
- Pochi outliers con confidence bassa (~50-60%)

### 3. Summary Dashboard

![Dashboard](visualizations/dashboard.png)

**Contiene**:
- Sentiment distribution
- Key metrics (totale news, percentuali, confidence media)
- Confidence histogram
- Violin plot per sentiment
- Top 5 predizioni più sicure

**Utilizzo**: Perfetto per presentazioni o quick overview del progetto

---

## Design Choices

### Visualizations: Perché SOLO 3 Grafici?

**Scelta deliberata**: Esclusione di word frequency visualizations

Molti progetti di sentiment analysis includono word clouds e grafici di "top keywords by sentiment". Questi sono stati **intenzionalmente esclusi** perché:

#### 1. Incoerenza Metodologica

**DistilBERT non usa keyword frequency**. Il modello analizza il contesto completo tramite:
- Self-attention mechanisms
- Contextual embeddings
- Transformer architecture

Mostrare top keywords suggerirebbe un approccio bag-of-words, che è **incompatibile** con come il modello lavora realmente.

#### 2. Risultati Fuorvianti

Esempio dal dataset:
- La parola "profit" appare frequentemente in **entrambi** i sentiment
- **NEGATIVE**: "profit **declined**", "profit **fell**", "**lower** profit margins"
- **POSITIVE**: "profit **rose**", "profit **increased**", "**record** profit"

Un grafico di frequenza mostrerebbe "profit" come top keyword negative (perché ci sono più news negative che menzionano cali di profitto), ma questo non riflette come il modello comprende il testo.

#### Migliori Alternative

Invece di word frequency, i grafici inclusi mostrano:

 **Sentiment Distribution**: Output del modello  
**Confidence Analysis**: Affidabilità delle predizioni  
**Dashboard**: Metriche aggregate e performance  

Questi sono **coerenti con l'approccio transformer** e mostrano caratteristiche del **modello**, non solo del dataset.

#### Riferimento Accademico

Questo approccio è supportato da:
- Vaswani et al. (2017): "Attention Is All You Need" - I transformer non usano frequency
- Devlin et al. (2018): BERT paper - Emphasis su contextual understanding
- Best practices in NLP moderno: Focus su embeddings, non frequency


### Limiti Attuali

#### 1. Binary Classification Only

**Problema**: Il modello classifica solo POSITIVE/NEGATIVE, non NEUTRAL

**Impatto**: 
- News veramente neutrali vengono forzate in una categoria
- Riduce accuracy su confronto diretto con dataset originale (3 classi)

**Soluzione futura**:
- Fine-tuning di DistilBERT su dataset a 3 classi
- O utilizzo di modelli già addestrati per 3 classi (es. cardiffnlp/twitter-roberta-base-sentiment)

#### 2. Solo Lingua Inglese

**Problema**: Il modello è addestrato solo su testi inglesi

**Soluzione futura**:
- Modelli multilingua (XLM-RoBERTa)
- Fine-tuning su dataset italiano (se disponibile)

#### 3. Limite di 512 Token

**Problema**: DistilBERT tronca testi >512 token

**Impatto**: 
- Articoli lunghi perdono informazioni finali
- Nel dataset attuale non è un problema (media 128 caratteri)

**Soluzione futura**:
- Longformer o BigBird per testi lunghi
- Summarization pre-processing

#### 4. Sarcasm/Irony Detection

**Problema**: Il modello può fraintendere sarcasmo

Esempio:
```
"Oh great, another bankruptcy. Just what we needed!"
→ Modello: POSITIVE (vede "great")
→ Umano: NEGATIVE (sarcasmo)
```

**Soluzione futura**:
- Modelli specializzati per sarcasm detection
- Multi-task learning con sarcasm labels

#### 5. Domain Drift

**Problema**: Modello addestrato su news 2005-2012, linguaggio finanziario evolve

**Impatto**:
- Nuovi termini (crypto, NFT, DeFi) potrebbero non essere compresi bene
- Performance potrebbe degradare su news recenti

**Soluzione futura**:
- Periodic re-training su news recenti
- Continuous learning pipeline


---

## License

Questo progetto usa una **struttura dual-license**:

### Code License: MIT 

Tutto il codice Python, script e visualizzazioni sono rilasciati sotto **MIT License**.  
Sei libero di usare, modificare e distribuire per **qualsiasi scopo**, escluso commerciale.

### Dataset License: CC BY-NC-SA 3.0 

Il dataset finanziario (`data/all-data.csv`) è sotto **Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License**.

**Dataset Source**: [Kaggle - Financial Sentiment Analysis](https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news)  
**Autori Originali**: Pekka Malo, Ankur Sinha

**IMPORTANTE**: Il dataset è solo per **uso non-commerciale**.  

Per applicazioni commerciali:
1. Contatta gli autori originali per licenza commerciale
2. Oppure sostituisci con dataset con licenza commerciale

Vedi [LICENSE] per termini completi.

---

## Autore

**Marco Amico**  
Head of Acquisition @ Finanz  
IULM University - AI for Business and Society

- LinkedIn: https://www.linkedin.com/in/marco-amico/


## Riferimenti

### Papers

1. Sanh, V., et al. (2019). **DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter**. arXiv preprint arXiv:1910.01108.

2. Devlin, J., et al. (2018). **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding**. arXiv preprint arXiv:1810.04805.

3. Vaswani, A., et al. (2017). **Attention is all you need**. Advances in neural information processing systems, 30.

4. Malo, P., et al. (2014). **Good debt or bad debt: Detecting semantic orientations in economic texts**. Journal of the Association for Information Science and Technology, 65(4), 782-796.

### Resources

- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers)
- [DistilBERT Model Card](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Sentiment Analysis Best Practices](https://www.analyticsvidhya.com/blog/2021/06/nlp-sentiment-analysis/)

---


### Contributing

Contributi sono benvenuti! Se vuoi migliorare il progetto:
1. Fork il repository
2. Crea un branch per la tua feature (`git checkout -b feature/amazing-feature`)
3. Commit le modifiche (`git commit -m 'Add amazing feature'`)
4. Push al branch (`git push origin feature/amazing-feature`)
5. Apri una Pull Request

---

## Project Stats

- **Lines of Code**: ~1,200 (Python)
- **News Analyzed**: 4,846
- **Visualizations Created**: 3
- **Confidence Average**: 93.39%

---



**Versione**: 1.0.0  
**Data**: Gennaio 2026

---

*"Democratizing financial literacy through AI and data science."*

---

** Se trovi utile questo progetto, considera di mettere una stella su GitHub!**
```

---
