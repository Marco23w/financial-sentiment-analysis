"""
Test Installazione Librerie
Verifica che tutte le librerie necessarie siano installate correttamente
"""

print("=" * 60)
print("🔍 VERIFICA INSTALLAZIONE LIBRERIE")
print("=" * 60)
print()

# Lista librerie da verificare
libraries = {
    'transformers': 'Hugging Face Transformers',
    'torch': 'PyTorch',
    'pandas': 'Pandas',
    'matplotlib': 'Matplotlib',
    'seaborn': 'Seaborn',
    'wordcloud': 'WordCloud',
    'openpyxl': 'OpenPyXL',
    'numpy': 'NumPy'
}

# Contatori
installed = 0
missing = []

# Verifica ogni libreria
for module, name in libraries.items():
    try:
        lib = __import__(module)
        version = getattr(lib, '__version__', 'N/A')
        print(f"✅ {name:30} v{version}")
        installed += 1
    except ImportError:
        print(f"❌ {name:30} NON INSTALLATO")
        missing.append(module)

print()
print("=" * 60)

# Risultato finale
if missing:
    print(f"⚠️  {len(missing)} librerie mancanti: {', '.join(missing)}")
    print()
    print("📦 Installa con:")
    print(f"   pip install {' '.join(missing)}")
else:
    print(f"🎉 TUTTE LE {installed} LIBRERIE SONO INSTALLATE!")
    print()
    print("✅ Sei pronto per iniziare il progetto!")

print("=" * 60)
