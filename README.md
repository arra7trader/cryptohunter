# CryptoHunter Bot 🔍

Bot pencari koin micin/potensial dari DEX (Decentralized Exchange) dengan analisis AI.

## Fitur

- **DexScreener API** - Mencari pair trending/baru dari berbagai chain
- **SNA Analysis (Drone Emprit Style)** - Deteksi hype berdasarkan volume spike
- **AI Prediction** - LSTM + Time-GPT untuk prediksi waktu pump

## Instalasi

```bash
# Install dependencies
pip install -r requirements.txt

# Jalankan bot
python main_bot.py
```

## Struktur Proyek

```
CryptoHunter/
├── main_bot.py              # Main orchestrator
├── requirements.txt         # Dependencies
├── README.md
└── modules/
    ├── __init__.py
    ├── dex_api.py           # DexScreener API client
    ├── sna_analyzer.py      # Social Network Analysis
    └── price_predictor.py   # LSTM + Time-GPT predictor
```

## Cara Kerja

1. **Scan DEX** - Mencari pair baru dengan likuiditas minimum
2. **SNA Filter** - Filter berdasarkan volume spike >500%
3. **AI Prediction** - Prediksi waktu pump dengan LSTM
4. **Report** - Output nama koin, skor SNA, akurasi, dan prediksi waktu

## Disclaimer

⚠️ Ini bukan financial advice. DYOR (Do Your Own Research)!
