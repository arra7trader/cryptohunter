# 🚀 CryptoHunter AI - Advanced Crypto Scanner & Prediction Engine

![Version](https://img.shields.io/badge/version-3.0.0-blue)
![Python](https://img.shields.io/badge/python-3.9+-green)
![Next.js](https://img.shields.io/badge/Next.js-16-black)
![License](https://img.shields.io/badge/license-MIT-purple)

<p align="center">
  <img src="https://img.shields.io/badge/AI-Powered-cyan?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Real--Time-Scanning-green?style=for-the-badge" />
  <img src="https://img.shields.io/badge/DexScreener-API-orange?style=for-the-badge" />
</p>

## 🎯 Overview

**CryptoHunter AI** adalah platform canggih untuk memantau dan memprediksi token cryptocurrency secara real-time. Menggunakan kombinasi **Machine Learning (LSTM, Transformer)** dan **Social Network Analysis (SNA)** untuk mendeteksi potensi pump pada token-token baru di DEX (Decentralized Exchange).

### ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🔍 **Live Scanner** | Pemantauan real-time token dari DexScreener API |
| 🤖 **AI Prediction** | Super-Ensemble model (LSTM + Transformer + Conv1D) |
| 📊 **SNA Analysis** | Drone Emprit-style volume spike detection |
| ⭐ **Watchlist** | Track dan monitor token favorit |
| 📈 **Trending** | Hot tokens, top gainers, dan AI picks |
| 🔔 **Alerts** | Price alert system |
| 🌐 **Multi-Chain** | Support Solana, Ethereum, BSC, Polygon, dll |

---

## 🏗️ Architecture

```
CryptoHunter/
├── 🐍 Backend (Python FastAPI)
│   ├── main_api.py          # REST API Server
│   ├── main_bot.py          # CLI Bot Scanner
│   ├── live_dashboard.py    # Terminal Dashboard
│   └── modules/
│       ├── dex_api.py       # DexScreener Client
│       ├── sna_analyzer.py  # SNA Analysis Engine
│       ├── price_predictor.py # AI/ML Models
│       └── db.py            # Database Manager
│
├── 🎨 Frontend (Next.js 16)
│   └── web_interface/
│       ├── app/
│       │   ├── page.js      # Main Dashboard
│       │   ├── layout.js    # App Layout
│       │   └── globals.css  # Styling
│       └── components/
│           ├── TokenTable.js
│           ├── DashboardHeader.js
│           ├── AIAnalysisPanel.js
│           ├── TrendingPanel.js
│           ├── WatchlistPanel.js
│           └── SearchBar.js
│
└── 🐳 Docker
    ├── Dockerfile
    └── docker-compose.yml
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Node.js 18+
- npm/yarn

### 1️⃣ Backend Setup

```bash
# Clone repository
git clone https://github.com/yourusername/CryptoHunter.git
cd CryptoHunter

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run API Server
python main_api.py
# Server runs at http://localhost:8000
```

### 2️⃣ Frontend Setup

```bash
# Navigate to frontend
cd web_interface

# Install dependencies
npm install

# Create .env.local
echo "NEXT_PUBLIC_API_URL=http://localhost:8000" > .env.local

# Run development server
npm run dev
# Frontend runs at http://localhost:3000
```

### 3️⃣ Docker (Optional)

```bash
docker-compose up -d
```

---

## 🎮 Usage

### Web Dashboard

1. Buka `http://localhost:3000` di browser
2. Dashboard akan otomatis memuat token dari DexScreener
3. Klik token untuk melihat detail dan menjalankan AI Analysis
4. Tambahkan token ke Watchlist dengan klik ⭐

### CLI Bot

```bash
python main_bot.py
```

### Live Terminal Dashboard

```bash
python live_dashboard.py
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API Status |
| GET | `/api/tokens` | Get all tokens |
| GET | `/api/market/summary` | Market statistics |
| GET | `/api/trending` | Trending tokens |
| GET | `/api/tokens/{address}/predict` | AI Prediction |
| GET | `/api/tokens/{address}/history` | Price history |
| GET | `/api/watchlist` | Get watchlist |
| POST | `/api/watchlist` | Add to watchlist |
| DELETE | `/api/watchlist/{address}` | Remove from watchlist |
| GET | `/api/alerts` | Get price alerts |
| POST | `/api/alerts` | Create alert |
| GET | `/api/search?q=` | Search tokens |
| GET | `/api/health` | Health check |
| WS | `/ws` | WebSocket real-time updates |

---

## 🤖 AI Models

### Super-Ensemble Predictor

Model kami menggunakan pendekatan ensemble yang menggabungkan:

1. **Transformer (Time-GPT Lite)**
   - Multi-Head Attention untuk pattern temporal
   - Layer Normalization
   
2. **Bidirectional LSTM**
   - Capture long-term dependencies
   - Return sequences untuk better context

3. **Conv1D**
   - Local pattern recognition
   - Price action signals

### SNA Analyzer V2

- Volume Spike Detection (>400% threshold)
- Buy/Sell Ratio Analysis
- Transaction Velocity Tracking
- Whale Activity Indicator
- Momentum Score Calculation

### Scoring Formula

```
SNA Score = (
    Volume Spike * 0.25 +
    Buy/Sell Ratio * 0.20 +
    Liquidity Score * 0.15 +
    Momentum * 0.15 +
    TX Velocity * 0.15 +
    Whale Activity * 0.10
)
```

---

## 🎨 UI Features

- **Glass Morphism Design** - Modern blur effects
- **Dark Theme** - Eye-friendly interface
- **Responsive** - Mobile & Desktop support
- **Real-time Updates** - Auto-refresh setiap 10 detik
- **Animated Charts** - Sparkline visualizations
- **Gradient Effects** - Neon glow animations

---

## 📊 Features Explained

### 1. Live Scanner Tab
- Real-time monitoring token dari berbagai chain
- Sortir berdasarkan Volume, Price Change, AI Score, dll
- Search functionality untuk cari token spesifik
- Circular AI score indicator per token

### 2. Trending Tab
- **Hot Tokens**: Token dengan volume tertinggi
- **Top Gainers**: Token dengan kenaikan harga terbesar
- **Top Losers**: Token dengan penurunan harga terbesar
- **AI Picks**: Token dengan skor AI tertinggi

### 3. Watchlist Tab
- Simpan token favorit untuk monitoring
- Quick access ke statistik token
- Notifikasi perubahan harga

### 4. AI Analysis Panel
- Deep analysis menggunakan Super-Ensemble model
- Pump probability percentage
- Estimated timeframe untuk pump
- Technical indicators (SNA Score, Liquidity, Volume, Market Cap)

---

## ⚙️ Configuration

### Environment Variables

```env
# Backend
TURSO_DATABASE_URL=your_turso_url
TURSO_AUTH_TOKEN=your_token

# Frontend (.env.local)
NEXT_PUBLIC_API_URL=http://localhost:8000
```

---

## 🛠️ Tech Stack

**Backend:**
- Python 3.9+
- FastAPI
- TensorFlow/Keras
- Pandas & NumPy
- Requests
- Colorama

**Frontend:**
- Next.js 16
- React 19
- TailwindCSS 4
- Lucide Icons
- Radix UI

**Database:**
- Turso (LibSQL)

---

## ⚠️ Disclaimer

```
🚨 IMPORTANT: This tool is for EDUCATIONAL and RESEARCH purposes only.

- Cryptocurrency trading involves significant risk
- AI predictions are NOT financial advice
- Past performance does not guarantee future results
- Always Do Your Own Research (DYOR)
- Never invest more than you can afford to lose
```

---

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

MIT License - see [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**CryptoHunter Team**

- Built with ❤️ and ☕
- Powered by AI

---

<p align="center">
  <strong>⚡ Happy Hunting! ⚡</strong>
</p>
