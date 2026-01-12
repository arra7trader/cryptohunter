"use client";

import { useEffect, useState } from "react";
import { 
  RefreshCw, Zap, Activity, TrendingUp, TrendingDown,
  Star, Sparkles, Shield, Cpu, BarChart3, ExternalLink, Eye,
  Rocket, AlertTriangle, Clock, Target, StopCircle, Coins, Brain
} from "lucide-react";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export default function Home() {
  const [tokens, setTokens] = useState([]);
  const [pumpSignals, setPumpSignals] = useState([]);
  const [indodaxTokens, setIndodaxTokens] = useState([]);
  const [indodaxSummary, setIndodaxSummary] = useState(null);
  const [aiForecast, setAiForecast] = useState(null);
  const [forecastLoading, setForecastLoading] = useState(false);
  const [selectedForecastCoin, setSelectedForecastCoin] = useState(null);
  const [loading, setLoading] = useState(true);
  const [lastUpdate, setLastUpdate] = useState(null);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState("scanner"); // "scanner", "pump", "indodax", "ai"

  // Fetch data
  const fetchData = async () => {
    try {
      setLoading(true);
      setError(null);
      
      // Fetch tokens (DexScreener)
      const res = await fetch(`${API_BASE}/api/tokens`);
      
      if (res.ok) {
        const data = await res.json();
        // Filter hanya token dengan confidence >= 70%
        const filteredData = Array.isArray(data) 
          ? data.filter(token => {
              const confidence = token.prediction?.confidence || token.sna_score || 0;
              return confidence >= 70;
            })
          : [];
        setTokens(filteredData);
      } else {
        setError(`API Error: ${res.status}`);
      }
      
      // Fetch pump signals
      try {
        const pumpRes = await fetch(`${API_BASE}/api/pump-signals`);
        if (pumpRes.ok) {
          const pumpData = await pumpRes.json();
          setPumpSignals(pumpData.signals || []);
        }
      } catch (e) {
        console.log("Pump signals not available yet");
      }
      
      // Fetch Indodax data
      try {
        const indodaxRes = await fetch(`${API_BASE}/api/indodax/tokens`);
        if (indodaxRes.ok) {
          const indodaxData = await indodaxRes.json();
          setIndodaxTokens(indodaxData || []);
        }
        
        const summaryRes = await fetch(`${API_BASE}/api/indodax/summary`);
        if (summaryRes.ok) {
          const summaryData = await summaryRes.json();
          setIndodaxSummary(summaryData);
        }
      } catch (e) {
        console.log("Indodax data not available yet");
      }
      
      setLastUpdate(new Date());
    } catch (err) {
      setError(`Failed to fetch: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  // Fetch AI Forecast for a specific coin
  const fetchAiForecast = async (symbol) => {
    setForecastLoading(true);
    setSelectedForecastCoin(symbol);
    try {
      const res = await fetch(`${API_BASE}/api/indodax/forecast/${symbol}`);
      if (res.ok) {
        const data = await res.json();
        setAiForecast(data.forecast);
      } else {
        console.error("Forecast failed");
        setAiForecast(null);
      }
    } catch (e) {
      console.error("AI Forecast error:", e);
      setAiForecast(null);
    } finally {
      setForecastLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 10000); // Auto-refresh every 10 seconds
    return () => clearInterval(interval);
  }, []);

  // Format numbers
  const formatNumber = (num) => {
    if (!num) return "0";
    if (num >= 1e9) return (num / 1e9).toFixed(2) + "B";
    if (num >= 1e6) return (num / 1e6).toFixed(2) + "M";
    if (num >= 1e3) return (num / 1e3).toFixed(2) + "K";
    return num.toFixed(2);
  };
  
  const formatIDR = (num) => {
    if (!num) return "Rp 0";
    if (num >= 1e12) return "Rp " + (num / 1e12).toFixed(2) + "T";
    if (num >= 1e9) return "Rp " + (num / 1e9).toFixed(2) + "M";
    if (num >= 1e6) return "Rp " + (num / 1e6).toFixed(2) + "Jt";
    if (num >= 1e3) return "Rp " + (num / 1e3).toFixed(2) + "Rb";
    return "Rp " + num.toFixed(0);
  };

  const formatPrice = (price) => {
    if (!price) return "$0";
    if (price < 0.000001) return "$" + price.toExponential(2);
    if (price < 0.01) return "$" + price.toFixed(8);
    return "$" + price.toFixed(4);
  };

  return (
    <main className="min-h-screen bg-gradient-to-b from-slate-900 via-slate-900 to-black">
      {/* Background Effects */}
      <div className="fixed inset-0 z-0 overflow-hidden pointer-events-none">
        <div className="absolute top-0 left-1/4 w-96 h-96 bg-cyan-500/10 rounded-full blur-3xl"></div>
        <div className="absolute bottom-0 right-1/4 w-96 h-96 bg-purple-500/10 rounded-full blur-3xl"></div>
      </div>

      {/* Content */}
      <div className="relative z-10">
        {/* Header */}
        <header className="sticky top-0 z-50 backdrop-blur-xl bg-slate-900/80 border-b border-slate-800">
          <div className="container mx-auto px-4 py-4">
            <div className="flex items-center justify-between">
              {/* Logo */}
              <div className="flex items-center gap-3">
                <div className="p-2 bg-gradient-to-r from-cyan-500 to-purple-500 rounded-lg">
                  <Zap className="h-6 w-6 text-white" />
                </div>
                <div>
                  <h1 className="text-xl font-bold text-white">CryptoHunter AI</h1>
                  <p className="text-xs text-slate-400">Real-time Token Scanner</p>
                </div>
              </div>

              {/* Status */}
              <div className="flex items-center gap-4">
                <div className="flex items-center gap-2 px-3 py-1.5 bg-emerald-500/10 border border-emerald-500/20 rounded-full">
                  <span className="relative flex h-2 w-2">
                    <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                    <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
                  </span>
                  <span className="text-xs font-mono text-emerald-400">LIVE</span>
                </div>
                
                <button 
                  onClick={fetchData}
                  disabled={loading}
                  className="flex items-center gap-2 px-4 py-2 bg-cyan-500/10 hover:bg-cyan-500/20 border border-cyan-500/30 rounded-lg text-cyan-400 transition-all"
                >
                  <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
                  <span className="text-sm">Refresh</span>
                </button>
              </div>
            </div>
          </div>
        </header>

        {/* Main Content */}
        <div className="container mx-auto px-4 py-6">
          {/* Stats */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
            <div className="glass-card rounded-xl p-4">
              <p className="text-slate-400 text-sm">Total Tokens</p>
              <p className="text-2xl font-bold text-white">{tokens.length}</p>
            </div>
            <div className="glass-card rounded-xl p-4">
              <p className="text-slate-400 text-sm">Bullish</p>
              <p className="text-2xl font-bold text-emerald-400">
                {tokens.filter(t => t.status === "PUMP").length}
              </p>
            </div>
            <div className="glass-card rounded-xl p-4">
              <p className="text-slate-400 text-sm">Bearish</p>
              <p className="text-2xl font-bold text-rose-400">
                {tokens.filter(t => t.status === "DUMP").length}
              </p>
            </div>
            <div className="glass-card rounded-xl p-4">
              <p className="text-slate-400 text-sm">Last Update</p>
              <p className="text-lg font-mono text-cyan-400">
                {lastUpdate ? lastUpdate.toLocaleTimeString() : "--:--:--"}
              </p>
            </div>
          </div>

          {/* Tab Navigation */}
          <div className="flex gap-2 mb-6">
            <button
              onClick={() => setActiveTab("scanner")}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-all ${
                activeTab === "scanner" 
                  ? "bg-cyan-500/20 border border-cyan-500/50 text-cyan-400" 
                  : "bg-slate-800/50 border border-slate-700 text-slate-400 hover:bg-slate-800"
              }`}
            >
              <Activity className="h-4 w-4" />
              Alpha Scanner
            </button>
            <button
              onClick={() => setActiveTab("pump")}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-all ${
                activeTab === "pump" 
                  ? "bg-purple-500/20 border border-purple-500/50 text-purple-400" 
                  : "bg-slate-800/50 border border-slate-700 text-slate-400 hover:bg-slate-800"
              }`}
            >
              <Rocket className="h-4 w-4" />
              Pump Predictor
            </button>
            <button
              onClick={() => setActiveTab("indodax")}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-all ${
                activeTab === "indodax" 
                  ? "bg-orange-500/20 border border-orange-500/50 text-orange-400" 
                  : "bg-slate-800/50 border border-slate-700 text-slate-400 hover:bg-slate-800"
              }`}
            >
              <Coins className="h-4 w-4" />
              Indodax
              <span className="px-1.5 py-0.5 bg-red-500/20 text-red-400 text-xs rounded">IDR</span>
            </button>
            <button
              onClick={() => setActiveTab("ai")}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-all ${
                activeTab === "ai" 
                  ? "bg-gradient-to-r from-violet-500/20 to-cyan-500/20 border border-violet-500/50 text-violet-400" 
                  : "bg-slate-800/50 border border-slate-700 text-slate-400 hover:bg-slate-800"
              }`}
            >
              <Brain className="h-4 w-4" />
              AI Forecaster
              <span className="px-1.5 py-0.5 bg-violet-500/20 text-violet-400 text-xs rounded animate-pulse">NEW</span>
            </button>
          </div>

          {/* Error Display */}
          {error && (
            <div className="mb-6 p-4 bg-rose-500/10 border border-rose-500/30 rounded-xl text-rose-400">
              <p className="font-semibold">Error:</p>
              <p>{error}</p>
            </div>
          )}

          {/* Indodax Tab */}
          {activeTab === "indodax" && (
            <>
            {/* Indodax Stats */}
            {indodaxSummary && (
              <div className="grid grid-cols-2 md:grid-cols-5 gap-4 mb-6">
                <div className="glass-card rounded-xl p-4 border border-orange-500/20">
                  <p className="text-slate-400 text-sm">Total Pairs</p>
                  <p className="text-2xl font-bold text-white">{indodaxSummary.total_tokens}</p>
                </div>
                <div className="glass-card rounded-xl p-4 border border-emerald-500/20">
                  <p className="text-slate-400 text-sm">Buy Signals</p>
                  <p className="text-2xl font-bold text-emerald-400">{indodaxSummary.buy_signals}</p>
                </div>
                <div className="glass-card rounded-xl p-4 border border-rose-500/20">
                  <p className="text-slate-400 text-sm">Sell Signals</p>
                  <p className="text-2xl font-bold text-rose-400">{indodaxSummary.sell_signals}</p>
                </div>
                <div className="glass-card rounded-xl p-4">
                  <p className="text-slate-400 text-sm">Volume 24h</p>
                  <p className="text-lg font-bold text-cyan-400">{formatIDR(indodaxSummary.total_volume_idr)}</p>
                </div>
                <div className="glass-card rounded-xl p-4">
                  <p className="text-slate-400 text-sm">USD/IDR Rate</p>
                  <p className="text-lg font-mono text-orange-400">Rp {indodaxSummary.usd_idr_rate?.toLocaleString()}</p>
                </div>
              </div>
            )}
            
            <div className="glass-card rounded-xl overflow-hidden mb-6">
              <div className="p-4 border-b border-slate-800 flex items-center justify-between">
                <h2 className="text-lg font-semibold text-white flex items-center gap-2">
                  <Coins className="h-5 w-5 text-orange-400" />
                  Indodax Market Scanner
                  <span className="text-xs px-2 py-1 bg-red-500/20 text-red-400 rounded-full">🇮🇩 Indonesia</span>
                </h2>
                <span className="text-xs text-slate-500">
                  Real-time signals from order book analysis
                </span>
              </div>

              {loading && indodaxTokens.length === 0 ? (
                <div className="p-8 text-center">
                  <RefreshCw className="h-8 w-8 text-orange-400 animate-spin mx-auto mb-3" />
                  <p className="text-slate-400">Loading Indodax data...</p>
                </div>
              ) : indodaxTokens.length === 0 ? (
                <div className="p-8 text-center">
                  <p className="text-slate-400">No Indodax data available</p>
                </div>
              ) : (
                <div className="overflow-x-auto">
                  <table className="w-full">
                    <thead className="bg-slate-800/50">
                      <tr>
                        <th className="text-left p-4 text-slate-400 font-medium text-sm">#</th>
                        <th className="text-left p-4 text-slate-400 font-medium text-sm">Coin</th>
                        <th className="text-right p-4 text-slate-400 font-medium text-sm">Harga (IDR)</th>
                        <th className="text-right p-4 text-slate-400 font-medium text-sm">Harga (USD)</th>
                        <th className="text-right p-4 text-slate-400 font-medium text-sm">24h %</th>
                        <th className="text-right p-4 text-slate-400 font-medium text-sm">Volume 24h</th>
                        <th className="text-right p-4 text-slate-400 font-medium text-sm">Buy %</th>
                        <th className="text-right p-4 text-slate-400 font-medium text-sm">Volatility</th>
                        <th className="text-center p-4 text-slate-400 font-medium text-sm">Signal</th>
                        <th className="text-center p-4 text-slate-400 font-medium text-sm">Confidence</th>
                        <th className="text-center p-4 text-slate-400 font-medium text-sm">Action</th>
                      </tr>
                    </thead>
                    <tbody>
                      {indodaxTokens.map((token, index) => (
                        <tr 
                          key={token.pair_id || index} 
                          className={`border-t border-slate-800/50 hover:bg-slate-800/30 transition-colors ${
                            token.signal_type?.includes('BUY') ? 'bg-emerald-500/5' : 
                            token.signal_type?.includes('SELL') ? 'bg-rose-500/5' : ''
                          }`}
                        >
                          <td className="p-4 text-slate-500 font-mono text-sm">{index + 1}</td>
                          <td className="p-4">
                            <div>
                              <p className="font-semibold text-white">{token.symbol}</p>
                              <p className="text-xs text-slate-500">{token.pair_id}</p>
                            </div>
                          </td>
                          <td className="p-4 text-right font-mono text-white">
                            Rp {token.price_idr?.toLocaleString()}
                          </td>
                          <td className="p-4 text-right font-mono text-slate-400">
                            ${token.price_usd?.toFixed(token.price_usd < 1 ? 6 : 2)}
                          </td>
                          <td className={`p-4 text-right font-mono ${(token.change_24h || 0) >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                            {(token.change_24h || 0) >= 0 ? '+' : ''}{(token.change_24h || 0).toFixed(2)}%
                          </td>
                          <td className="p-4 text-right font-mono text-slate-300">
                            {formatIDR(token.volume_24h_idr)}
                          </td>
                          <td className={`p-4 text-right font-mono ${(token.buy_pressure || 50) >= 55 ? 'text-emerald-400' : (token.buy_pressure || 50) < 45 ? 'text-rose-400' : 'text-slate-300'}`}>
                            {(token.buy_pressure || 50).toFixed(0)}%
                          </td>
                          <td className={`p-4 text-right font-mono ${(token.volatility || 0) > 10 ? 'text-yellow-400' : 'text-slate-400'}`}>
                            {(token.volatility || 0).toFixed(1)}%
                          </td>
                          <td className="p-4 text-center">
                            <span className={`
                              inline-flex items-center gap-1 px-3 py-1.5 rounded-full text-xs font-bold
                              ${token.signal_type === 'STRONG_BUY' 
                                ? "bg-emerald-500/20 text-emerald-400 border border-emerald-500/30" 
                                : token.signal_type === 'BUY'
                                ? "bg-green-500/20 text-green-400 border border-green-500/30"
                                : token.signal_type === 'SELL'
                                ? "bg-orange-500/20 text-orange-400 border border-orange-500/30"
                                : token.signal_type === 'STRONG_SELL'
                                ? "bg-rose-500/20 text-rose-400 border border-rose-500/30"
                                : "bg-slate-500/20 text-slate-400 border border-slate-500/30"
                              }
                            `}>
                              {token.signal_type === 'STRONG_BUY' && <Rocket className="h-3 w-3" />}
                              {token.signal_type === 'BUY' && <TrendingUp className="h-3 w-3" />}
                              {token.signal_type === 'SELL' && <TrendingDown className="h-3 w-3" />}
                              {token.signal_type === 'STRONG_SELL' && <AlertTriangle className="h-3 w-3" />}
                              {token.signal?.split(' ').slice(1).join(' ') || token.signal_type}
                            </span>
                          </td>
                          <td className="p-4 text-center">
                            <span className={`text-sm font-bold ${
                              token.confidence >= 70 ? 'text-emerald-400' : 
                              token.confidence >= 50 ? 'text-cyan-400' : 'text-slate-400'
                            }`}>
                              {token.confidence?.toFixed(0)}%
                            </span>
                          </td>
                          <td className="p-4 text-center">
                            <a 
                              href={`https://indodax.com/market/${token.symbol}`}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="inline-flex items-center gap-1 px-3 py-1.5 bg-orange-500/10 hover:bg-orange-500/20 border border-orange-500/30 rounded-lg text-orange-400 text-xs transition-all"
                            >
                              <ExternalLink className="h-3 w-3" />
                              Trade
                            </a>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
            </>
          )}

          {/* AI Forecaster Tab */}
          {activeTab === "ai" && (
            <>
              <div className="glass-card rounded-xl p-6 mb-6 bg-gradient-to-r from-violet-500/5 to-cyan-500/5 border border-violet-500/20">
                <div className="flex items-center gap-3 mb-4">
                  <Brain className="h-8 w-8 text-violet-400" />
                  <div>
                    <h2 className="text-xl font-bold text-white">AI Price Forecaster</h2>
                    <p className="text-sm text-slate-400">Multi-Model Deep Learning: LSTM, Bi-LSTM, GRU, Conv1D, Transformer</p>
                  </div>
                  <span className="ml-auto px-3 py-1 bg-violet-500/20 text-violet-400 rounded-full text-xs font-bold">Target 90%+ Accuracy</span>
                </div>
                
                <div className="grid grid-cols-5 gap-3 mb-4">
                  <div className="p-3 rounded-lg bg-slate-800/50 text-center">
                    <p className="text-violet-400 font-bold">LSTM</p>
                    <p className="text-xs text-slate-500">Long Short-Term</p>
                  </div>
                  <div className="p-3 rounded-lg bg-slate-800/50 text-center">
                    <p className="text-cyan-400 font-bold">Bi-LSTM</p>
                    <p className="text-xs text-slate-500">Bidirectional</p>
                  </div>
                  <div className="p-3 rounded-lg bg-slate-800/50 text-center">
                    <p className="text-emerald-400 font-bold">GRU</p>
                    <p className="text-xs text-slate-500">Gated Recurrent</p>
                  </div>
                  <div className="p-3 rounded-lg bg-slate-800/50 text-center">
                    <p className="text-orange-400 font-bold">Conv1D</p>
                    <p className="text-xs text-slate-500">Convolutional</p>
                  </div>
                  <div className="p-3 rounded-lg bg-slate-800/50 text-center">
                    <p className="text-rose-400 font-bold">Transformer</p>
                    <p className="text-xs text-slate-500">Time-GPT Style</p>
                  </div>
                </div>

                <p className="text-xs text-slate-500">
                  Select a coin from the list below to get AI-powered price predictions using ensemble of 5 deep learning models.
                </p>
              </div>

              {/* Forecast Result */}
              {forecastLoading && (
                <div className="glass-card rounded-xl p-8 mb-6 text-center">
                  <RefreshCw className="h-10 w-10 text-violet-400 animate-spin mx-auto mb-4" />
                  <p className="text-lg text-white">Training AI Models for {selectedForecastCoin}...</p>
                  <p className="text-sm text-slate-400 mt-2">Running LSTM, Bi-LSTM, GRU, Conv1D, Transformer ensemble</p>
                </div>
              )}

              {aiForecast && !forecastLoading && (
                <div className="glass-card rounded-xl overflow-hidden mb-6 border border-violet-500/30">
                  <div className="p-4 bg-gradient-to-r from-violet-500/10 to-cyan-500/10 border-b border-violet-500/20">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-3">
                        <div className="w-12 h-12 rounded-xl bg-violet-500/20 flex items-center justify-center">
                          <span className="text-2xl font-bold text-violet-400">{aiForecast.symbol?.charAt(0)}</span>
                        </div>
                        <div>
                          <h3 className="text-xl font-bold text-white">{aiForecast.symbol} Forecast</h3>
                          <p className="text-sm text-slate-400">Current: Rp {aiForecast.current_price?.toLocaleString()}</p>
                        </div>
                      </div>
                      <div className={`px-4 py-2 rounded-xl text-lg font-bold ${
                        aiForecast.signal_type?.includes('BUY') ? 'bg-emerald-500/20 text-emerald-400' :
                        aiForecast.signal_type?.includes('SELL') ? 'bg-rose-500/20 text-rose-400' :
                        'bg-slate-500/20 text-slate-400'
                      }`}>
                        {aiForecast.signal}
                      </div>
                    </div>
                  </div>
                  
                  <div className="p-6">
                    {/* Data Sources Info */}
                    <div className="flex items-center gap-2 mb-4 text-xs">
                      <span className="text-slate-500">Data Sources:</span>
                      {aiForecast.data_sources?.map((source, i) => (
                        <span key={i} className="px-2 py-1 bg-cyan-500/20 text-cyan-400 rounded">
                          {source}
                        </span>
                      ))}
                      <span className="text-slate-500 ml-2">
                        ({aiForecast.total_candles?.toLocaleString()} candles)
                      </span>
                    </div>

                    <div className="grid grid-cols-3 gap-4 mb-6">
                      <div className="p-4 rounded-xl bg-slate-800/50 text-center">
                        <p className="text-sm text-slate-400 mb-1">Predicted 1H</p>
                        <p className="text-lg font-mono text-white">Rp {aiForecast.predicted_price_1h?.toLocaleString()}</p>
                        <p className={`text-sm font-bold ${aiForecast.change_1h_pct >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                          {aiForecast.change_1h_pct >= 0 ? '+' : ''}{aiForecast.change_1h_pct?.toFixed(2)}%
                        </p>
                      </div>
                      <div className="p-4 rounded-xl bg-slate-800/50 text-center">
                        <p className="text-sm text-slate-400 mb-1">Predicted 4H</p>
                        <p className="text-lg font-mono text-white">Rp {aiForecast.predicted_price_4h?.toLocaleString()}</p>
                        <p className={`text-sm font-bold ${aiForecast.change_4h_pct >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                          {aiForecast.change_4h_pct >= 0 ? '+' : ''}{aiForecast.change_4h_pct?.toFixed(2)}%
                        </p>
                      </div>
                      <div className="p-4 rounded-xl bg-slate-800/50 text-center">
                        <p className="text-sm text-slate-400 mb-1">Predicted 24H</p>
                        <p className="text-lg font-mono text-white">Rp {aiForecast.predicted_price_24h?.toLocaleString()}</p>
                        <p className={`text-sm font-bold ${aiForecast.change_24h_pct >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                          {aiForecast.change_24h_pct >= 0 ? '+' : ''}{aiForecast.change_24h_pct?.toFixed(2)}%
                        </p>
                      </div>
                    </div>

                    <div className="flex items-center justify-between mb-4">
                      <div className="flex items-center gap-6">
                        <div className="text-center">
                          <p className="text-sm text-slate-400">Model Accuracy</p>
                          <p className={`text-3xl font-bold ${
                            aiForecast.accuracy >= 85 ? 'text-emerald-400' :
                            aiForecast.accuracy >= 70 ? 'text-cyan-400' :
                            aiForecast.accuracy >= 50 ? 'text-yellow-400' : 'text-rose-400'
                          }`}>{aiForecast.accuracy?.toFixed(1)}%</p>
                        </div>
                        <div className="text-center">
                          <p className="text-sm text-slate-400">Risk Level</p>
                          <p className={`text-lg font-bold ${
                            aiForecast.risk_level === 'HIGH' ? 'text-rose-400' :
                            aiForecast.risk_level === 'MEDIUM' ? 'text-yellow-400' : 'text-emerald-400'
                          }`}>{aiForecast.risk_level}</p>
                        </div>
                      </div>
                    </div>

                    <div className="p-4 rounded-xl bg-slate-800/30 mb-4">
                      <p className="text-sm text-slate-400 mb-2">Model Predictions & Accuracy</p>
                      <div className="grid grid-cols-5 gap-2">
                        {aiForecast.model_scores && Object.entries(aiForecast.model_scores).map(([model, score]) => (
                          <div key={model} className="p-2 rounded-lg bg-slate-800/50 text-center">
                            <p className="text-xs text-slate-500 uppercase">{model}</p>
                            <p className={`font-mono font-bold ${score >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                              {score >= 0 ? '+' : ''}{score?.toFixed(2)}%
                            </p>
                            {aiForecast.model_accuracies && (
                              <p className="text-xs text-cyan-400 mt-1">
                                Acc: {aiForecast.model_accuracies[model]?.toFixed(1)}%
                              </p>
                            )}
                          </div>
                        ))}
                      </div>
                    </div>

                    <div className="p-4 rounded-xl bg-violet-500/10 border border-violet-500/20">
                      <p className="text-violet-300 text-sm">{aiForecast.recommendation}</p>
                    </div>
                  </div>
                </div>
              )}

              {/* Coin Selection List */}
              <div className="glass-card rounded-xl overflow-hidden">
                <div className="p-4 border-b border-slate-800">
                  <h3 className="text-lg font-semibold text-white flex items-center gap-2">
                    <Coins className="h-5 w-5 text-orange-400" />
                    Select Coin for AI Forecast
                  </h3>
                  <p className="text-xs text-slate-500 mt-1">Click on any coin to get AI-powered price prediction</p>
                </div>

                <div className="overflow-x-auto">
                  <table className="w-full">
                    <thead className="bg-slate-800/50">
                      <tr>
                        <th className="text-left p-4 text-slate-400 font-medium text-sm">#</th>
                        <th className="text-left p-4 text-slate-400 font-medium text-sm">Coin</th>
                        <th className="text-right p-4 text-slate-400 font-medium text-sm">Price (IDR)</th>
                        <th className="text-right p-4 text-slate-400 font-medium text-sm">24h %</th>
                        <th className="text-right p-4 text-slate-400 font-medium text-sm">Volume</th>
                        <th className="text-center p-4 text-slate-400 font-medium text-sm">Action</th>
                      </tr>
                    </thead>
                    <tbody>
                      {indodaxTokens.slice(0, 50).map((token, index) => (
                        <tr 
                          key={token.pair_id || index}
                          className="border-t border-slate-800/50 hover:bg-violet-500/5 cursor-pointer transition-colors"
                          onClick={() => fetchAiForecast(token.symbol)}
                        >
                          <td className="p-4 text-slate-500 font-mono text-sm">{index + 1}</td>
                          <td className="p-4">
                            <p className="font-semibold text-white">{token.symbol}</p>
                          </td>
                          <td className="p-4 text-right font-mono text-white">
                            Rp {token.price_idr?.toLocaleString()}
                          </td>
                          <td className={`p-4 text-right font-mono ${(token.change_24h || 0) >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                            {(token.change_24h || 0) >= 0 ? '+' : ''}{(token.change_24h || 0).toFixed(2)}%
                          </td>
                          <td className="p-4 text-right font-mono text-slate-300">
                            {formatIDR(token.volume_24h_idr)}
                          </td>
                          <td className="p-4 text-center">
                            <button 
                              className={`inline-flex items-center gap-1 px-4 py-2 rounded-lg text-sm font-semibold transition-all ${
                                selectedForecastCoin === token.symbol && forecastLoading
                                  ? 'bg-violet-500/20 text-violet-400 cursor-wait'
                                  : 'bg-violet-500/10 hover:bg-violet-500/20 border border-violet-500/30 text-violet-400'
                              }`}
                              disabled={forecastLoading}
                            >
                              <Brain className={`h-4 w-4 ${selectedForecastCoin === token.symbol && forecastLoading ? 'animate-pulse' : ''}`} />
                              {selectedForecastCoin === token.symbol && forecastLoading ? 'Training...' : 'AI Forecast'}
                            </button>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </>
          )}

          {/* Pump Predictor Tab */}
          {activeTab === "pump" && (
            <div className="glass-card rounded-xl overflow-hidden mb-6">
              <div className="p-4 border-b border-slate-800 flex items-center justify-between">
                <h2 className="text-lg font-semibold text-white flex items-center gap-2">
                  <Rocket className="h-5 w-5 text-purple-400" />
                  1-Hour Pump/Dump Predictor
                  <span className="text-xs px-2 py-1 bg-yellow-500/20 text-yellow-400 rounded-full">AI Model</span>
                </h2>
                <div className="flex items-center gap-2 text-xs text-slate-500">
                  <Clock className="h-3 w-3" />
                  Predicts movement in next 1 hour
                </div>
              </div>

              {loading && pumpSignals.length === 0 ? (
                <div className="p-8 text-center">
                  <RefreshCw className="h-8 w-8 text-purple-400 animate-spin mx-auto mb-3" />
                  <p className="text-slate-400">Analyzing market signals...</p>
                </div>
              ) : pumpSignals.length === 0 ? (
                <div className="p-8 text-center">
                  <p className="text-slate-400">No signals yet. Waiting for data...</p>
                </div>
              ) : (
                <div className="overflow-x-auto">
                  <table className="w-full">
                    <thead className="bg-slate-800/50">
                      <tr>
                        <th className="text-left p-4 text-slate-400 font-medium text-sm">#</th>
                        <th className="text-left p-4 text-slate-400 font-medium text-sm">Token</th>
                        <th className="text-center p-4 text-slate-400 font-medium text-sm">Signal</th>
                        <th className="text-center p-4 text-slate-400 font-medium text-sm">Confidence</th>
                        <th className="text-center p-4 text-slate-400 font-medium text-sm">Predicted %</th>
                        <th className="text-left p-4 text-slate-400 font-medium text-sm">Key Factors</th>
                        <th className="text-center p-4 text-slate-400 font-medium text-sm">Risk</th>
                        <th className="text-center p-4 text-slate-400 font-medium text-sm">SL / TP</th>
                        <th className="text-left p-4 text-slate-400 font-medium text-sm">Suggestion</th>
                      </tr>
                    </thead>
                    <tbody>
                      {pumpSignals.slice(0, 20).map((signal, index) => (
                        <tr 
                          key={signal.address || index} 
                          className={`border-t border-slate-800/50 hover:bg-slate-800/30 transition-colors ${
                            signal.signal_type?.includes('PUMP') ? 'bg-emerald-500/5' : 
                            signal.signal_type?.includes('DUMP') ? 'bg-rose-500/5' : ''
                          }`}
                        >
                          <td className="p-4 text-slate-500 font-mono text-sm">{index + 1}</td>
                          <td className="p-4">
                            <div>
                              <p className="font-semibold text-white">{signal.token || "Unknown"}</p>
                              <p className="text-xs text-slate-500">{signal.chain}</p>
                            </div>
                          </td>
                          <td className="p-4 text-center">
                            <span className={`
                              inline-flex items-center gap-1 px-3 py-1.5 rounded-full text-sm font-bold
                              ${signal.signal_type === 'STRONG_PUMP' 
                                ? "bg-emerald-500/20 text-emerald-400 border border-emerald-500/30" 
                                : signal.signal_type === 'PUMP'
                                ? "bg-green-500/20 text-green-400 border border-green-500/30"
                                : signal.signal_type === 'DUMP'
                                ? "bg-orange-500/20 text-orange-400 border border-orange-500/30"
                                : signal.signal_type === 'STRONG_DUMP'
                                ? "bg-rose-500/20 text-rose-400 border border-rose-500/30"
                                : "bg-slate-500/20 text-slate-400 border border-slate-500/30"
                              }
                            `}>
                              {signal.signal_type === 'STRONG_PUMP' && <Rocket className="h-4 w-4" />}
                              {signal.signal_type === 'PUMP' && <TrendingUp className="h-4 w-4" />}
                              {signal.signal_type === 'DUMP' && <TrendingDown className="h-4 w-4" />}
                              {signal.signal_type === 'STRONG_DUMP' && <AlertTriangle className="h-4 w-4" />}
                              {signal.signal?.split(' ').slice(1).join(' ') || signal.signal_type}
                            </span>
                          </td>
                          <td className="p-4 text-center">
                            <div className="flex flex-col items-center">
                              <span className={`text-lg font-bold ${
                                signal.confidence >= 80 ? 'text-emerald-400' : 
                                signal.confidence >= 60 ? 'text-cyan-400' : 'text-slate-400'
                              }`}>
                                {signal.confidence?.toFixed(0)}%
                              </span>
                            </div>
                          </td>
                          <td className="p-4 text-center">
                            <span className={`font-mono font-bold ${
                              signal.predicted_change_pct > 0 ? 'text-emerald-400' : 
                              signal.predicted_change_pct < 0 ? 'text-rose-400' : 'text-slate-400'
                            }`}>
                              {signal.predicted_change_pct > 0 ? '+' : ''}{signal.predicted_change_pct?.toFixed(1)}%
                            </span>
                          </td>
                          <td className="p-4">
                            <div className="flex flex-wrap gap-1 max-w-xs">
                              {(signal.key_factors || []).slice(0, 3).map((factor, i) => (
                                <span key={i} className="text-xs px-2 py-0.5 bg-slate-800 rounded text-slate-300">
                                  {factor}
                                </span>
                              ))}
                            </div>
                          </td>
                          <td className="p-4 text-center">
                            <span className={`
                              px-2 py-1 rounded text-xs font-medium
                              ${signal.risk_level === 'VERY HIGH' ? 'bg-rose-500/20 text-rose-400' :
                                signal.risk_level === 'HIGH' ? 'bg-orange-500/20 text-orange-400' :
                                signal.risk_level === 'MEDIUM' ? 'bg-yellow-500/20 text-yellow-400' :
                                'bg-emerald-500/20 text-emerald-400'
                              }
                            `}>
                              {signal.risk_level}
                            </span>
                          </td>
                          <td className="p-4 text-center">
                            <div className="flex items-center justify-center gap-2 text-xs">
                              <span className="text-rose-400 flex items-center gap-1">
                                <StopCircle className="h-3 w-3" />
                                {signal.stop_loss_pct}%
                              </span>
                              <span className="text-slate-500">/</span>
                              <span className="text-emerald-400 flex items-center gap-1">
                                <Target className="h-3 w-3" />
                                {signal.take_profit_pct}%
                              </span>
                            </div>
                          </td>
                          <td className="p-4">
                            <p className={`text-xs ${
                              signal.entry_suggestion?.includes('ENTER') || signal.entry_suggestion?.includes('GOOD') 
                                ? 'text-emerald-400' 
                                : signal.entry_suggestion?.includes('AVOID') || signal.entry_suggestion?.includes('STAY')
                                ? 'text-rose-400'
                                : 'text-slate-400'
                            }`}>
                              {signal.entry_suggestion}
                            </p>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          )}

          {/* Alpha Scanner Tab */}
          {activeTab === "scanner" && (
          <>
          {/* Token Table */}
          <div className="glass-card rounded-xl overflow-hidden">
            <div className="p-4 border-b border-slate-800 flex items-center justify-between">
              <h2 className="text-lg font-semibold text-white flex items-center gap-2">
                <Activity className="h-5 w-5 text-cyan-400" />
                Alpha Hunter Scanner
                <span className="text-xs px-2 py-1 bg-purple-500/20 text-purple-400 rounded-full">V3</span>
              </h2>
              <span className="text-xs text-slate-500">
                Showing tokens with 70%+ confidence only
              </span>
            </div>

            {loading && tokens.length === 0 ? (
              <div className="p-8 text-center">
                <RefreshCw className="h-8 w-8 text-cyan-400 animate-spin mx-auto mb-3" />
                <p className="text-slate-400">Scanning for Alpha...</p>
              </div>
            ) : tokens.length === 0 ? (
              <div className="p-8 text-center">
                <p className="text-slate-400">No high-confidence tokens found at the moment</p>
                <p className="text-xs text-slate-500 mt-2">Waiting for alpha signals...</p>
              </div>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead className="bg-slate-800/50">
                    <tr>
                      <th className="text-left p-4 text-slate-400 font-medium text-sm">#</th>
                      <th className="text-left p-4 text-slate-400 font-medium text-sm">Token</th>
                      <th className="text-left p-4 text-slate-400 font-medium text-sm">Chain</th>
                      <th className="text-right p-4 text-slate-400 font-medium text-sm">Price</th>
                      <th className="text-right p-4 text-slate-400 font-medium text-sm">5m %</th>
                      <th className="text-right p-4 text-slate-400 font-medium text-sm">Buy %</th>
                      <th className="text-right p-4 text-slate-400 font-medium text-sm">Volume</th>
                      <th className="text-right p-4 text-slate-400 font-medium text-sm">Liquidity</th>
                      <th className="text-center p-4 text-slate-400 font-medium text-sm">Age</th>
                      <th className="text-center p-4 text-slate-400 font-medium text-sm">Alpha Score</th>
                      <th className="text-center p-4 text-slate-400 font-medium text-sm">Rating</th>
                      <th className="text-center p-4 text-slate-400 font-medium text-sm">Action</th>
                    </tr>
                  </thead>
                  <tbody>
                    {tokens.map((token, index) => (
                      <tr 
                        key={token.token_address || index} 
                        className={`border-t border-slate-800/50 hover:bg-slate-800/30 transition-colors ${token.is_alpha ? 'bg-emerald-500/5' : ''}`}
                      >
                        <td className="p-4 text-slate-500 font-mono text-sm">{index + 1}</td>
                        <td className="p-4">
                          <div>
                            <div className="flex items-center gap-2">
                              <p className="font-semibold text-white">{token.token || "Unknown"}</p>
                              {token.is_alpha && (
                                <span className="px-1.5 py-0.5 bg-emerald-500/20 text-emerald-400 text-xs rounded">ALPHA</span>
                              )}
                            </div>
                            <p className="text-xs text-slate-500 font-mono truncate max-w-32">
                              {token.token_address?.slice(0, 8)}...{token.token_address?.slice(-6)}
                            </p>
                          </div>
                        </td>
                        <td className="p-4">
                          <span className="px-2 py-1 bg-slate-800 rounded text-xs text-slate-300 uppercase">
                            {token.chain || "?"}
                          </span>
                        </td>
                        <td className="p-4 text-right font-mono text-white">
                          {formatPrice(token.price_usd)}
                        </td>
                        <td className={`p-4 text-right font-mono ${(token.price_change_5m || 0) >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                          {(token.price_change_5m || 0) >= 0 ? '+' : ''}{(token.price_change_5m || 0).toFixed(2)}%
                        </td>
                        <td className={`p-4 text-right font-mono ${(token.buy_pressure || 50) >= 55 ? 'text-emerald-400' : (token.buy_pressure || 50) < 45 ? 'text-rose-400' : 'text-slate-300'}`}>
                          {(token.buy_pressure || 50).toFixed(0)}%
                        </td>
                        <td className="p-4 text-right font-mono text-slate-300">
                          ${formatNumber(token.volume_1h)}
                        </td>
                        <td className="p-4 text-right font-mono text-slate-300">
                          ${formatNumber(token.liquidity_usd)}
                        </td>
                        <td className="p-4 text-center">
                          <span className={`text-xs font-mono ${
                            (token.token_age_hours || 999) <= 24 ? 'text-emerald-400' : 
                            (token.token_age_hours || 999) <= 72 ? 'text-yellow-400' : 'text-slate-400'
                          }`}>
                            {(token.token_age_hours || 999) <= 24 ? `${Math.round(token.token_age_hours)}h` :
                             (token.token_age_hours || 999) <= 168 ? `${Math.round((token.token_age_hours || 0) / 24)}d` :
                             '7d+'}
                          </span>
                        </td>
                        <td className="p-4 text-center">
                          <div className="flex items-center justify-center gap-2">
                            <div 
                              className={`w-12 h-2 rounded-full overflow-hidden ${
                                (token.sna_score || 0) >= 80 ? 'bg-emerald-900' : 
                                (token.sna_score || 0) >= 70 ? 'bg-cyan-900' : 'bg-slate-700'
                              }`}
                              title={`Alpha Score: ${(token.sna_score || 0).toFixed(1)}`}
                            >
                              <div 
                                className={`h-full rounded-full ${
                                  (token.sna_score || 0) >= 80 ? 'bg-gradient-to-r from-emerald-500 to-emerald-400' : 
                                  (token.sna_score || 0) >= 70 ? 'bg-gradient-to-r from-cyan-500 to-cyan-400' : 
                                  'bg-gradient-to-r from-slate-500 to-slate-400'
                                }`}
                                style={{ width: `${Math.min(token.sna_score || 0, 100)}%` }}
                              ></div>
                            </div>
                            <span className={`text-xs font-bold font-mono ${
                              (token.sna_score || 0) >= 80 ? 'text-emerald-400' : 
                              (token.sna_score || 0) >= 70 ? 'text-cyan-400' : 'text-slate-400'
                            }`}>
                              {(token.sna_score || 0).toFixed(0)}%
                            </span>
                          </div>
                        </td>
                        <td className="p-4 text-center">
                          <span className={`
                            inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium
                            ${token.alpha_rating?.includes('S-TIER') 
                              ? "bg-yellow-500/20 text-yellow-400 border border-yellow-500/30" 
                              : token.alpha_rating?.includes('A-TIER')
                              ? "bg-emerald-500/20 text-emerald-400 border border-emerald-500/30"
                              : token.alpha_rating?.includes('B-TIER')
                              ? "bg-cyan-500/20 text-cyan-400 border border-cyan-500/30"
                              : "bg-slate-500/20 text-slate-400 border border-slate-500/30"
                            }
                          `}>
                            {token.alpha_rating || "N/A"}
                          </span>
                        </td>
                        <td className="p-4 text-center">
                          <a 
                            href={`https://dexscreener.com/${token.chain?.toLowerCase()}/${token.token_address}`}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="inline-flex items-center gap-1 px-3 py-1.5 bg-cyan-500/10 hover:bg-cyan-500/20 border border-cyan-500/30 rounded-lg text-cyan-400 text-xs transition-all"
                          >
                            <ExternalLink className="h-3 w-3" />
                            View
                          </a>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
          </>
          )}

          {/* Footer */}
          <footer className="text-center py-8 mt-8 border-t border-slate-800">
            <div className="flex items-center justify-center gap-6 text-slate-500 text-sm">
              <div className="flex items-center gap-2">
                <Shield className="h-4 w-4" />
                <span>Secure API</span>
              </div>
              <div className="flex items-center gap-2">
                <Cpu className="h-4 w-4" />
                <span>AI Powered</span>
              </div>
              <div className="flex items-center gap-2">
                <BarChart3 className="h-4 w-4" />
                <span>Real-time Data</span>
              </div>
            </div>
            <p className="text-slate-600 text-xs mt-4">
              CryptoHunter AI © 2026 • Not Financial Advice • DYOR
            </p>
          </footer>
        </div>
      </div>
    </main>
  );
}
