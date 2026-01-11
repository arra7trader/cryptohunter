
"use client";

import { useEffect, useState } from "react";
import DashboardHeader from "@/components/DashboardHeader";
import TokenTable from "@/components/TokenTable";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Loader2, RefreshCw, Zap, Crosshair } from "lucide-react";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export default function Home() {
  const [tokens, setTokens] = useState([]);
  const [summary, setSummary] = useState(null);
  const [loading, setLoading] = useState(true);
  const [lastUpdate, setLastUpdate] = useState(null); // Fix hydration mismatch
  const [selectedToken, setSelectedToken] = useState(null);

  // Fetch data
  const fetchData = async () => {
    try {
      setLoading(true);
      const [tokensRes, summaryRes] = await Promise.all([
        fetch(`${API_BASE}/api/tokens`),
        fetch(`${API_BASE}/api/market/summary`)
      ]);

      if (tokensRes.ok && summaryRes.ok) {
        const tokensData = await tokensRes.json();
        const summaryData = await summaryRes.json();
        setTokens(Array.isArray(tokensData) ? tokensData : []);
        setSummary(summaryData);
        setLastUpdate(new Date());
      }
    } catch (error) {
      console.error("Failed to fetch data:", error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    setLastUpdate(new Date()); // Set initial date on client
    fetchData();
    const interval = setInterval(fetchData, 10000);
    return () => clearInterval(interval);
  }, []);

  // ... inside return ...
  // <Badge variant="secondary" className="bg-slate-800 text-slate-400">
  //   Updated: {lastUpdate ? lastUpdate.toLocaleTimeString() : '...'}
  // </Badge>

  const [selectedModel, setSelectedModel] = useState("bilstm");
  const [analysisLoading, setAnalysisLoading] = useState(false);

  // Deep Analysis Function
  const runDeepAnalysis = async () => {
    if (!selectedToken) return;

    try {
      setAnalysisLoading(true);
      const res = await fetch(`${API_BASE}/api/tokens/${selectedToken.token_address}/predict?chain=${selectedToken.chain}&model=${selectedModel}`);

      if (res.ok) {
        const data = await res.json();
        // Update the selected token with new detailed prediction data
        setSelectedToken(prev => ({
          ...prev,
          prediction: {
            ...prev.prediction,
            confidence: data.prediction.ensemble.confidence,
            pump_in_hours: data.prediction.ensemble.pump_in_hours,
            source: data.model_type // Mark source as the specific model
          },
          detailed_analysis: data // Store full response if needed
        }));
      } else {
        console.error("Analysis failed");
      }
    } catch (error) {
      console.error("Deep analysis error:", error);
    } finally {
      setAnalysisLoading(false);
    }
  };

  return (
    <main className="min-h-screen p-6 space-y-8 relative overflow-hidden">
      {/* Background Decor */}
      <div className="fixed inset-0 z-[-1] bg-[radial-gradient(circle_at_top,_var(--tw-gradient-stops))] from-slate-900 via-[#0a0f1d] to-black"></div>
      <div className="fixed top-0 left-0 w-full h-[500px] bg-gradient-to-b from-cyan-900/10 to-transparent pointer-events-none"></div>

      {/* Navbar */}
      <div className="flex items-center justify-between container mx-auto pt-4">
        <div>
          <h1 className="text-4xl font-bold tracking-tight text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 via-blue-500 to-purple-600 flex items-center gap-3 text-glow">
            <Zap className="text-cyan-400 fill-cyan-400 h-8 w-8" />
            CRYPTOHUNTER
            <span className="text-sm font-light text-slate-300 tracking-[0.2em] border-l border-slate-600 pl-3 ml-1">AI SENTINEL V2.1</span>
          </h1>
          <p className="text-slate-400 text-sm mt-1 ml-11">Advanced DEX Monitoring & Price Prediction Engine</p>
        </div>
        <div className="flex items-center gap-6">
          <div className="text-right hidden md:block">
            <p className="text-[10px] text-slate-400 uppercase tracking-widest font-bold mb-1">System Status</p>
            <div className="flex items-center justify-end gap-2 bg-black/30 px-3 py-1 rounded-full border border-emerald-500/20">
              <span className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
              </span>
              <span className="text-xs font-mono text-emerald-400 font-bold tracking-wider">SYSTEM ONLINE</span>
            </div>
          </div>
          <Button variant="outline" size="sm" onClick={fetchData} disabled={loading} className="gap-2 border-slate-700 hover:bg-cyan-500/10 hover:text-cyan-400 hover:border-cyan-500/50 transition-all">
            <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
            Sync Data
          </Button>
        </div>
      </div>

      <div className="container mx-auto space-y-8">
        {/* Summary Cards */}
        <DashboardHeader data={summary} />

        {/* Main Content Split */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">

          {/* Left: Token List */}
          <div className="lg:col-span-2 space-y-5">
            <div className="flex items-center justify-between px-2">
              <h2 className="text-lg font-semibold flex items-center gap-3 text-white">
                <span className="relative flex h-3 w-3">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-blue-500 opacity-75"></span>
                  <span className="relative inline-flex rounded-full h-3 w-3 bg-blue-500"></span>
                </span>
                Live Market Scanner
              </h2>
              <Badge variant="secondary" className="bg-slate-900 border border-slate-800 text-slate-400 font-mono">
                Last Sync: <span className="text-cyan-400 ml-2">{lastUpdate ? lastUpdate.toLocaleTimeString() : '--:--:--'}</span>
              </Badge>
            </div>
            <TokenTable tokens={tokens} onSelectToken={setSelectedToken} />
            <p className="text-center text-xs text-slate-400 mt-4">
              Auto-refreshing every 10s • Syncing with Python Backend (30s cycle)
            </p>
          </div>

          {/* Right: AI Analysis Panel */}
          <div className="space-y-5">
            <h2 className="text-lg font-semibold flex items-center gap-2 text-white px-2">
              <Zap className="h-5 w-5 text-purple-500" />
              Sentinel AI Analysis
            </h2>

            <Card className="min-h-[500px] border-0 glass-card relative overflow-hidden group">
              {/* Decorative background elements for the card */}
              <div className="absolute top-0 right-0 w-32 h-32 bg-purple-500/10 blur-[50px] rounded-full pointer-events-none"></div>
              <div className="absolute bottom-0 left-0 w-32 h-32 bg-blue-500/10 blur-[50px] rounded-full pointer-events-none"></div>

              {selectedToken ? (
                <>
                  <CardHeader className="relative z-10 border-b border-white/5 pb-2">
                    <div className="flex items-start justify-between">
                      <div>
                        <div className="flex items-center gap-2 mb-2">
                          <Badge className="bg-blue-500/20 text-blue-300 border-blue-500/30 hover:bg-blue-500/30">{selectedToken.chain}</Badge>
                          <span className="text-xs text-slate-400 font-mono">{selectedToken.token_address}</span>
                        </div>
                        <CardTitle className="text-3xl font-black text-transparent bg-clip-text bg-gradient-to-br from-white to-slate-400">
                          {selectedToken.token}
                        </CardTitle>
                      </div>
                      <div className="flex flex-col items-end">
                        <div className="text-xs text-purple-400 font-bold uppercase tracking-wider mb-1">AI Confidence</div>
                        <div className="text-3xl font-bold text-white shadow-purple-500/50 drop-shadow-lg">
                          {selectedToken.prediction?.confidence}%
                        </div>
                      </div>
                    </div>
                  </CardHeader>
                  <CardContent className="space-y-5 pt-4 relative z-10">

                    {/* Model Selector & Action */}
                    <div className="flex items-center gap-2">
                      <select
                        value={selectedModel}
                        onChange={(e) => setSelectedModel(e.target.value)}
                        className="bg-slate-900/80 border border-slate-700 text-slate-300 text-xs rounded-lg p-2.5 focus:ring-cyan-500 focus:border-cyan-500 w-[140px]"
                      >
                        <option value="bilstm">Bi-LSTM (Default)</option>
                        <option value="lstm">LSTM (Standard)</option>
                        <option value="gru">GRU (Fast)</option>
                        <option value="conv1d">Conv1D (Pattern)</option>
                        <option value="transformer">Time-GPT (Slow)</option>
                      </select>
                      <Button
                        onClick={runDeepAnalysis}
                        disabled={analysisLoading}
                        className={`flex-1 bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-500 hover:to-indigo-500 border border-purple-500/30 font-bold tracking-wider ${analysisLoading ? 'opacity-80' : ''}`}
                      >
                        {analysisLoading ? (
                          <><Loader2 className="mr-2 h-4 w-4 animate-spin" /> TRAINING MODEL...</>
                        ) : (
                          <><Zap className="mr-2 h-4 w-4" /> RUN DEEP ANALYSIS</>
                        )}
                      </Button>
                    </div>

                    {/* AI Prediction Block */}
                    <div className="bg-gradient-to-r from-slate-900 to-black rounded-xl p-1 border border-white/10 shadow-inner">
                      <div className="bg-white/5 rounded-lg p-5 grid grid-cols-2 gap-4">
                        <div className="text-center border-r border-white/10">
                          <p className="text-[10px] text-slate-500 uppercase tracking-widest mb-1">Pump Probability</p>
                          <div className={`text-2xl font-bold ${selectedToken.prediction?.confidence > 70 ? 'text-emerald-400' : 'text-yellow-400'}`}>
                            {selectedToken.prediction?.confidence > 70 ? 'VERY HIGH' : 'MODERATE'}
                          </div>
                        </div>
                        <div className="text-center">
                          <p className="text-[10px] text-slate-500 uppercase tracking-widest mb-1">Est. Timeframe</p>
                          <div className="text-2xl font-bold text-cyan-300">
                            &lt; {selectedToken.prediction?.pump_in_hours} Hours
                          </div>
                        </div>
                      </div>
                      <div className="px-4 py-2 bg-black/40 text-[10px] text-center text-slate-500 font-mono border-t border-white/5">
                        Source: {selectedToken.prediction?.source === 'sna_fast' ? 'Rapid Scan (SNA)' : `Deep Learning (${selectedModel.toUpperCase()})`}
                      </div>
                    </div>

                    {/* Stats Grid */}
                    <div className="grid grid-cols-2 gap-3 text-sm">
                      <div className="p-3 rounded-lg bg-white/5 border border-white/5 flex flex-col">
                        <span className="text-slate-400 text-xs text-center mb-1">SNA Score</span>
                        <span className="font-mono text-yellow-400 font-bold text-center text-lg">{(selectedToken.sna_score || 0).toFixed(0)}</span>
                      </div>
                      <div className="p-3 rounded-lg bg-white/5 border border-white/5 flex flex-col">
                        <span className="text-slate-400 text-xs text-center mb-1">Liquidity</span>
                        <span className="font-mono text-emerald-400 font-bold text-center text-lg">${((selectedToken.liquidity_usd || 0) / 1000).toFixed(1)}K</span>
                      </div>
                      <div className="p-3 rounded-lg bg-white/5 border border-white/5 flex flex-col">
                        <span className="text-slate-400 text-xs text-center mb-1">Vol (1H)</span>
                        <span className="font-mono text-white font-bold text-center text-lg">${((selectedToken.volume_1h || 0) / 1000).toFixed(1)}K</span>
                      </div>
                      <div className="p-3 rounded-lg bg-white/5 border border-white/5 flex flex-col">
                        <span className="text-slate-400 text-xs text-center mb-1">Market Cap</span>
                        <span className="font-mono text-blue-300 font-bold text-center text-lg">${((selectedToken.market_cap || 0) / 1000).toFixed(1)}K</span>
                      </div>
                    </div>

                    <Button
                      className="w-full h-12 bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 text-white font-bold tracking-wider rounded-xl shadow-lg shadow-cyan-900/40 border border-white/10 transition-all hover:scale-[1.02]"
                      onClick={() => window.open(`https://dexscreener.com/${selectedToken.chain.toLowerCase()}/${selectedToken.token_address}`, '_blank')}
                    >
                      <Crosshair className="mr-2 h-5 w-5" />
                      ANALYZE ON DEXSCREENER
                    </Button>
                  </CardContent>
                </>
              ) : (
                <div className="h-full flex flex-col items-center justify-center p-8 text-center text-slate-500 space-y-6 min-h-[400px]">
                  <div className="relative">
                    <div className="absolute inset-0 bg-cyan-500/20 blur-xl rounded-full"></div>
                    <Crosshair className="h-20 w-20 text-slate-600 relative z-10" />
                  </div>
                  <div>
                    <h3 className="text-lg font-bold text-slate-300">Awaiting Target Selection</h3>
                    <p className="text-sm mt-2 text-slate-400 max-w-[250px] mx-auto leading-relaxed">
                      Select any token from the Live Scanner to initiate Deep AI Analysis and Probability Scoring.
                    </p>
                  </div>
                </div>
              )}
            </Card>
          </div>
        </div>
      </div>
    </main>
  );
}
