
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
        setTokens(tokensData);
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
  <Badge variant="secondary" className="bg-slate-800 text-slate-400">
    Updated: {lastUpdate ? lastUpdate.toLocaleTimeString() : '...'}
  </Badge>

  return (
    <main className="min-h-screen p-6 space-y-8 bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-slate-900 via-slate-950 to-black">
      {/* Navbar */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-emerald-400 flex items-center gap-2">
            <Zap className="text-blue-400 fill-blue-400 h-6 w-6" />
            CRYPTOHUNTER <span className="text-slate-600 font-light hidden sm:inline">| AI SENTINEL</span>
          </h1>
          <p className="text-slate-500 text-sm mt-1">Real-time DEX Monitoring & AI Prediction System</p>
        </div>
        <div className="flex items-center gap-4">
          <div className="text-right hidden md:block">
            <p className="text-xs text-slate-500 uppercase tracking-widest font-semibold">System Status</p>
            <div className="flex items-center justify-end gap-2">
              <span className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
              </span>
              <span className="text-xs font-mono text-emerald-400">ONLINE</span>
            </div>
          </div>
          <Button variant="outline" size="sm" onClick={fetchData} disabled={loading} className="gap-2 border-slate-700 hover:bg-slate-800">
            <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
            Refresh
          </Button>
        </div>
      </div>

      {/* Summary Cards */}
      <DashboardHeader data={summary} />

      {/* Main Content Split */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">

        {/* Left: Token List */}
        <div className="lg:col-span-2 space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-xl font-semibold flex items-center gap-2">
              <div className="h-2 w-2 rounded-full bg-blue-500"></div>
              Live Scanner
            </h2>
            <Badge variant="secondary" className="bg-slate-800 text-slate-400">
              Updated: {lastUpdate ? lastUpdate.toLocaleTimeString() : '...'}
            </Badge>
          </div>
          <TokenTable tokens={tokens} onSelectToken={setSelectedToken} />
        </div>

        {/* Right: AI Analysis Panel */}
        <div className="space-y-4">
          <h2 className="text-xl font-semibold flex items-center gap-2">
            <div className="h-2 w-2 rounded-full bg-purple-500"></div>
            Sentinel AI Analysis
          </h2>

          <Card className="min-h-[400px] border-slate-800 bg-slate-900/50 backdrop-blur-sm">
            {selectedToken ? (
              <>
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <div>
                      <CardTitle className="text-2xl text-blue-400">{selectedToken.token}</CardTitle>
                      <CardDescription className="font-mono text-xs mt-1">{selectedToken.token_address}</CardDescription>
                    </div>
                    <Badge variant="outline" className="text-xs border-purple-500/50 text-purple-400">
                      AI CONFIDENCE: {selectedToken.prediction?.confidence}%
                    </Badge>
                  </div>
                </CardHeader>
                <CardContent className="space-y-6">
                  {/* AI Prediction Block */}
                  <div className="bg-slate-950/80 rounded-lg p-4 border border-slate-800">
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <p className="text-xs text-slate-500 uppercase tracking-wider">Pump Prob.</p>
                        <div className="text-xl font-bold text-emerald-400">
                          {selectedToken.prediction?.confidence > 70 ? 'HIGH' : 'MODERATE'}
                        </div>
                      </div>
                      <div>
                        <p className="text-xs text-slate-500 uppercase tracking-wider">Target Time</p>
                        <div className="text-xl font-bold text-white">
                          ~{selectedToken.prediction?.pump_in_hours} Hours
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Stats Grid */}
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div className="flex justify-between border-b border-slate-800 pb-2">
                      <span className="text-slate-500">SNA Score</span>
                      <span className="font-mono text-yellow-400">{(selectedToken.sna_score || 0).toFixed(0)}/100</span>
                    </div>
                    <div className="flex justify-between border-b border-slate-800 pb-2">
                      <span className="text-slate-500">Liquidity</span>
                      <span className="font-mono">${((selectedToken.liquidity_usd || 0) / 1000).toFixed(1)}K</span>
                    </div>
                    <div className="flex justify-between border-b border-slate-800 pb-2">
                      <span className="text-slate-500">Volume 1H</span>
                      <span className="font-mono">${((selectedToken.volume_1h || 0) / 1000).toFixed(1)}K</span>
                    </div>
                    <div className="flex justify-between border-b border-slate-800 pb-2">
                      <span className="text-slate-500">Chain</span>
                      <span className="font-mono uppercase text-blue-300">{selectedToken.chain}</span>
                    </div>
                  </div>

                  <Button
                    className="w-full bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white font-bold shadow-lg shadow-purple-900/20"
                    onClick={() => window.open(`https://dexscreener.com/${selectedToken.chain.toLowerCase()}/${selectedToken.token_address}`, '_blank')}
                  >
                    View on DexScreener
                  </Button>
                </CardContent>
              </>
            ) : (
              <div className="h-full flex flex-col items-center justify-center p-8 text-center text-slate-500 space-y-4">
                <Crosshair className="h-16 w-16 text-slate-700" />
                <div>
                  <p className="font-medium text-slate-400">No Token Selected</p>
                  <p className="text-sm mt-1">Select a token from the Live Scanner list to view deep AI analysis and details.</p>
                </div>
              </div>
            )}
          </Card>
        </div>
      </div>
    </main>
  );
}
