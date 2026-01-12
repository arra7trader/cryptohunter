"use client";

import { useEffect, useState, useCallback } from "react";
import DashboardHeader from "@/components/DashboardHeader";
import TokenTable from "@/components/TokenTable";
import TrendingPanel from "@/components/TrendingPanel";
import WatchlistPanel from "@/components/WatchlistPanel";
import AIAnalysisPanel from "@/components/AIAnalysisPanel";
import SearchBar from "@/components/SearchBar";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { 
  RefreshCw, Zap, Activity, TrendingUp, 
  Star, Menu, X, Sparkles, Shield, Cpu, BarChart3
} from "lucide-react";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export default function Home() {
  const [tokens, setTokens] = useState([]);
  const [summary, setSummary] = useState(null);
  const [trending, setTrending] = useState(null);
  const [watchlist, setWatchlist] = useState([]);
  const [loading, setLoading] = useState(true);
  const [lastUpdate, setLastUpdate] = useState(null);
  const [selectedToken, setSelectedToken] = useState(null);
  const [activeTab, setActiveTab] = useState("scanner");
  const [searchQuery, setSearchQuery] = useState("");
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [sortBy, setSortBy] = useState("volume_1h");

  // Fetch all data
  const fetchData = useCallback(async () => {
    try {
      setLoading(true);
      const [tokensRes, summaryRes, trendingRes, watchlistRes] = await Promise.all([
        fetch(`${API_BASE}/api/tokens?sort_by=${sortBy}`),
        fetch(`${API_BASE}/api/market/summary`),
        fetch(`${API_BASE}/api/trending`),
        fetch(`${API_BASE}/api/watchlist`)
      ]);

      if (tokensRes.ok) {
        const tokensData = await tokensRes.json();
        setTokens(Array.isArray(tokensData) ? tokensData : []);
      }
      
      if (summaryRes.ok) {
        const summaryData = await summaryRes.json();
        setSummary(summaryData);
      }
      
      if (trendingRes.ok) {
        const trendingData = await trendingRes.json();
        setTrending(trendingData);
      }
      
      if (watchlistRes.ok) {
        const watchlistData = await watchlistRes.json();
        setWatchlist(Array.isArray(watchlistData) ? watchlistData : []);
      }
      
      setLastUpdate(new Date());
    } catch (error) {
      console.error("Failed to fetch data:", error);
    } finally {
      setLoading(false);
    }
  }, [sortBy]);

  useEffect(() => {
    setLastUpdate(new Date());
    fetchData();
    const interval = setInterval(fetchData, 10000);
    return () => clearInterval(interval);
  }, [fetchData]);

  // Add to watchlist
  const handleAddToWatchlist = async (token) => {
    try {
      const res = await fetch(`${API_BASE}/api/watchlist`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          token_address: token.token_address,
          chain: token.chain
        })
      });
      if (res.ok) {
        fetchData();
      }
    } catch (error) {
      console.error("Failed to add to watchlist:", error);
    }
  };

  // Remove from watchlist
  const handleRemoveFromWatchlist = async (address) => {
    try {
      await fetch(`${API_BASE}/api/watchlist/${address}`, { method: "DELETE" });
      fetchData();
    } catch (error) {
      console.error("Failed to remove from watchlist:", error);
    }
  };

  // Search tokens
  const filteredTokens = searchQuery 
    ? tokens.filter(t => 
        t.token?.toLowerCase().includes(searchQuery.toLowerCase()) ||
        t.token_address?.toLowerCase().includes(searchQuery.toLowerCase())
      )
    : tokens;

  return (
    <main className="min-h-screen relative overflow-hidden">
      {/* Animated Background */}
      <div className="fixed inset-0 z-[-2]">
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-slate-900 via-[#0a0f1d] to-black"></div>
        <div className="absolute top-0 left-1/4 w-[500px] h-[500px] bg-cyan-500/10 rounded-full blur-[120px] animate-pulse"></div>
        <div className="absolute bottom-0 right-1/4 w-[400px] h-[400px] bg-purple-500/10 rounded-full blur-[100px] animate-pulse" style={{animationDelay: '1s'}}></div>
        <div className="absolute top-1/2 left-1/2 w-[300px] h-[300px] bg-blue-500/5 rounded-full blur-[80px] animate-float"></div>
      </div>
      
      {/* Grid Pattern Overlay */}
      <div className="fixed inset-0 z-[-1] pattern-grid opacity-30"></div>

      {/* Top Navigation */}
      <nav className="sticky top-0 z-50 backdrop-blur-xl bg-slate-900/70 border-b border-white/5">
        <div className="container mx-auto px-4 py-3">
          <div className="flex items-center justify-between">
            {/* Logo */}
            <div className="flex items-center gap-4">
              <div className="relative">
                <div className="absolute -inset-2 bg-gradient-to-r from-cyan-500 to-purple-500 rounded-lg blur opacity-30"></div>
                <div className="relative flex items-center gap-2 bg-slate-900 px-3 py-2 rounded-lg border border-white/10">
                  <Zap className="h-6 w-6 text-cyan-400 fill-cyan-400" />
                  <span className="text-xl font-black text-gradient">CRYPTOHUNTER</span>
                </div>
              </div>
              <Badge className="hidden md:flex bg-gradient-to-r from-purple-500/20 to-cyan-500/20 border-purple-500/30 text-purple-300">
                <Sparkles className="h-3 w-3 mr-1" /> AI V3.0
              </Badge>
            </div>

            {/* Desktop Nav */}
            <div className="hidden md:flex items-center gap-6">
              <button 
                onClick={() => setActiveTab("scanner")}
                className={`tab-item ${activeTab === "scanner" ? "active" : ""}`}
              >
                <Activity className="h-4 w-4 inline mr-2" />
                Scanner
              </button>
              <button 
                onClick={() => setActiveTab("trending")}
                className={`tab-item ${activeTab === "trending" ? "active" : ""}`}
              >
                <TrendingUp className="h-4 w-4 inline mr-2" />
                Trending
              </button>
              <button 
                onClick={() => setActiveTab("watchlist")}
                className={`tab-item ${activeTab === "watchlist" ? "active" : ""}`}
              >
                <Star className="h-4 w-4 inline mr-2" />
                Watchlist
                {watchlist.length > 0 && (
                  <span className="ml-2 px-1.5 py-0.5 text-xs bg-cyan-500/20 text-cyan-400 rounded-full">
                    {watchlist.length}
                  </span>
                )}
              </button>
            </div>

            {/* Right Actions */}
            <div className="flex items-center gap-3">
              {/* Status Indicator */}
              <div className="hidden sm:flex items-center gap-2 px-3 py-1.5 bg-emerald-500/10 border border-emerald-500/20 rounded-full">
                <span className="relative flex h-2 w-2">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                  <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
                </span>
                <span className="text-xs font-mono text-emerald-400">LIVE</span>
              </div>

              {/* Refresh Button */}
              <Button 
                variant="outline" 
                size="sm" 
                onClick={fetchData} 
                disabled={loading}
                className="gap-2 border-slate-700 hover:bg-cyan-500/10 hover:text-cyan-400 hover:border-cyan-500/50"
              >
                <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
                <span className="hidden sm:inline">Sync</span>
              </Button>

              {/* Mobile Menu Toggle */}
              <Button 
                variant="ghost" 
                size="sm"
                className="md:hidden"
                onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              >
                {mobileMenuOpen ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
              </Button>
            </div>
          </div>

          {/* Mobile Menu */}
          {mobileMenuOpen && (
            <div className="md:hidden mt-4 pb-2 border-t border-white/5 pt-4">
              <div className="flex flex-col gap-2">
                <button 
                  onClick={() => { setActiveTab("scanner"); setMobileMenuOpen(false); }}
                  className={`flex items-center gap-2 px-4 py-2 rounded-lg ${activeTab === "scanner" ? "bg-cyan-500/20 text-cyan-400" : "text-slate-400"}`}
                >
                  <Activity className="h-4 w-4" /> Scanner
                </button>
                <button 
                  onClick={() => { setActiveTab("trending"); setMobileMenuOpen(false); }}
                  className={`flex items-center gap-2 px-4 py-2 rounded-lg ${activeTab === "trending" ? "bg-cyan-500/20 text-cyan-400" : "text-slate-400"}`}
                >
                  <TrendingUp className="h-4 w-4" /> Trending
                </button>
                <button 
                  onClick={() => { setActiveTab("watchlist"); setMobileMenuOpen(false); }}
                  className={`flex items-center gap-2 px-4 py-2 rounded-lg ${activeTab === "watchlist" ? "bg-cyan-500/20 text-cyan-400" : "text-slate-400"}`}
                >
                  <Star className="h-4 w-4" /> Watchlist ({watchlist.length})
                </button>
              </div>
            </div>
          )}
        </div>
      </nav>

      {/* Main Content */}
      <div className="container mx-auto px-4 py-6 space-y-6">
        
        {/* Stats Dashboard */}
        <DashboardHeader data={summary} trending={trending} />

        {/* Search & Filters Bar */}
        <div className="flex flex-col sm:flex-row gap-4 items-center justify-between">
          <SearchBar value={searchQuery} onChange={setSearchQuery} />
          
          <div className="flex items-center gap-3">
            <span className="text-xs text-slate-500">Sort by:</span>
            <select 
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value)}
              className="bg-slate-800/50 border border-slate-700 rounded-lg px-3 py-1.5 text-sm text-slate-300 focus:border-cyan-500 focus:outline-none"
            >
              <option value="volume_1h">Volume (1H)</option>
              <option value="price_change_5m">Change (5M)</option>
              <option value="price_change_1h">Change (1H)</option>
              <option value="sna_score">AI Score</option>
              <option value="liquidity_usd">Liquidity</option>
              <option value="market_cap">Market Cap</option>
            </select>
          </div>
        </div>

        {/* Tab Content */}
        {activeTab === "scanner" && (
          <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
            {/* Main Table */}
            <div className="xl:col-span-2 space-y-4">
              <div className="flex items-center justify-between">
                <h2 className="text-lg font-semibold flex items-center gap-3 text-white">
                  <span className="relative flex h-3 w-3">
                    <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-cyan-500 opacity-75"></span>
                    <span className="relative inline-flex rounded-full h-3 w-3 bg-cyan-500"></span>
                  </span>
                  Live Market Scanner
                </h2>
                <Badge variant="secondary" className="bg-slate-900 border border-slate-800 text-slate-400 font-mono">
                  {lastUpdate ? lastUpdate.toLocaleTimeString() : '--:--:--'}
                </Badge>
              </div>
              
              <TokenTable 
                tokens={filteredTokens} 
                onSelectToken={setSelectedToken}
                onAddToWatchlist={handleAddToWatchlist}
                watchlist={watchlist}
              />
              
              <p className="text-center text-xs text-slate-500">
                Auto-refreshing every 10s • Powered by DexScreener API
              </p>
            </div>

            {/* AI Analysis Panel */}
            <div className="space-y-4">
              <AIAnalysisPanel 
                selectedToken={selectedToken}
                apiBase={API_BASE}
              />
            </div>
          </div>
        )}

        {activeTab === "trending" && (
          <TrendingPanel trending={trending} onSelectToken={setSelectedToken} />
        )}

        {activeTab === "watchlist" && (
          <WatchlistPanel 
            watchlist={watchlist}
            onRemove={handleRemoveFromWatchlist}
            onSelectToken={setSelectedToken}
          />
        )}

        {/* Footer */}
        <footer className="text-center py-8 border-t border-white/5 mt-12">
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
    </main>
  );
}
