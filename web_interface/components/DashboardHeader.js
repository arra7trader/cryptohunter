"use client";

import { useState, useEffect } from "react";
import { Activity, BarChart3, TrendingUp, TrendingDown, Zap, Target, Flame, Wallet } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { WalletMultiButton } from "@solana/wallet-adapter-react-ui";

export default function DashboardHeader({ data, trending }) {
    const {
        total_volume = 0,
        avg_change = 0,
        gainers = 0,
        losers = 0,
        active_tokens = 0,
        top_gainer = null,
        top_loser = null,
        market_sentiment = "neutral",
        scan_progress = 100,
        scan_status = "Idle"
    } = data || {};

    const [sentimentData, setSentimentData] = useState(null);
    const [isClient, setIsClient] = useState(false);

    useEffect(() => {
        setIsClient(true);
        const fetchSentiment = async () => {
            try {
                // Assuming apiBase is available or using relative path if proxy is set up
                // If not, we might need to hardcode localhost:8000 or use an env var
                const res = await fetch('http://localhost:8000/api/market/sentiment');
                if (res.ok) {
                    const data = await res.json();
                    setSentimentData(data);
                }
            } catch (error) {
                console.error("Failed to fetch sentiment:", error);
            }
        };

        fetchSentiment();
    }, []);

    // Use fetched sentiment or fallback to props
    const currentSentiment = sentimentData ? sentimentData.classification?.toLowerCase() : market_sentiment;
    const sentimentValue = sentimentData ? sentimentData.value : null;

    const getSentimentColor = () => {
        const sent = currentSentiment?.toLowerCase() || "";
        if (sent.includes("bull") || sent.includes("greed")) return "text-emerald-400";
        if (sent.includes("bear") || sent.includes("fear")) return "text-rose-400";
        return "text-slate-400";
    };

    const getSentimentIcon = () => {
        const sent = currentSentiment?.toLowerCase() || "";
        if (sent.includes("bull") || sent.includes("greed")) return <TrendingUp className="h-5 w-5" />;
        if (sent.includes("bear") || sent.includes("fear")) return <TrendingDown className="h-5 w-5" />;
        return <Activity className="h-5 w-5" />;
    };

    return (
        <div className="space-y-4">
            {/* Header Top Bar */}
            <div className="flex justify-between items-center mb-4">
                <div>
                    <h1 className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-cyan-400 to-purple-400">
                        CryptoHunter AI
                    </h1>
                    <p className="text-xs text-slate-400">Advanced Alpha Scanner & Predictor</p>
                </div>

                {/* Wallet Connection Button */}
                <div className="wallet-adapter-button-trigger">
                    {isClient && <WalletMultiButton />}
                </div>
            </div>

            {/* Scan Progress Bar */}
            {scan_status !== "Idle" && (
                <div className="glass-card rounded-xl p-4">
                    <div className="flex items-center justify-between mb-2">
                        <span className="text-sm text-cyan-400 font-medium flex items-center gap-2">
                            <Zap className="h-4 w-4 animate-pulse" />
                            {scan_status}
                        </span>
                        <span className="text-sm text-slate-400">{scan_progress}%</span>
                    </div>
                    <div className="progress-bar">
                        <div className="progress-bar-fill" style={{ width: `${scan_progress}%` }}></div>
                    </div>
                </div>
            )}

            {/* Main Stats Grid */}
            <div className="grid gap-4 grid-cols-2 lg:grid-cols-5">
                {/* Active Scans */}
                <div className="stat-card glass-card-hover">
                    <div className="flex items-center justify-between">
                        <div className="p-2 rounded-lg bg-cyan-500/10">
                            <Activity className="h-5 w-5 text-cyan-400" />
                        </div>
                        <span className="text-[10px] text-slate-500 uppercase tracking-wider">Active</span>
                    </div>
                    <div className="mt-3">
                        <div className="text-3xl font-bold text-white">{active_tokens}</div>
                        <p className="text-xs text-slate-400 mt-1">Tokens Monitored</p>
                    </div>
                </div>

                {/* Total Volume */}
                <div className="stat-card glass-card-hover">
                    <div className="flex items-center justify-between">
                        <div className="p-2 rounded-lg bg-purple-500/10">
                            <BarChart3 className="h-5 w-5 text-purple-400" />
                        </div>
                        <span className="text-[10px] text-slate-500 uppercase tracking-wider">Volume</span>
                    </div>
                    <div className="mt-3">
                        <div className="text-3xl font-bold text-white">
                            ${((total_volume || 0) / 1000000).toFixed(2)}M
                        </div>
                        <p className="text-xs text-slate-400 mt-1">Past 1-Hour</p>
                    </div>
                </div>

                {/* Gainers */}
                <div className="stat-card glass-card-hover border-l-2 border-l-emerald-500/50">
                    <div className="flex items-center justify-between">
                        <div className="p-2 rounded-lg bg-emerald-500/10">
                            <TrendingUp className="h-5 w-5 text-emerald-400" />
                        </div>
                        <span className="text-[10px] text-slate-500 uppercase tracking-wider">Gainers</span>
                    </div>
                    <div className="mt-3">
                        <div className="text-3xl font-bold text-emerald-400">+{gainers}</div>
                        <p className="text-xs text-slate-400 mt-1">
                            {top_gainer ? `Top: ${top_gainer.token} (+${top_gainer.change?.toFixed(1)}%)` : 'Loading...'}
                        </p>
                    </div>
                </div>

                {/* Losers */}
                <div className="stat-card glass-card-hover border-l-2 border-l-rose-500/50">
                    <div className="flex items-center justify-between">
                        <div className="p-2 rounded-lg bg-rose-500/10">
                            <TrendingDown className="h-5 w-5 text-rose-400" />
                        </div>
                        <span className="text-[10px] text-slate-500 uppercase tracking-wider">Losers</span>
                    </div>
                    <div className="mt-3">
                        <div className="text-3xl font-bold text-rose-400">-{losers}</div>
                        <p className="text-xs text-slate-400 mt-1">
                            {top_loser ? `Top: ${top_loser.token} (${top_loser.change?.toFixed(1)}%)` : 'Loading...'}
                        </p>
                    </div>
                </div>

                {/* Market Sentiment (Enhanced) */}
                <div className="stat-card glass-card-hover col-span-2 lg:col-span-1">
                    <div className="flex items-center justify-between">
                        <div className={`p-2 rounded-lg ${currentSentiment && (currentSentiment.includes('bull') || currentSentiment.includes('greed')) ? 'bg-emerald-500/10' :
                            currentSentiment && (currentSentiment.includes('bear') || currentSentiment.includes('fear')) ? 'bg-rose-500/10' : 'bg-slate-500/10'
                            }`}>
                            {getSentimentIcon()}
                        </div>
                        <span className="text-[10px] text-slate-500 uppercase tracking-wider">Sentiment</span>
                    </div>
                    <div className="mt-3">
                        <div className={`text-2xl font-bold uppercase ${getSentimentColor()}`}>
                            {currentSentiment || "Neutral"}
                        </div>
                        <p className="text-xs text-slate-400 mt-1">
                            {sentimentValue ? `Index: ${sentimentValue}` : `Avg: ${avg_change >= 0 ? '+' : ''}${avg_change?.toFixed(2)}%`}
                        </p>
                    </div>
                </div>
            </div>

            {/* Hot Tokens Quick View */}
            {trending?.hot && trending.hot.length > 0 && (
                <div className="glass-card rounded-xl p-4">
                    <div className="flex items-center gap-2 mb-3">
                        <Flame className="h-4 w-4 text-orange-400" />
                        <span className="text-sm font-semibold text-white">Hot Right Now</span>
                    </div>
                    <div className="flex flex-wrap gap-2">
                        {trending.hot.slice(0, 5).map((token, idx) => (
                            <div
                                key={idx}
                                className="flex items-center gap-2 px-3 py-1.5 bg-slate-800/50 rounded-full border border-slate-700/50 hover:border-cyan-500/50 transition-all cursor-pointer"
                            >
                                <span className="text-sm font-medium text-white">{token.token}</span>
                                <span className={`text-xs font-bold ${token.price_change_1h >= 0 ? 'text-emerald-400' : 'text-rose-400'
                                    }`}>
                                    {token.price_change_1h >= 0 ? '+' : ''}{token.price_change_1h?.toFixed(1)}%
                                </span>
                            </div>
                        ))}
                    </div>
                </div>
            )}
        </div>
    );
}
