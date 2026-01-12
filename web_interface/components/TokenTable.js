"use client";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ArrowUpRight, ArrowDownRight, RefreshCw, ExternalLink, Star, StarOff } from "lucide-react";

export default function TokenTable({ tokens, onSelectToken, onAddToWatchlist, watchlist = [] }) {
    if (!tokens || !Array.isArray(tokens) || tokens.length === 0) {
        return (
            <div className="glass-card rounded-xl p-8">
                <div className="flex flex-col items-center justify-center h-64 text-slate-400">
                    <RefreshCw className="h-10 w-10 mb-4 animate-spin text-cyan-500" />
                    <p className="text-lg font-medium">Scanning Markets...</p>
                    <p className="text-sm text-slate-500 mt-2">Fetching latest token data from DexScreener</p>
                </div>
            </div>
        );
    }

    const formatPrice = (price) => {
        if (typeof price !== 'number') return '$0.00';
        if (price > 1) return `$${price.toFixed(4)}`;
        if (price > 0.0001) return `$${price.toFixed(6)}`;
        if (price > 0.00000001) return `$${price.toFixed(10)}`;
        return `$${price.toFixed(14)}`;
    };

    const formatNumber = (num) => {
        if (typeof num !== 'number') return '$0';
        if (num >= 1000000) return `$${(num / 1000000).toFixed(2)}M`;
        if (num >= 1000) return `$${(num / 1000).toFixed(1)}K`;
        return `$${num.toFixed(0)}`;
    };

    const isWatchlisted = (address) => {
        return watchlist.some(w => w.token_address === address);
    };

    // Mini sparkline component
    const Sparkline = ({ data, positive }) => {
        if (!data || data.length === 0) {
            return <div className="w-16 h-6 bg-slate-800/50 rounded animate-pulse"></div>;
        }
        
        const max = Math.max(...data);
        const min = Math.min(...data);
        const range = max - min || 1;
        
        return (
            <div className="sparkline">
                {data.slice(-15).map((val, i) => (
                    <div 
                        key={i}
                        className={`sparkline-bar ${positive ? '' : 'negative'}`}
                        style={{ height: `${((val - min) / range) * 100}%`, minHeight: '2px' }}
                    />
                ))}
            </div>
        );
    };

    return (
        <div className="glass-card rounded-xl overflow-hidden">
            <div className="overflow-x-auto">
                <table className="w-full text-sm text-left">
                    <thead className="bg-gradient-to-r from-slate-800/50 to-slate-900/50 text-slate-400 uppercase text-xs font-bold tracking-wider border-b border-white/5">
                        <tr>
                            <th className="px-4 py-4 w-8"></th>
                            <th className="px-4 py-4">Token</th>
                            <th className="px-4 py-4">Price</th>
                            <th className="px-4 py-4">5M</th>
                            <th className="px-4 py-4">1H</th>
                            <th className="px-4 py-4 hidden lg:table-cell">Chart</th>
                            <th className="px-4 py-4 hidden md:table-cell">Liquidity</th>
                            <th className="px-4 py-4">AI Score</th>
                            <th className="px-4 py-4 text-right">Actions</th>
                        </tr>
                    </thead>
                    <tbody className="divide-y divide-white/5">
                        {tokens.map((token, idx) => (
                            <tr
                                key={token.token_address || idx}
                                className="token-row group hover:bg-gradient-to-r hover:from-cyan-500/5 hover:to-transparent cursor-pointer transition-all duration-200"
                                onClick={() => onSelectToken(token)}
                            >
                                {/* Watchlist Star */}
                                <td className="px-4 py-4">
                                    <button
                                        onClick={(e) => {
                                            e.stopPropagation();
                                            onAddToWatchlist(token);
                                        }}
                                        className={`p-1 rounded-full transition-all ${
                                            isWatchlisted(token.token_address) 
                                                ? 'text-yellow-400 hover:text-yellow-300' 
                                                : 'text-slate-600 hover:text-slate-400'
                                        }`}
                                    >
                                        {isWatchlisted(token.token_address) 
                                            ? <Star className="h-4 w-4 fill-current" />
                                            : <StarOff className="h-4 w-4" />
                                        }
                                    </button>
                                </td>

                                {/* Token Info */}
                                <td className="px-4 py-4">
                                    <div className="flex items-center space-x-3">
                                        <div>
                                            <div className="flex items-center gap-2">
                                                <span className="font-bold text-white text-base group-hover:text-cyan-400 transition-colors">
                                                    {token.token}
                                                </span>
                                                {token.status === "PUMP" && (
                                                    <span className="status-pump px-1.5 py-0.5 rounded text-[10px] font-bold animate-pulse">
                                                        🚀 PUMP
                                                    </span>
                                                )}
                                                {token.status === "CRASH" && (
                                                    <span className="status-dump px-1.5 py-0.5 rounded text-[10px] font-bold">
                                                        💥 CRASH
                                                    </span>
                                                )}
                                            </div>
                                            <div className="flex items-center gap-2 mt-1">
                                                <Badge variant="outline" className="text-[9px] h-4 px-1.5 py-0 border-slate-700 bg-black/50 text-slate-500">
                                                    {token.chain}
                                                </Badge>
                                                <span className="text-[10px] text-slate-600 font-mono">
                                                    {token.token_address?.substring(0, 6)}...
                                                </span>
                                            </div>
                                        </div>
                                    </div>
                                </td>

                                {/* Price */}
                                <td className="px-4 py-4">
                                    <span className="font-mono text-slate-200 font-medium">
                                        {formatPrice(token.price_usd)}
                                    </span>
                                </td>

                                {/* 5M Change */}
                                <td className="px-4 py-4">
                                    <div className={`flex items-center font-bold text-sm ${
                                        token.price_change_5m >= 0 ? 'text-emerald-400' : 'text-rose-400'
                                    }`}>
                                        {token.price_change_5m >= 0 
                                            ? <ArrowUpRight className="h-4 w-4 mr-0.5" /> 
                                            : <ArrowDownRight className="h-4 w-4 mr-0.5" />
                                        }
                                        {Math.abs(token.price_change_5m || 0).toFixed(1)}%
                                    </div>
                                </td>

                                {/* 1H Change */}
                                <td className="px-4 py-4">
                                    <div className={`font-medium text-sm ${
                                        token.price_change_1h >= 0 ? 'text-emerald-400/80' : 'text-rose-400/80'
                                    }`}>
                                        {(token.price_change_1h >= 0 ? '+' : '')}{(token.price_change_1h || 0).toFixed(1)}%
                                    </div>
                                </td>

                                {/* Sparkline */}
                                <td className="px-4 py-4 hidden lg:table-cell">
                                    <Sparkline 
                                        data={token.sparkline || []} 
                                        positive={token.price_change_1h >= 0}
                                    />
                                </td>

                                {/* Liquidity */}
                                <td className="px-4 py-4 hidden md:table-cell">
                                    <span className="text-slate-400 font-mono text-xs">
                                        {formatNumber(token.liquidity_usd)}
                                    </span>
                                </td>

                                {/* AI Score */}
                                <td className="px-4 py-4">
                                    <div className="flex items-center gap-2">
                                        {/* Circular Progress */}
                                        <div className="relative w-10 h-10">
                                            <svg className="circular-progress w-full h-full" viewBox="0 0 36 36">
                                                <circle
                                                    className="text-slate-800"
                                                    strokeWidth="3"
                                                    stroke="currentColor"
                                                    fill="transparent"
                                                    r="16"
                                                    cx="18"
                                                    cy="18"
                                                />
                                                <circle
                                                    className={`${
                                                        (token.prediction?.confidence || 0) >= 75 
                                                            ? 'text-emerald-400' 
                                                            : (token.prediction?.confidence || 0) >= 50 
                                                                ? 'text-yellow-400' 
                                                                : 'text-slate-500'
                                                    }`}
                                                    strokeWidth="3"
                                                    strokeDasharray={`${(token.prediction?.confidence || 0)}, 100`}
                                                    strokeLinecap="round"
                                                    stroke="currentColor"
                                                    fill="transparent"
                                                    r="16"
                                                    cx="18"
                                                    cy="18"
                                                />
                                            </svg>
                                            <span className="absolute inset-0 flex items-center justify-center text-[10px] font-bold text-white">
                                                {token.prediction?.confidence || 0}
                                            </span>
                                        </div>
                                    </div>
                                </td>

                                {/* Actions */}
                                <td className="px-4 py-4 text-right">
                                    <Button
                                        size="sm"
                                        className="h-8 w-8 p-0 bg-transparent hover:bg-cyan-500/20 text-slate-400 hover:text-cyan-400 rounded-full transition-all"
                                        onClick={(e) => {
                                            e.stopPropagation();
                                            window.open(`https://dexscreener.com/${token.chain?.toLowerCase()}/${token.token_address}`, '_blank');
                                        }}
                                    >
                                        <ExternalLink className="h-4 w-4" />
                                    </Button>
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </div>
    );
}
