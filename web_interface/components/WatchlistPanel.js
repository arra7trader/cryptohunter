"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Star, Trash2, ExternalLink, ArrowUpRight, ArrowDownRight, Clock } from "lucide-react";

export default function WatchlistPanel({ watchlist, onRemove, onSelectToken }) {
    if (!watchlist || watchlist.length === 0) {
        return (
            <Card className="glass-card border-0">
                <CardContent className="py-16">
                    <div className="flex flex-col items-center justify-center text-center space-y-4">
                        <div className="relative">
                            <div className="absolute inset-0 bg-yellow-500/10 blur-xl rounded-full"></div>
                            <div className="relative p-6 bg-slate-900/80 rounded-2xl border border-white/10">
                                <Star className="h-12 w-12 text-slate-600" />
                            </div>
                        </div>
                        <div className="space-y-2">
                            <h3 className="text-xl font-bold text-slate-200">Watchlist Empty</h3>
                            <p className="text-sm text-slate-400 max-w-md mx-auto">
                                Add tokens to your watchlist by clicking the star icon in the scanner. 
                                Track your favorite tokens and get quick access to their performance.
                            </p>
                        </div>
                    </div>
                </CardContent>
            </Card>
        );
    }

    const formatPrice = (price) => {
        if (typeof price !== 'number') return '$0.00';
        if (price > 1) return `$${price.toFixed(4)}`;
        if (price > 0.0001) return `$${price.toFixed(6)}`;
        return `$${price.toFixed(10)}`;
    };

    const formatNumber = (num) => {
        if (typeof num !== 'number') return '$0';
        if (num >= 1000000) return `$${(num / 1000000).toFixed(2)}M`;
        if (num >= 1000) return `$${(num / 1000).toFixed(1)}K`;
        return `$${num.toFixed(0)}`;
    };

    const formatDate = (dateStr) => {
        if (!dateStr) return 'Unknown';
        const date = new Date(dateStr);
        // Use consistent format to avoid hydration mismatch (no locale dependency)
        const year = date.getFullYear();
        const month = String(date.getMonth() + 1).padStart(2, '0');
        const day = String(date.getDate()).padStart(2, '0');
        const hours = String(date.getHours()).padStart(2, '0');
        const minutes = String(date.getMinutes()).padStart(2, '0');
        return `${year}-${month}-${day} ${hours}:${minutes}`;
    };

    return (
        <div className="space-y-4">
            {/* Header */}
            <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                    <div className="p-2 rounded-lg bg-yellow-500/10">
                        <Star className="h-5 w-5 text-yellow-400 fill-yellow-400" />
                    </div>
                    <div>
                        <h2 className="text-lg font-bold text-white">Your Watchlist</h2>
                        <p className="text-xs text-slate-400">{watchlist.length} tokens tracked</p>
                    </div>
                </div>
            </div>

            {/* Watchlist Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {watchlist.map((token, idx) => (
                    <Card 
                        key={token.token_address || idx}
                        className="glass-card-hover border-0 cursor-pointer group"
                        onClick={() => onSelectToken && onSelectToken(token)}
                    >
                        <CardContent className="p-4">
                            {/* Header */}
                            <div className="flex items-start justify-between mb-4">
                                <div>
                                    <div className="flex items-center gap-2">
                                        <span className="text-xl font-bold text-white group-hover:text-cyan-400 transition-colors">
                                            {token.token}
                                        </span>
                                        {token.status === "PUMP" && (
                                            <span className="status-pump px-1.5 py-0.5 rounded text-[9px] font-bold">
                                                🚀
                                            </span>
                                        )}
                                    </div>
                                    <div className="flex items-center gap-2 mt-1">
                                        <Badge variant="outline" className="text-[9px] h-4 px-1.5 border-slate-700 bg-black/50 text-slate-500">
                                            {token.chain}
                                        </Badge>
                                        <span className="text-[10px] text-slate-600 font-mono">
                                            {token.token_address?.substring(0, 6)}...
                                        </span>
                                    </div>
                                </div>
                                <Button
                                    variant="ghost"
                                    size="sm"
                                    className="h-8 w-8 p-0 text-slate-500 hover:text-rose-400 hover:bg-rose-500/10"
                                    onClick={(e) => {
                                        e.stopPropagation();
                                        onRemove(token.token_address);
                                    }}
                                >
                                    <Trash2 className="h-4 w-4" />
                                </Button>
                            </div>

                            {/* Price Info */}
                            <div className="space-y-3">
                                <div className="flex items-center justify-between">
                                    <span className="text-slate-400 text-sm">Price</span>
                                    <span className="font-mono text-white font-medium">
                                        {formatPrice(token.price_usd)}
                                    </span>
                                </div>

                                <div className="flex items-center justify-between">
                                    <span className="text-slate-400 text-sm">Change (1H)</span>
                                    <div className={`flex items-center gap-1 font-bold ${
                                        (token.price_change_1h || 0) >= 0 ? 'text-emerald-400' : 'text-rose-400'
                                    }`}>
                                        {(token.price_change_1h || 0) >= 0 
                                            ? <ArrowUpRight className="h-4 w-4" />
                                            : <ArrowDownRight className="h-4 w-4" />
                                        }
                                        {Math.abs(token.price_change_1h || 0).toFixed(2)}%
                                    </div>
                                </div>

                                <div className="flex items-center justify-between">
                                    <span className="text-slate-400 text-sm">Liquidity</span>
                                    <span className="font-mono text-slate-300 text-sm">
                                        {formatNumber(token.liquidity_usd)}
                                    </span>
                                </div>

                                <div className="flex items-center justify-between">
                                    <span className="text-slate-400 text-sm">AI Score</span>
                                    <div className="flex items-center gap-2">
                                        <div className={`px-2 py-0.5 rounded text-xs font-bold ${
                                            (token.prediction?.confidence || 0) >= 70 
                                                ? 'bg-emerald-500/20 text-emerald-400' 
                                                : 'bg-yellow-500/20 text-yellow-400'
                                        }`}>
                                            {token.prediction?.confidence || 0}%
                                        </div>
                                    </div>
                                </div>
                            </div>

                            {/* Footer */}
                            <div className="mt-4 pt-3 border-t border-white/5 flex items-center justify-between">
                                <div className="flex items-center gap-1 text-[10px] text-slate-500">
                                    <Clock className="h-3 w-3" />
                                    Added: {formatDate(token.added_at)}
                                </div>
                                <Button
                                    size="sm"
                                    variant="ghost"
                                    className="h-6 px-2 text-[10px] text-slate-400 hover:text-cyan-400"
                                    onClick={(e) => {
                                        e.stopPropagation();
                                        window.open(`https://dexscreener.com/${token.chain?.toLowerCase()}/${token.token_address}`, '_blank');
                                    }}
                                >
                                    <ExternalLink className="h-3 w-3 mr-1" />
                                    DEX
                                </Button>
                            </div>
                        </CardContent>
                    </Card>
                ))}
            </div>
        </div>
    );
}
