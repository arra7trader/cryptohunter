"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Flame, TrendingUp, TrendingDown, Sparkles, ArrowUpRight, ArrowDownRight, Zap } from "lucide-react";

export default function TrendingPanel({ trending, onSelectToken }) {
    if (!trending) {
        return (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                {[...Array(4)].map((_, i) => (
                    <Card key={i} className="glass-card border-0">
                        <CardContent className="p-6">
                            <div className="animate-pulse space-y-4">
                                <div className="h-4 bg-slate-800 rounded w-1/2"></div>
                                <div className="space-y-3">
                                    <div className="h-12 bg-slate-800 rounded"></div>
                                    <div className="h-12 bg-slate-800 rounded"></div>
                                    <div className="h-12 bg-slate-800 rounded"></div>
                                </div>
                            </div>
                        </CardContent>
                    </Card>
                ))}
            </div>
        );
    }

    const TrendingItem = ({ token, showChange = true, changeField = "price_change_1h" }) => {
        const change = token[changeField] || 0;
        return (
            <div 
                className="flex items-center justify-between p-3 rounded-lg bg-white/5 hover:bg-white/10 cursor-pointer transition-all group"
                onClick={() => onSelectToken && onSelectToken(token)}
            >
                <div className="flex items-center gap-3">
                    <div className="w-8 h-8 rounded-full bg-gradient-to-br from-cyan-500/20 to-purple-500/20 flex items-center justify-center">
                        <span className="text-xs font-bold text-white">
                            {token.token?.substring(0, 2)}
                        </span>
                    </div>
                    <div>
                        <div className="font-bold text-white group-hover:text-cyan-400 transition-colors">
                            {token.token}
                        </div>
                        <div className="text-[10px] text-slate-500 uppercase">{token.chain}</div>
                    </div>
                </div>
                {showChange && (
                    <div className={`flex items-center gap-1 font-bold text-sm ${
                        change >= 0 ? 'text-emerald-400' : 'text-rose-400'
                    }`}>
                        {change >= 0 ? <ArrowUpRight className="h-4 w-4" /> : <ArrowDownRight className="h-4 w-4" />}
                        {Math.abs(change).toFixed(1)}%
                    </div>
                )}
            </div>
        );
    };

    return (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {/* Hot Tokens */}
            <Card className="glass-card border-0 border-t-2 border-t-orange-500/50">
                <CardHeader className="pb-3">
                    <CardTitle className="flex items-center gap-2 text-base">
                        <div className="p-1.5 rounded-lg bg-orange-500/10">
                            <Flame className="h-4 w-4 text-orange-400" />
                        </div>
                        <span className="text-white">Hot Tokens</span>
                        <Badge className="ml-auto bg-orange-500/20 text-orange-400 border-orange-500/30">
                            Volume
                        </Badge>
                    </CardTitle>
                </CardHeader>
                <CardContent className="space-y-2">
                    {trending.hot?.slice(0, 5).map((token, idx) => (
                        <TrendingItem key={idx} token={token} />
                    )) || <p className="text-slate-500 text-sm">No data</p>}
                </CardContent>
            </Card>

            {/* Top Gainers */}
            <Card className="glass-card border-0 border-t-2 border-t-emerald-500/50">
                <CardHeader className="pb-3">
                    <CardTitle className="flex items-center gap-2 text-base">
                        <div className="p-1.5 rounded-lg bg-emerald-500/10">
                            <TrendingUp className="h-4 w-4 text-emerald-400" />
                        </div>
                        <span className="text-white">Top Gainers</span>
                        <Badge className="ml-auto bg-emerald-500/20 text-emerald-400 border-emerald-500/30">
                            +%
                        </Badge>
                    </CardTitle>
                </CardHeader>
                <CardContent className="space-y-2">
                    {trending.gainers?.slice(0, 5).map((token, idx) => (
                        <TrendingItem key={idx} token={token} />
                    )) || <p className="text-slate-500 text-sm">No data</p>}
                </CardContent>
            </Card>

            {/* Top Losers */}
            <Card className="glass-card border-0 border-t-2 border-t-rose-500/50">
                <CardHeader className="pb-3">
                    <CardTitle className="flex items-center gap-2 text-base">
                        <div className="p-1.5 rounded-lg bg-rose-500/10">
                            <TrendingDown className="h-4 w-4 text-rose-400" />
                        </div>
                        <span className="text-white">Top Losers</span>
                        <Badge className="ml-auto bg-rose-500/20 text-rose-400 border-rose-500/30">
                            -%
                        </Badge>
                    </CardTitle>
                </CardHeader>
                <CardContent className="space-y-2">
                    {trending.losers?.slice(0, 5).map((token, idx) => (
                        <TrendingItem key={idx} token={token} />
                    )) || <p className="text-slate-500 text-sm">No data</p>}
                </CardContent>
            </Card>

            {/* AI Picks */}
            <Card className="glass-card border-0 border-t-2 border-t-purple-500/50">
                <CardHeader className="pb-3">
                    <CardTitle className="flex items-center gap-2 text-base">
                        <div className="p-1.5 rounded-lg bg-purple-500/10">
                            <Sparkles className="h-4 w-4 text-purple-400" />
                        </div>
                        <span className="text-white">AI Picks</span>
                        <Badge className="ml-auto bg-purple-500/20 text-purple-400 border-purple-500/30">
                            <Zap className="h-3 w-3 mr-1" />
                            Score
                        </Badge>
                    </CardTitle>
                </CardHeader>
                <CardContent className="space-y-2">
                    {trending.ai_picks?.slice(0, 5).map((token, idx) => (
                        <div 
                            key={idx}
                            className="flex items-center justify-between p-3 rounded-lg bg-white/5 hover:bg-white/10 cursor-pointer transition-all group"
                            onClick={() => onSelectToken && onSelectToken(token)}
                        >
                            <div className="flex items-center gap-3">
                                <div className="w-8 h-8 rounded-full bg-gradient-to-br from-purple-500/20 to-cyan-500/20 flex items-center justify-center">
                                    <span className="text-xs font-bold text-white">
                                        {token.token?.substring(0, 2)}
                                    </span>
                                </div>
                                <div>
                                    <div className="font-bold text-white group-hover:text-purple-400 transition-colors">
                                        {token.token}
                                    </div>
                                    <div className="text-[10px] text-slate-500 uppercase">{token.chain}</div>
                                </div>
                            </div>
                            <div className="flex items-center gap-2">
                                <div className="relative w-8 h-8">
                                    <svg className="circular-progress w-full h-full" viewBox="0 0 36 36">
                                        <circle
                                            className="text-slate-800"
                                            strokeWidth="3"
                                            stroke="currentColor"
                                            fill="transparent"
                                            r="14"
                                            cx="18"
                                            cy="18"
                                        />
                                        <circle
                                            className="text-purple-400"
                                            strokeWidth="3"
                                            strokeDasharray={`${token.sna_score || 0}, 100`}
                                            strokeLinecap="round"
                                            stroke="currentColor"
                                            fill="transparent"
                                            r="14"
                                            cx="18"
                                            cy="18"
                                        />
                                    </svg>
                                    <span className="absolute inset-0 flex items-center justify-center text-[9px] font-bold text-white">
                                        {Math.round(token.sna_score || 0)}
                                    </span>
                                </div>
                            </div>
                        </div>
                    )) || <p className="text-slate-500 text-sm">No data</p>}
                </CardContent>
            </Card>
        </div>
    );
}
