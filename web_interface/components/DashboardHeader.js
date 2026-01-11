
"use client";

import { Activity, BarChart3, TrendingUp, TrendingDown, Zap } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";

export default function DashboardHeader({ data }) {
    const { total_volume, avg_change, gainers, losers, active_tokens } = data || {
        total_volume: 0,
        avg_change: 0,
        gainers: 0,
        losers: 0,
        active_tokens: 0,
    };

    return (
        <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-4">
            <Card className="glass-card border-l-4 border-l-cyan-500 hover:border-l-cyan-400 transition-all duration-300">
                <CardContent className="p-6 flex flex-col items-start space-y-3">
                    <div className="flex items-center space-x-2 text-cyan-400">
                        <Activity className="h-5 w-5" />
                        <span className="text-sm font-semibold uppercase tracking-wider">Active Scans</span>
                    </div>
                    <div className="text-4xl font-bold text-white text-glow">{active_tokens}</div>
                    <p className="text-xs text-slate-300">Live Tokens Monitored</p>
                </CardContent>
            </Card>

            <Card className="glass-card border-l-4 border-l-violet-500 hover:border-l-violet-400 transition-all duration-300">
                <CardContent className="p-6 flex flex-col items-start space-y-3">
                    <div className="flex items-center space-x-2 text-violet-400">
                        <BarChart3 className="h-5 w-5" />
                        <span className="text-sm font-semibold uppercase tracking-wider">Total Volume</span>
                    </div>
                    <div className="text-4xl font-bold text-white">${((total_volume || 0) / 1000000).toFixed(2)}M</div>
                    <p className="text-xs text-slate-300">Past 1-Hour Aggregated</p>
                </CardContent>
            </Card>

            <Card className="glass-card border-l-4 border-l-emerald-500 hover:border-l-emerald-400 transition-all duration-300">
                <CardContent className="p-6 flex flex-col items-start space-y-3">
                    <div className="flex items-center space-x-2 text-emerald-400">
                        <TrendingUp className="h-5 w-5" />
                        <span className="text-sm font-semibold uppercase tracking-wider">Gainers</span>
                    </div>
                    <div className="text-4xl font-bold text-emerald-400">+{gainers}</div>
                    <p className="text-xs text-slate-300">Avg Change: <span className="text-emerald-400">+{Math.abs(avg_change || 0).toFixed(2)}%</span></p>
                </CardContent>
            </Card>

            <Card className="glass-card border-l-4 border-l-rose-500 hover:border-l-rose-400 transition-all duration-300">
                <CardContent className="p-6 flex flex-col items-start space-y-3">
                    <div className="flex items-center space-x-2 text-rose-400">
                        <TrendingDown className="h-5 w-5" />
                        <span className="text-sm font-semibold uppercase tracking-wider">Losers</span>
                    </div>
                    <div className="text-4xl font-bold text-rose-500">-{losers}</div>
                    <p className="text-xs text-slate-300">Volatile Market Detected</p>
                </CardContent>
            </Card>
        </div>
    );
}
