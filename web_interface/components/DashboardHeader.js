
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
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
            <Card className="bg-gradient-to-br from-slate-900 to-slate-800 border-slate-700">
                <CardContent className="p-6 flex flex-col items-center justify-center space-y-2">
                    <div className="flex items-center space-x-2 text-slate-400">
                        <Activity className="h-4 w-4" />
                        <span className="text-sm font-medium">Active Scans</span>
                    </div>
                    <div className="text-3xl font-bold">{active_tokens}</div>
                    <p className="text-xs text-slate-500">Live Tokens Monitored</p>
                </CardContent>
            </Card>

            <Card className="bg-gradient-to-br from-slate-900 to-slate-800 border-slate-700">
                <CardContent className="p-6 flex flex-col items-center justify-center space-y-2">
                    <div className="flex items-center space-x-2 text-slate-400">
                        <BarChart3 className="h-4 w-4" />
                        <span className="text-sm font-medium">Total Volume (1H)</span>
                    </div>
                    <div className="text-3xl font-bold">${((total_volume || 0) / 1000000).toFixed(2)}M</div>
                    <p className="text-xs text-slate-500">Across all providers</p>
                </CardContent>
            </Card>

            <Card className="bg-gradient-to-br from-slate-900 to-slate-800 border-slate-700">
                <CardContent className="p-6 flex flex-col items-center justify-center space-y-2">
                    <div className="flex items-center space-x-2 text-slate-400">
                        <TrendingUp className="h-4 w-4 text-green-500" />
                        <span className="text-sm font-medium">Gainers</span>
                    </div>
                    <div className="text-3xl font-bold text-green-500">{gainers}</div>
                    <p className="text-xs text-slate-500">Avg Change: {avg_change > 0 ? "+" : ""}{(avg_change || 0).toFixed(2)}%</p>
                </CardContent>
            </Card>

            <Card className="bg-gradient-to-br from-slate-900 to-slate-800 border-slate-700">
                <CardContent className="p-6 flex flex-col items-center justify-center space-y-2">
                    <div className="flex items-center space-x-2 text-slate-400">
                        <TrendingDown className="h-4 w-4 text-red-500" />
                        <span className="text-sm font-medium">Losers</span>
                    </div>
                    <div className="text-3xl font-bold text-red-500">{losers}</div>
                    <p className="text-xs text-slate-500">Volatile Market</p>
                </CardContent>
            </Card>
        </div>
    );
}
