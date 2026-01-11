
"use client";

import { useEffect, useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ArrowUpRight, ArrowDownRight, RefreshCw, Crosshair, ExternalLink } from "lucide-react";

export default function TokenTable({ tokens, onSelectToken }) {
    if (!tokens || !Array.isArray(tokens) || tokens.length === 0) {
        return (
            <div className="flex flex-col items-center justify-center h-64 text-muted-foreground p-8 border rounded-lg border-dashed border-slate-700 bg-slate-900/50">
                <RefreshCw className="h-8 w-8 mb-4 animate-spin" />
                <p>Scanning DexScreener...</p>
            </div>
        );
    }

    const formatPrice = (price) => {
        if (typeof price !== 'number') return '$0.00';
        if (price > 1) return `$${price.toFixed(4)}`;
        if (price > 0.0001) return `$${price.toFixed(6)}`;
        return `$${price.toFixed(10)}`;
    };

    const formatNumber = (num) => {
        if (typeof num !== 'number') return '0';
        if (num >= 1000000) return `$${(num / 1000000).toFixed(2)}M`;
        if (num >= 1000) return `$${(num / 1000).toFixed(1)}K`;
        return `$${num.toFixed(0)}`;
    };

    return (
        <div className="glass-card rounded-xl overflow-hidden">
            <div className="overflow-x-auto">
                <table className="w-full text-sm text-left">
                    <thead className="bg-white/5 text-slate-300 uppercase text-xs font-bold tracking-wider border-b border-white/10">
                        <tr>
                            <th className="px-6 py-4">Token</th>
                            <th className="px-6 py-4">Price</th>
                            <th className="px-6 py-4">5m Change</th>
                            <th className="px-6 py-4">1h Change</th>
                            <th className="px-6 py-4">Liquidity</th>
                            <th className="px-6 py-4">M. Cap</th>
                            <th className="px-6 py-4">AI Score</th>
                            <th className="px-6 py-4 text-right">Action</th>
                        </tr>
                    </thead>
                    <tbody className="divide-y divide-white/5">
                        {tokens.map((token, idx) => (
                            <tr
                                key={idx}
                                className="group hover:bg-white/5 transition-all duration-200 cursor-pointer"
                                onClick={() => onSelectToken(token)}
                            >
                                <td className="px-6 py-4">
                                    <div className="flex items-center space-x-3">
                                        <div className="font-bold text-white text-lg group-hover:text-cyan-400 transition-colors">
                                            {token.token}
                                        </div>
                                        <Badge variant="outline" className="text-[10px] h-5 px-1.5 py-0 border-slate-700 bg-black/50 text-slate-400">
                                            {token.chain}
                                        </Badge>
                                    </div>
                                    <div className="text-xs text-slate-500 font-mono mt-1 opacity-50 group-hover:opacity-100 transition-opacity">
                                        {token.token_address.substring(0, 6)}...{token.token_address.substring(token.token_address.length - 4)}
                                    </div>
                                </td>
                                <td className="px-6 py-4 font-mono text-slate-300 font-medium">
                                    {formatPrice(token.price_usd)}
                                </td>
                                <td className="px-6 py-4">
                                    <div className={`flex items-center font-bold ${token.price_change_5m >= 0 ? 'text-emerald-400' : 'text-rose-500'}`}>
                                        {token.price_change_5m >= 0 ? <ArrowUpRight className="h-4 w-4 mr-1" /> : <ArrowDownRight className="h-4 w-4 mr-1" />}
                                        {Math.abs(token.price_change_5m).toFixed(2)}%
                                    </div>
                                </td>
                                <td className="px-6 py-4">
                                    <div className={`flex items-center font-medium ${token.price_change_1h >= 0 ? 'text-emerald-400/80' : 'text-rose-500/80'}`}>
                                        {token.price_change_1h.toFixed(2)}%
                                    </div>
                                </td>
                                <td className="px-6 py-4 text-slate-300 font-mono text-xs">
                                    {formatNumber(token.liquidity_usd)}
                                </td>
                                <td className="px-6 py-4 text-slate-300 font-mono text-xs">
                                    {formatNumber(token.market_cap)}
                                </td>
                                <td className="px-6 py-4">
                                    <div className="flex items-center space-x-3">
                                        <div className="relative">
                                            <svg className="h-10 w-10 -rotate-90" viewBox="0 0 36 36">
                                                <path
                                                    className="text-slate-800"
                                                    d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                                                    fill="none"
                                                    stroke="currentColor"
                                                    strokeWidth="4"
                                                />
                                                <path
                                                    className={`${token.prediction?.confidence > 80 ? "text-emerald-400" : token.prediction?.confidence > 60 ? "text-yellow-400" : "text-slate-500"}`}
                                                    d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                                                    fill="none"
                                                    stroke="currentColor"
                                                    strokeWidth="4"
                                                    strokeDasharray={`${token.prediction?.confidence}, 100`}
                                                />
                                            </svg>
                                            <div className="absolute top-0 left-0 w-full h-full flex items-center justify-center text-[10px] font-bold">
                                                {token.prediction?.confidence}
                                            </div>
                                        </div>

                                        {token.status === "PUMP" && <span className="inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-bold bg-emerald-500/20 text-emerald-400 border border-emerald-500/30 animate-pulse">PUMP</span>}
                                        {token.status === "CRASH" && <span className="inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-bold bg-rose-500/20 text-rose-400 border border-rose-500/30">CRASH</span>}
                                    </div>
                                </td>
                                <td className="px-6 py-4 text-right">
                                    <Button
                                        size="sm"
                                        className="h-8 w-8 p-0 bg-transparent hover:bg-cyan-500/20 text-slate-400 hover:text-cyan-400 rounded-full transition-all"
                                        onClick={(e) => {
                                            e.stopPropagation();
                                            window.open(`https://dexscreener.com/${token.chain.toLowerCase()}/${token.token_address}`, '_blank');
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
