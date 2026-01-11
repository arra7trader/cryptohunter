
"use client";

import { useEffect, useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ArrowUpRight, ArrowDownRight, RefreshCw, Crosshair, ExternalLink } from "lucide-react";

export default function TokenTable({ tokens, onSelectToken }) {
    if (!tokens || tokens.length === 0) {
        return (
            <div className="flex flex-col items-center justify-center h-64 text-muted-foreground p-8 border rounded-lg border-dashed border-slate-700 bg-slate-900/50">
                <RefreshCw className="h-8 w-8 mb-4 animate-spin" />
                <p>Scanning DexScreener...</p>
            </div>
        );
    }

    const formatPrice = (price) => {
        if (price > 1) return `$${price.toFixed(4)}`;
        if (price > 0.0001) return `$${price.toFixed(6)}`;
        return `$${price.toFixed(10)}`;
    };

    const formatNumber = (num) => {
        if (num >= 1000000) return `$${(num / 1000000).toFixed(2)}M`;
        if (num >= 1000) return `$${(num / 1000).toFixed(1)}K`;
        return `$${num.toFixed(0)}`;
    };

    return (
        <div className="rounded-md border border-slate-800 bg-slate-950/50 overflow-hidden">
            <div className="overflow-x-auto">
                <table className="w-full text-sm text-left">
                    <thead className="bg-slate-900/80 text-slate-400 uppercase text-xs font-semibold">
                        <tr>
                            <th className="px-4 py-3">Token</th>
                            <th className="px-4 py-3">Price</th>
                            <th className="px-4 py-3">5m</th>
                            <th className="px-4 py-3">1h</th>
                            <th className="px-4 py-3">Liquidity</th>
                            <th className="px-4 py-3">M. Cap</th>
                            <th className="px-4 py-3">AI Conf.</th>
                            <th className="px-4 py-3 text-right">Action</th>
                        </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-800">
                        {tokens.map((token, idx) => (
                            <tr
                                key={idx}
                                className="hover:bg-slate-900/60 transition-colors cursor-pointer group"
                                onClick={() => onSelectToken(token)}
                            >
                                <td className="px-4 py-3">
                                    <div className="flex items-center space-x-2">
                                        <div className="font-bold text-white">{token.token}</div>
                                        <Badge variant="outline" className="text-[10px] h-5 px-1 py-0 border-slate-700 text-slate-500">
                                            {token.chain}
                                        </Badge>
                                    </div>
                                </td>
                                <td className="px-4 py-3 font-mono text-slate-300">
                                    {formatPrice(token.price_usd)}
                                </td>
                                <td className="px-4 py-3">
                                    <div className={`flex items-center ${token.price_change_5m >= 0 ? 'text-green-500' : 'text-red-500'}`}>
                                        {token.price_change_5m >= 0 ? <ArrowUpRight className="h-3 w-3 mr-1" /> : <ArrowDownRight className="h-3 w-3 mr-1" />}
                                        {Math.abs(token.price_change_5m).toFixed(2)}%
                                    </div>
                                </td>
                                <td className="px-4 py-3">
                                    <div className={`flex items-center ${token.price_change_1h >= 0 ? 'text-green-500' : 'text-red-500'}`}>
                                        {token.price_change_1h.toFixed(2)}%
                                    </div>
                                </td>
                                <td className="px-4 py-3 text-slate-400 font-mono text-xs">
                                    {formatNumber(token.liquidity_usd)}
                                </td>
                                <td className="px-4 py-3 text-slate-400 font-mono text-xs">
                                    {formatNumber(token.market_cap)}
                                </td>
                                <td className="px-4 py-3">
                                    <div className="flex items-center space-x-2">
                                        <span
                                            className={`font-bold ${token.prediction?.confidence > 80 ? "text-green-400" :
                                                    token.prediction?.confidence > 60 ? "text-yellow-400" : "text-slate-500"
                                                }`}
                                        >
                                            {token.prediction?.confidence}%
                                        </span>
                                        {token.status === "PUMP" && <Badge variant="success" className="text-[10px] py-0 h-5">PUMP</Badge>}
                                        {token.status === "CRASH" && <Badge variant="danger" className="text-[10px] py-0 h-5">CRASH</Badge>}
                                    </div>
                                </td>
                                <td className="px-4 py-3 text-right">
                                    <Button
                                        size="sm"
                                        variant="ghost"
                                        className="h-7 w-7 p-0 hover:bg-slate-800"
                                        onClick={(e) => {
                                            e.stopPropagation();
                                            window.open(`https://dexscreener.com/${token.chain.toLowerCase()}/${token.token_address}`, '_blank');
                                        }}
                                    >
                                        <ExternalLink className="h-4 w-4 text-slate-500 group-hover:text-blue-400" />
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
