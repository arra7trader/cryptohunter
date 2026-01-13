"use client";

import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Zap, Loader2, Crosshair, Brain, Target, Clock, TrendingUp, ExternalLink } from "lucide-react";

export default function AIAnalysisPanel({ selectedToken, apiBase }) {
    const [analysisLoading, setAnalysisLoading] = useState(false);
    const [detailedPrediction, setDetailedPrediction] = useState(null);

    const runDeepAnalysis = async () => {
        if (!selectedToken) return;

        try {
            setAnalysisLoading(true);
            // Use the new Enhanced Predictor Endpoint
            // Using symbol (e.g. BTC) instead of token_address for enhanced prediction as it relies on symbol-aggregated data
            // If token has no symbol map, it might fail or fallback.
            const symbol = selectedToken.token || "BTC";
            const res = await fetch(
                `http://localhost:8000/api/predict/enhanced/${symbol}`
            );

            if (res.ok) {
                const data = await res.json();
                setDetailedPrediction(data);
            } else {
                const errorData = await res.json().catch(() => ({ detail: "Unknown Error" }));
                console.error("Analysis failed:", errorData);
                // alert(`Analysis failed: ${errorData.detail || "Server Error"}`); 
                // Don't alert blocking, just log
            }
        } catch (error) {
            console.error("Deep analysis error:", error);
            // alert("Network Error: Could not reach backend.");
        } finally {
            setAnalysisLoading(false);
        }
    };

    const getPrediction = () => detailedPrediction || selectedToken?.prediction;

    if (!selectedToken) {
        return (
            <Card className="min-h-[500px] glass-card border-0 relative overflow-hidden">
                <div className="absolute top-0 right-0 w-40 h-40 bg-purple-500/5 blur-[60px] rounded-full"></div>
                <div className="absolute bottom-0 left-0 w-40 h-40 bg-cyan-500/5 blur-[60px] rounded-full"></div>

                <div className="h-full flex flex-col items-center justify-center p-8 text-center space-y-6 min-h-[400px]">
                    <div className="relative">
                        <div className="absolute inset-0 bg-gradient-to-r from-cyan-500/20 to-purple-500/20 blur-xl rounded-full animate-pulse"></div>
                        <div className="relative p-6 bg-slate-900/80 rounded-2xl border border-white/10">
                            <Crosshair className="h-16 w-16 text-slate-600" />
                        </div>
                    </div>
                    <div className="space-y-2">
                        <h3 className="text-xl font-bold text-slate-200">Sentinel AI Ready</h3>
                        <p className="text-sm text-slate-400 max-w-[280px] mx-auto leading-relaxed">
                            Select a token from the scanner to activate <br />
                            <span className="text-cyan-400 font-bold">Enhanced AI + Sentiment Analysis</span>.
                        </p>
                    </div>
                    <div className="flex items-center gap-2 text-xs text-slate-500">
                        <Brain className="h-4 w-4" />
                        <span>Powered by LSTM + Binance Data + Fear & Greed</span>
                    </div>
                </div>
            </Card>
        );
    }

    const prediction = getPrediction();
    const confidence = prediction?.confidence || 0;

    // Check if we have enhanced data
    const isEnhanced = !!prediction?.sentiment_adjustment;

    return (
        <Card className="min-h-[500px] glass-card border-0 relative overflow-hidden">
            {/* Background decorations */}
            <div className="absolute top-0 right-0 w-40 h-40 bg-purple-500/10 blur-[60px] rounded-full pointer-events-none"></div>
            <div className="absolute bottom-0 left-0 w-40 h-40 bg-cyan-500/10 blur-[60px] rounded-full pointer-events-none"></div>

            <CardHeader className="relative z-10 border-b border-white/5 pb-4">
                <div className="flex items-start justify-between">
                    <div>
                        <div className="flex items-center gap-2 mb-2">
                            <Badge className="badge-info">{selectedToken.chain}</Badge>
                            <span className="text-[10px] text-slate-500 font-mono">
                                {selectedToken.token_address?.substring(0, 8)}...
                            </span>
                        </div>
                        <CardTitle className="text-3xl font-black text-gradient">
                            {selectedToken.token}
                        </CardTitle>
                    </div>
                    <div className="text-right">
                        <div className="text-[10px] text-purple-400 font-bold uppercase tracking-wider mb-1">
                            AI Confidence
                        </div>
                        <div className={`text-4xl font-black ${confidence >= 75 ? 'text-emerald-400' :
                                confidence >= 50 ? 'text-yellow-400' : 'text-slate-400'
                            }`}>
                            {confidence ? confidence.toFixed(1) : 0}%
                        </div>
                    </div>
                </div>
            </CardHeader>

            <CardContent className="space-y-5 pt-5 relative z-10">
                {/* Run Analysis Button */}
                <Button
                    onClick={runDeepAnalysis}
                    disabled={analysisLoading}
                    className="w-full h-12 bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-500 hover:to-indigo-500 
                               border border-purple-500/30 font-bold tracking-wide text-base 
                               shadow-[0_0_30px_rgba(139,92,246,0.3)] hover:shadow-[0_0_40px_rgba(139,92,246,0.5)] transition-all"
                >
                    {analysisLoading ? (
                        <>
                            <Loader2 className="mr-2 h-5 w-5 animate-spin" />
                            ANALYZING MARKET DATA...
                        </>
                    ) : (
                        <>
                            <Zap className="mr-2 h-5 w-5 fill-yellow-400 text-yellow-100" />
                            RUN ENHANCED ANALYSIS
                        </>
                    )}
                </Button>

                {/* Prediction Result */}
                <div className="bg-gradient-to-br from-slate-900 to-slate-900/50 rounded-xl p-1 border border-white/10">
                    <div className="bg-white/5 rounded-lg p-4">
                        <div className="grid grid-cols-2 gap-4">
                            <div className="text-center border-r border-white/10 pr-4">
                                <div className="flex items-center justify-center gap-1 text-[10px] text-slate-500 uppercase tracking-wider mb-2">
                                    <Target className="h-3 w-3" />
                                    Signal
                                </div>
                                <div className={`text-xl font-bold ${confidence >= 75 ? 'text-emerald-400' :
                                        confidence >= 50 ? 'text-yellow-400' : 'text-slate-400'
                                    }`}>
                                    {prediction?.adjusted_signal || prediction?.signal || "WAIT"}
                                </div>
                            </div>
                            <div className="text-center pl-4">
                                <div className="flex items-center justify-center gap-1 text-[10px] text-slate-500 uppercase tracking-wider mb-2">
                                    <Clock className="h-3 w-3" />
                                    Forecast (24h)
                                </div>
                                <div className="text-xl font-bold text-cyan-300">
                                    {prediction?.predicted_price_24h ?
                                        `$${prediction.predicted_price_24h < 0.01 ? prediction.predicted_price_24h.toFixed(6) : prediction.predicted_price_24h.toFixed(2)}`
                                        : '...'}
                                </div>
                            </div>
                        </div>
                    </div>
                    {isEnhanced && (
                        <div className="px-4 py-2 bg-purple-900/20 text-[10px] text-center text-purple-300 font-mono border-t border-purple-500/20 rounded-b-lg">
                            Adjusted by: {prediction.sentiment_adjustment}
                        </div>
                    )}
                </div>

                {/* Enhanced Stats Grid */}
                <div className="grid grid-cols-2 gap-3">
                    <div className="stat-card">
                        <span className="text-slate-400 text-xs text-center block mb-1">Sentiment</span>
                        <span className={`font-mono font-bold text-center block text-sm ${prediction?.fear_greed_classification?.includes('Fear') ? 'text-rose-400' : 'text-emerald-400'
                            }`}>
                            {prediction?.fear_greed_classification || "N/A"} ({prediction?.fear_greed_value || 0})
                        </span>
                    </div>
                    <div className="stat-card">
                        <span className="text-slate-400 text-xs text-center block mb-1">Binance Vol (24h)</span>
                        <span className="font-mono text-white font-bold text-center block text-sm">
                            ${prediction?.binance_volume_24h ? (prediction.binance_volume_24h / 1000000).toFixed(2) + 'M' : "N/A"}
                        </span>
                    </div>
                    <div className="stat-card">
                        <span className="text-slate-400 text-xs text-center block mb-1">Sources</span>
                        <span className="font-mono text-blue-300 font-bold text-center block text-xs truncate">
                            {prediction?.data_sources ? prediction.data_sources.length : 0} Sources
                        </span>
                    </div>
                    <div className="stat-card">
                        <span className="text-slate-400 text-xs text-center block mb-1">SNA Score</span>
                        <span className="font-mono text-yellow-400 font-bold text-center block text-xl">
                            {(selectedToken.sna_score || 0).toFixed(0)}
                        </span>
                    </div>
                </div>

                {/* View on DexScreener */}
                <Button
                    className="w-full h-11 bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 
                               text-white font-bold tracking-wide rounded-xl shadow-lg shadow-cyan-900/30 
                               border border-white/10 transition-all hover:scale-[1.02]"
                    onClick={() => window.open(
                        `https://dexscreener.com/${selectedToken.chain?.toLowerCase()}/${selectedToken.token_address}`,
                        '_blank'
                    )}
                >
                    <ExternalLink className="mr-2 h-4 w-4" />
                    VIEW ON DEXSCREENER
                </Button>

                {/* Warning */}
                <p className="text-[10px] text-slate-600 text-center">
                    ⚠️ AI predictions are not financial advice. Always DYOR.
                </p>
            </CardContent>
        </Card>
    );
}
