"use client";

import dynamic from "next/dynamic";
import { RefreshCw } from "lucide-react";

// Dynamic import with SSR disabled - this completely prevents hydration mismatch
const Dashboard = dynamic(() => import("./Dashboard"), {
  ssr: false,
  loading: () => (
    <main className="min-h-screen bg-gradient-to-b from-slate-900 via-slate-900 to-black flex items-center justify-center">
      <div className="text-center">
        <RefreshCw className="h-10 w-10 text-cyan-400 animate-spin mx-auto mb-4" />
        <p className="text-slate-400">Loading CryptoHunter AI...</p>
      </div>
    </main>
  ),
});

export default function Home() {
  return <Dashboard />;
}
