"use client";

import { Search, X } from "lucide-react";

export default function SearchBar({ value, onChange }) {
    return (
        <div className="relative w-full sm:w-80">
            <div className="absolute inset-y-0 left-0 flex items-center pl-3 pointer-events-none">
                <Search className="h-4 w-4 text-slate-500" />
            </div>
            <input
                type="text"
                placeholder="Search tokens..."
                value={value}
                onChange={(e) => onChange(e.target.value)}
                className="w-full h-10 pl-10 pr-10 bg-slate-800/50 border border-slate-700 rounded-xl 
                           text-sm text-slate-200 placeholder:text-slate-500
                           focus:outline-none focus:border-cyan-500/50 focus:ring-1 focus:ring-cyan-500/20
                           transition-all"
            />
            {value && (
                <button
                    onClick={() => onChange("")}
                    className="absolute inset-y-0 right-0 flex items-center pr-3 text-slate-500 hover:text-slate-300"
                >
                    <X className="h-4 w-4" />
                </button>
            )}
        </div>
    );
}
