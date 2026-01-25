import React from 'react';
import { type Model } from '../../api';
// import { Activity, Cpu, DollarSign } from 'lucide-react';

interface ModelStatusProps {
    model?: Model;
    usage?: {
        prompt_tokens: number;
        completion_tokens: number;
        total_tokens: number;
    };
}

export const ModelStatus: React.FC<ModelStatusProps> = ({ model, usage }) => {
    if (!model) return <div className="text-[10px] text-slate-500 italic">No model active</div>;
    // Pricing is usually per 1M tokens or similar, need to normalize based on typical API (usually raw number from API is unit price? No, usually per 1M)
    // Check OpenRouter/Ollama. Assuming the parsed numbers are per-token or per-1k/1m.
    // Let's assume the pricing string is parsed into a strictly numeric "per token" value by backend LLM service?
    // Actually api.ts says pricing properties are strings.
    // Let's just display tokens for now to be safe, or try a naive calc if formatted.
    // If the string is "$0.5/M", parsing it is complex.
    // Let's stick to displaying the strings if we can't parse, or just tokens.

    // BETTER: Just display the token counts and the pricing rate.
    // BETTER: Just display the token counts and the pricing rate.

    return (
        <div className="bg-slate-900/50 rounded border border-slate-800 p-3 space-y-2">
            <div className="flex items-start justify-between">
                <div>
                    <div className="text-xs font-bold text-slate-200">{model.name}</div>
                    <div className="text-[10px] text-slate-500 uppercase tracking-wider">{model.provider || 'Unknown Provider'}</div>
                </div>
                {/* <div className="p-1 bg-slate-800 rounded text-slate-400">
                    <CPU size={12} />
                </div> */}
            </div>

            {/* Stats Grid */}
            <div className="grid grid-cols-2 gap-2 mt-2">
                <div className="bg-slate-950/50 p-1.5 rounded border border-slate-800/50">
                    <div className="text-[9px] text-slate-500 uppercase font-bold mb-0.5">Input</div>
                    <div className="text-xs font-mono text-blue-400">{usage?.prompt_tokens || 0}</div>
                </div>
                <div className="bg-slate-950/50 p-1.5 rounded border border-slate-800/50">
                    <div className="text-[9px] text-slate-500 uppercase font-bold mb-0.5">Output</div>
                    <div className="text-xs font-mono text-purple-400">{usage?.completion_tokens || 0}</div>
                </div>
            </div>

            {/* Pricing Rate Info */}
            {model.pricing && (
                <div className="pt-2 border-t border-slate-800/50 text-[9px] text-slate-500 flex justify-between">
                    <span>In: {model.pricing.prompt}</span>
                    <span>Out: {model.pricing.completion}</span>
                </div>
            )}
        </div>
    );
};
