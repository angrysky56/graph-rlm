import React, { useState } from 'react';
import { Terminal, ChevronDown, ChevronRight, Check, X } from 'lucide-react';

interface ToolBlockProps {
    toolName: string;
    args: any;
    result?: string;
}

export const ToolBlock: React.FC<ToolBlockProps> = ({ toolName, args, result }) => {
    const [isOpen, setIsOpen] = useState(false);

    return (
        <div className="my-2 border border-slate-700 bg-slate-900/50 rounded-md overflow-hidden font-mono text-xs">
            <button
                onClick={() => setIsOpen(!isOpen)}
                className="w-full flex items-center gap-2 px-3 py-2 bg-slate-800/50 text-slate-300 hover:bg-slate-800 transition-colors"
            >
                <Terminal size={12} className="text-amber-500" />
                <span className="font-bold text-amber-500">{toolName}</span>
                <span className="opacity-50 truncate max-w-[200px]">{JSON.stringify(args)}</span>
                <span className="ml-auto opacity-50">
                    {isOpen ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
                </span>
            </button>

            {isOpen && (
                <div className="p-3 border-t border-slate-700/50 bg-black/20">
                    <div className="mb-2">
                        <span className="text-slate-500 block mb-1 uppercase tracking-wider text-[10px]">Arguments</span>
                        <pre className="bg-black/30 p-2 rounded text-slate-300 overflow-x-auto border border-slate-800">
                            {JSON.stringify(args, null, 2)}
                        </pre>
                    </div>
                    {result && (
                        <div>
                            <span className="text-slate-500 block mb-1 uppercase tracking-wider text-[10px] flex items-center gap-1">
                                {result.startsWith("Error") ? <X size={10} className="text-red-500" /> : <Check size={10} className="text-green-500" />}
                                Result
                            </span>
                            <pre className={`p-2 rounded overflow-x-auto border ${result.startsWith("Error") ? 'bg-red-900/10 text-red-300 border-red-900/30' : 'bg-green-900/10 text-green-300 border-green-900/30'}`}>
                                {result}
                            </pre>
                        </div>
                    )}
                </div>
            )}
        </div>
    );
};
