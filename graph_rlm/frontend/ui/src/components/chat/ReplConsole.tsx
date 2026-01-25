import React, { useEffect, useRef } from 'react';


interface ReplEntry {
    type: 'input' | 'output' | 'error' | 'info';
    content: string;
    timestamp: number;
    style?: 'code';
    isStreaming?: boolean;
}

interface ReplConsoleProps {
    entries?: ReplEntry[];
}

export const ReplConsole: React.FC<ReplConsoleProps> = ({ entries = [] }) => {
    const bottomRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [entries]); // Auto-scroll on new entries

    const getEntryStyle = (entry: ReplEntry) => {
        if (entry.type === 'error') return 'text-red-400';
        if (entry.type === 'info') return 'text-slate-500 italic'; // Thinking
        if (entry.type === 'input') return 'text-amber-200 font-bold';
        if (entry.style === 'code') return 'text-emerald-400';
        return 'text-slate-300';
    };

    return (
        <div className="flex flex-col h-full bg-black font-mono text-[10px] p-2 overflow-hidden">
            <div className="text-slate-500 mb-2 uppercase tracking-widest border-b border-slate-800 pb-1 flex justify-between">
                <span>REPL Console</span>
                <span className="text-green-500 flex items-center gap-1">● Live</span>
            </div>

            <div className="flex-1 overflow-y-auto space-y-1 custom-scrollbar">
                {entries.length === 0 && (
                    <div className="text-slate-700 italic">Waiting for execution...</div>
                )}

                {entries.map((entry, idx) => (
                    <div key={idx} className={`flex flex-col ${getEntryStyle(entry)}`}>
                        <div className="flex gap-2 opacity-50 mb-0.5 select-none">
                            <span className="text-blue-500">{entry.type === 'input' ? '>>>' : ''}</span>
                            <span>{new Date(entry.timestamp).toLocaleTimeString()}</span>
                        </div>
                        <div className={`break-words whitespace-pre-wrap pl-4 border-l-2 ${entry.type === 'input' ? 'border-amber-500/50' : 'border-transparent'}`}>
                            {entry.content}
                            {entry.isStreaming && <span className="animate-pulse">_</span>}
                        </div>
                    </div>
                ))}
                <div ref={bottomRef} />
            </div>

            <div className="mt-2 pt-2 border-t border-slate-800 flex gap-2 items-center text-slate-500">
                <span>$</span>
                <span className="animate-pulse bg-slate-500 w-1.5 h-3 inline-block"></span>
            </div>
        </div>
    );
};
