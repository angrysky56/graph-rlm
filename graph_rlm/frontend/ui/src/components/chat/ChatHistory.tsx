
import React, { useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import { User, Bot, AlertTriangle, CheckCircle, Terminal, Trash2 } from 'lucide-react';
import { api } from '../../api';

interface ChatEntry {
    id?: string;
    type: 'input' | 'output' | 'info' | 'error';
    content: string;
    timestamp: number;
    style?: 'code' | 'thinking' | 'trace' | 'success' | 'error' | 'report' | 'monitor';
    isStreaming?: boolean;
    role?: string; // Sometimes used for explicit role
    repl_id?: string;
    metrics?: any;
}

interface ChatHistoryProps {
    entries: ChatEntry[];
    onDelete?: (id: string) => void;
}

export const ChatHistory: React.FC<ChatHistoryProps> = ({ entries, onDelete }) => {
    const bottomRef = useRef<HTMLDivElement>(null);

    const displayEntries = entries.filter(e => {
        // Essential Filters
        const isSubstantive = e.style === 'report' || e.style === 'success' || e.style === 'thinking';
        const isUser = e.type === 'input';
        const isStandardOutput = e.type === 'output' && !e.style;
        const isMainError = e.type === 'error' && e.style !== 'trace';
        const isMonitor = e.style === 'monitor';

        return isUser || isSubstantive || isMainError || isStandardOutput || isMonitor;
    });

    useEffect(() => {
        bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [displayEntries.length, displayEntries[displayEntries.length-1]?.content]);

    return (
        <div className="flex-1 overflow-y-auto px-4 py-6 space-y-6 custom-scrollbar min-w-0">
            {displayEntries.length === 0 && (
                <div className="flex flex-col items-center justify-center h-full text-slate-600 opacity-50">
                    <Bot size={48} className="mb-4" />
                    <p>Start a conversation to begin...</p>
                </div>
            )}

            {displayEntries.map((entry, i) => {
                const isUser = entry.type === 'input';
                const isError = entry.type === 'error' || entry.style === 'error';
                const isCode = entry.style === 'code';
                const isSuccess = entry.style === 'success';
                const isStandardOutput = entry.type === 'output' && !entry.style;

                return (
                    // Added min-w-0 to prevent flex item expansion causing overflow
                    <div key={i} className={`flex gap-4 ${isUser ? 'flex-row-reverse' : ''} w-full max-w-4xl mx-auto min-w-0`}>
                        {/* Avatar */}
                        <div className={`mt-1 shrink-0 w-8 h-8 rounded-full flex items-center justify-center ${
                            isUser ? 'bg-blue-600 text-white' :
                            isError ? 'bg-red-900/50 text-red-500' :
                            isSuccess ? 'bg-emerald-900/50 text-emerald-500' :
                            'bg-slate-800 text-slate-400'
                        }`}>
                            {isUser ? <User size={16} /> :
                             isError ? <AlertTriangle size={16} /> :
                             isSuccess ? <CheckCircle size={16} /> :
                             isCode ? <Terminal size={14} /> :
                             <Bot size={16} />}
                        </div>

                        {/* Message Bubble - Added max-w-full and break-words */}
                        <div className={`flex-1 min-w-0 max-w-full ${isUser ? 'text-right' : ''}`}>
                            <div className={`inline-block text-left rounded-2xl px-5 py-3 text-[13px] leading-relaxed shadow-sm max-w-full ${
                                isUser ? 'bg-indigo-600 text-white rounded-tr-sm' :
                                isError ? 'bg-red-950/20 border border-red-900/30 text-red-200 rounded-tl-sm w-full' :
                                isStandardOutput ? 'bg-slate-800/40 border border-slate-700/50 text-slate-300 rounded-tl-sm w-full' :
                                entry.style === 'monitor' ? 'bg-blue-950/20 border border-blue-900/30 text-blue-200 rounded-tl-sm w-full font-mono text-[12px]' :
                                'bg-slate-800/40 border border-slate-700/50 text-slate-300 rounded-tl-sm w-full'
                            }`}>
                                {entry.style === 'monitor' ? (
                                    <div className="space-y-2">
                                        <div className="flex items-center gap-2 text-blue-400 font-bold text-[10px] uppercase tracking-widest">
                                            <Terminal size={12} />
                                            <span>System Telemetry</span>
                                        </div>
                                        <div className="opacity-90">{entry.content}</div>
                                        {entry.metrics && (
                                            <div className="mt-2 pt-2 border-t border-blue-900/30 text-[11px] opacity-70 whitespace-pre-wrap">
                                                {JSON.stringify(entry.metrics, null, 2)}
                                            </div>
                                        )}
                                    </div>
                                ) : isCode ? (
                                    <details className="group">
                                        <summary className="cursor-pointer font-bold select-none outline-none text-[11px] uppercase tracking-wider text-slate-400 hover:text-slate-200 transition-colors flex items-center gap-2">
                                            <span className="group-open:rotate-90 transition-transform">▶</span>
                                            <span>Code Execution {entry.repl_id ? `(REPL: ${entry.repl_id})` : ''}</span>
                                        </summary>
                                        <div className="mt-3 pt-3 border-t border-slate-800/50">
                                            <pre className="whitespace-pre-wrap break-words max-w-full overflow-x-auto text-slate-300">{entry.content}</pre>
                                        </div>
                                    </details>
                                ) : (
                                    <div className={`markdown-content break-words max-w-full ${isUser ? 'text-white' : 'text-slate-200'}`}>
                                       <ReactMarkdown
                                            components={{
                                                code({node, inline, className, children, ...props}: any) {
                                                    const match = /language-(\w+)/.exec(className || '');
                                                    const isCodeBlock = !inline && match;

                                                    if (isCodeBlock) {
                                                        return (
                                                            <details className="group mt-4 mb-4 border border-slate-700/50 rounded-md overflow-hidden bg-[#0d1117]">
                                                                <summary className="cursor-pointer font-bold select-none outline-none text-[10px] uppercase tracking-wider text-slate-400 hover:text-slate-200 transition-colors bg-slate-800/40 px-4 py-2.5 flex items-center gap-2 border-b border-slate-700/50">
                                                                    <span className="group-open:rotate-90 transition-transform">▶</span>
                                                                    <span>Generated Code ({match[1]})</span>
                                                                </summary>
                                                                <div className="p-4 overflow-x-auto text-[12px] leading-relaxed">
                                                                    <code className={className} {...props}>
                                                                        {children}
                                                                    </code>
                                                                </div>
                                                            </details>
                                                        );
                                                    }

                                                    return (
                                                        <code className={`${isUser ? 'bg-indigo-700/50 text-indigo-100' : 'bg-slate-900/50 text-indigo-300 border border-slate-700/30'} px-1.5 py-0.5 rounded-md font-mono text-[12px]`} {...props}>
                                                            {children}
                                                        </code>
                                                    );
                                                }
                                            }}
                                       >
                                           {entry.content}
                                       </ReactMarkdown>
                                    </div>
                                )}
                            </div>
                            <div className={`mt-1.5 text-[10px] font-medium tracking-wide text-slate-500 ${isUser ? 'text-right' : 'text-left'} flex items-center gap-3 ${isUser ? 'justify-end' : 'justify-start'}`}>
                                <span>{new Date(entry.timestamp).toLocaleTimeString()}</span>
                                {entry.id && onDelete && (
                                    <button
                                        onClick={() => {
                                            if (confirm("Delete this thought node?")) {
                                                onDelete(entry.id!);
                                            }
                                        }}
                                        className="text-slate-700 hover:text-red-400 p-1 opacity-0 group-hover:opacity-100 transition-opacity"
                                        title="Delete Message"
                                    >
                                        <Trash2 size={10} />
                                    </button>
                                )}
                            </div>
                        </div>
                    </div>
                );
            })}
            <div ref={bottomRef} />
        </div>
    );
};
